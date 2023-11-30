import os
import torch
import torch.nn.functional as F
from torch.optim import Adam
from rl.sac_utils import soft_update, hard_update
from rl.sac_models import GaussianPolicy, QNetwork, DeterministicPolicy, ActorTimestepPolicy, ActorTimestepStatePolicy, QResNetwork, ResidualBlock, ResNet


class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.new_states = []
        self.logprobs = []
        self.scraps = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []
    
    def is_done(self):
        if len(self.is_terminals) == 0:
            return False
        return self.is_terminals[-1]

    # def is_ready_to_update(self, max_ep_len):
    #     return len(self.states) >= max_ep_len -1 and len(self.actions) == len(self.rewards)

    def append(self, buf_to_append):
        self.actions.extend(buf_to_append.actions)
        self.states.extend(buf_to_append.states)
        self.new_states.extend(buf_to_append.new_states)
        self.logprobs.extend(buf_to_append.logprobs)
        self.scraps.extend(buf_to_append.scraps)
        self.rewards.extend(buf_to_append.rewards)
        self.state_values.extend(buf_to_append.state_values)
        self.is_terminals.extend(buf_to_append.is_terminals)

    def size(self):
        return len(self.states)

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.new_states[:]
        del self.logprobs[:]
        del self.scraps[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]

    def __str__(self) -> str:
        return "RolloutBuffer(actions={}, states={}, logprobs={}, rewards={}, state_values={}, is_terminals={})".format(
            len(self.actions), len(self.states), len(self.logprobs), len(self.rewards), len(self.state_values), len(self.is_terminals)
        )
    
    # print full buffer
    def print_buffer(self):
        print("RolloutBuffer(actions={}, logprobs={}, rewards={}, state_values={}, is_terminals={})".format(
            self.actions, self.logprobs, self.rewards, self.state_values, self.is_terminals
        ))

class SAC(object):
    def __init__(self, num_inputs, action_space, gamma, tau, alpha, init_alpha, policy, target_update_interval, automatic_entropy_tuning, device, hidden_size, actor_lr, critic_lr, alpha_lr, res_net):

        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha

        self.policy_type = policy
        self.target_update_interval = target_update_interval
        self.automatic_entropy_tuning = automatic_entropy_tuning

        self.device = device
        self.res_net = res_net

        self.critic = QNetwork(num_inputs, action_space.shape[0], hidden_size).to(self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=critic_lr)

        self.critic_target = QNetwork(num_inputs, action_space.shape[0], hidden_size).to(self.device)
        hard_update(self.critic_target, self.critic)

        if self.policy_type == "Gaussian":
            # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
            if self.automatic_entropy_tuning is True:
                self.target_entropy = -torch.prod(torch.Tensor(action_space.shape).to(self.device)).item()
                self.log_alpha = torch.log(torch.tensor([init_alpha])).to(self.device).requires_grad_()
                # self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
                self.alpha_optim = Adam([self.log_alpha], lr=alpha_lr)

            self.policy = GaussianPolicy(num_inputs, action_space.shape[0], hidden_size, action_space).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=actor_lr)
        elif self.policy_type == "Mixed":
            if self.automatic_entropy_tuning is True:
                self.target_entropy = -torch.prod(torch.Tensor(action_space.shape).to(self.device)).item()
                self.log_alpha = torch.log(torch.tensor([init_alpha])).to(self.device).requires_grad_()
                # self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
                self.alpha_optim = Adam([self.log_alpha], lr=alpha_lr)

            self.policy = ActorTimestepStatePolicy(num_inputs, hidden_dim=hidden_size, output_dim=4).to(self.devicwe)
            # self.policy = ActorTimestepPolicy(ResidualBlock, [3, 4, 6, 3]).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=actor_lr)
        else:
            self.alpha = 0
            self.automatic_entropy_tuning = False
            self.policy = DeterministicPolicy(num_inputs, action_space.shape[0], hidden_size, action_space).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=actor_lr)

    def select_action(self, state, evaluate=False):
        # state = torch.FloatTensor(state).unsqueeze(0)
        if evaluate is False:
            action, _, _, env_action, _ = self.policy.sample(state)
        else:
            assert False, "env_action corresponds to sampled action, not mean action"
            _, _, action, env_action, _ = self.policy.sample(state)
        return action, env_action

    def update_parameters(self, memory, batch_size, updates):
        info = {}
        # Sample a batch from memory
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory.sample(batch_size=batch_size)
        # print("next_state_batch", next_state_batch.shape)
        # state_batch = torch.FloatTensor(state_batch).to(self.device)
        # next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        # action_batch = torch.FloatTensor(action_batch).to(self.device)
        # reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        # mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)
        reward_batch = reward_batch.unsqueeze(1)
        mask_batch = mask_batch.unsqueeze(1)
        with torch.no_grad():
            next_state_action, next_state_log_pi, _, _, _ = self.policy.sample(next_state_batch)
            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + mask_batch * self.gamma * (min_qf_next_target)
        qf1, qf2 = self.critic(state_batch, action_batch)  # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1_loss = F.mse_loss(qf1, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(qf2, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf_loss = qf1_loss + qf2_loss

        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()

        pi, log_pi, _, _, pi_info = self.policy.sample(state_batch)
        for (k, v) in pi_info.items():
            info['policy_' + k] = v

        qf1_pi, qf2_pi = self.critic(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean() # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
            alpha_tlogs = self.alpha.clone() # For TensorboardX logs
        else:
            alpha_loss = torch.tensor(0.).to(self.device)
            alpha_tlogs = torch.tensor(self.alpha) # For TensorboardX logs


        if updates % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)

        info.update({
            'critic_1_loss': qf1_loss.item(), 
            'critic_2_loss': qf2_loss.item(), 
            'policy_loss': policy_loss.item(), 
            'policy_log_prob_mean': log_pi.mean(dim=0).item(),
            'policy_log_prob_std': log_pi.std(dim=0).item(),
            'min_qf_pi_mean': min_qf_pi.mean(dim=0).item(),
            'min_qf_pi_std': min_qf_pi.std(dim=0).item(),
            'ent_loss': alpha_loss.item(), 
            'alpha': alpha_tlogs.item(),
            'alpha_loss': alpha_loss.item(),
        })
        return info

    # Save model parameters
    def save_checkpoint(self, env_name, policy_name, suffix="", ckpt_path=None):
        if not os.path.exists('checkpoints/'):
            os.makedirs('checkpoints/')
        if ckpt_path is None:
            ckpt_path = "checkpoints/sac_checkpoint_{}_{}_{}".format(env_name, policy_name, suffix)
        print('Saving models to {}'.format(ckpt_path))
        torch.save({'policy_state_dict': self.policy.state_dict(),
                    'critic_state_dict': self.critic.state_dict(),
                    'critic_target_state_dict': self.critic_target.state_dict(),
                    'critic_optimizer_state_dict': self.critic_optim.state_dict(),
                    'policy_optimizer_state_dict': self.policy_optim.state_dict()}, ckpt_path)

    # Load model parameters
    def load_checkpoint(self, ckpt_path, evaluate=False):
        print('Loading models from {}'.format(ckpt_path))
        if ckpt_path is not None:
            checkpoint = torch.load(ckpt_path)
            self.policy.load_state_dict(checkpoint['policy_state_dict'])
            self.critic.load_state_dict(checkpoint['critic_state_dict'])
            self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
            self.critic_optim.load_state_dict(checkpoint['critic_optimizer_state_dict'])
            self.policy_optim.load_state_dict(checkpoint['policy_optimizer_state_dict'])

            if evaluate:
                self.policy.eval()
                self.critic.eval()
                self.critic_target.eval()
            else:
                self.policy.train()
                self.critic.train()
                self.critic_target.train()