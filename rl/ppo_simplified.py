import os
import glob
import time
from datetime import datetime

import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical
import copy

import numpy as np

import gym
from .models import ResidualBlock, ResNet, ActorNet 
from matplotlib import pyplot as plt
import time
# import roboschool
# import pybullet_envs

DEBUG = True

class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
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
        self.logprobs.extend(buf_to_append.logprobs)
        self.rewards.extend(buf_to_append.rewards)
        self.state_values.extend(buf_to_append.state_values)
        self.is_terminals.extend(buf_to_append.is_terminals)

    def size(self):
        return len(self.states)

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
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


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, has_continuous_action_space, action_std_init, device='cpu', res_net=True):
        super(ActorCritic, self).__init__()

        self.has_continuous_action_space = has_continuous_action_space
        self.device = device
        if has_continuous_action_space:
            self.action_dim = action_dim
            # -1 because only the first dim is continuous
            self.action_var = torch.full((action_dim - 1,), action_std_init * action_std_init).to(self.device)

        # actor
        if has_continuous_action_space :
            if res_net:
                # # randomly initialize the weights
                # def init_weights(m, init_type='normal'):
                #     if hasattr(m, 'weight') and (m.__class__.__name__.find('Conv') != -1 or m.__class__.__name__.find('Linear') != -1):
                #         if init_type == 'normal':
                #             nn.init.normal_(m.weight.data, 0.0, 0.02)
                #         elif init_type == 'xavier':
                #             nn.init.xavier_normal_(m.weight.data, gain=0.02)
                #         elif init_type == 'kaiming':
                #             nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                #         elif init_type == 'orthogonal':
                #             nn.init.orthogonal_(m.weight.data, gain=0.02)
                #         else:
                #             raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
                #         if hasattr(m, 'bias') and m.bias is not None:
                #             nn.init.constant_(m.bias.data, 0.0)
                #     elif hasattr(m, 'weight') and (m.__class__.__name__.find('BatchNorm') != -1):
                #         nn.init.normal_(m.weight.data, 1.0, 0.02)
                #         nn.init.constant_(m.bias.data, 0.0)
                        
                # self._init_weights = lambda m: init_weights(m, init_type='normal')
                self.actor = ActorNet(ResidualBlock, [3, 4, 6, 3], num_classes=self.action_dim)
                # self.actor.apply(self._init_weights)
            else:
                self.actor = nn.Sequential(
                                nn.Linear(state_dim, 64),
                                nn.Tanh(),
                                nn.Linear(64, 64),
                                nn.Tanh(),
                                nn.Linear(64, 6),
                            )

        else:
            self.actor = nn.Sequential(
                            nn.Linear(state_dim, 64),
                            nn.Tanh(),
                            nn.Linear(64, 64),
                            nn.Tanh(),
                            nn.Linear(64, action_dim),
                            nn.Softmax(dim=-1)
                        )

        
        # critic
        if res_net:
            self.critic = ResNet(ResidualBlock, [3, 4, 6, 3], num_classes=1)
        else:
            self.critic = nn.Sequential(
                            nn.Linear(state_dim, 64),
                            nn.Tanh(),
                            nn.Linear(64, 64),
                            nn.Tanh(),
                            nn.Linear(64, 1)
                        )
        
        
    def set_action_std(self, new_action_std):

        if self.has_continuous_action_space:
            self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(self.device)
        else:
            print("Calling ActorCritic::set_action_std() on discrete action space policy")


    def forward(self):
        raise NotImplementedError

    def act(self, state):
        # first value of action is categorical, second value is continuous
        action = self.actor(state)
        

        # if self.has_continuous_action_space:
        action_mean = action[:, 6:]
        cov_mat = torch.diag(self.action_var).unsqueeze(dim=0).to(self.device)
        dist_1 = MultivariateNormal(action_mean, cov_mat)
        # else:
        action_probs = action[:, :6]
        dist_2 = Categorical(action_probs)

        # print("dist 1 sample", dist_1.sample())
        # print("dist 2 sample", dist_2.sample())
        action = torch.cat((dist_2.sample().unsqueeze(dim=-1), dist_1.sample()), dim=-1)
        action_logprob_1 = dist_1.log_prob(action[:, 1:])
        action_logprob_2 = dist_2.log_prob(action[:, 0])
        action_logprob = torch.sum(action_logprob_1, dim=-1) + action_logprob_2
        state_val = self.critic(state)

        # print("action", action)

        return action.detach(), action_logprob.detach(), state_val.detach()

        # action = torch.clamp(dist.sample(), min=0.0, max=6.0)
        # action_logprob = dist.log_prob(action)
        # state_val = self.critic(state)

        # return action.detach(), action_logprob.detach(), state_val.detach()
    

    def evaluate(self, state, action):

        if self.has_continuous_action_space:
            # print("State is nan", torch.isnan(state).any())
            action_mean = self.actor(state)
            action_var = self.action_var.expand_as(action_mean)
            cov_mat = torch.diag_embed(action_var).to(self.device)
            # print("action_mean", action_mean)
            # print("cov_mat", cov_mat)
            dist = MultivariateNormal(action_mean, cov_mat)
            
            # for single action continuous environments
            if self.action_dim == 1:
                action = action.reshape(-1, self.action_dim)

        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)
        
        return action_logprobs, state_values, dist_entropy

class PPO:
    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std_init=0.6, device='cpu', res_net=True):

        self.has_continuous_action_space = has_continuous_action_space

        if has_continuous_action_space:
            self.action_std = action_std_init

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        
        self.buffer = RolloutBuffer()

        self.device = device

        self.policy = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init, device=device, res_net=res_net).to(self.device)
        self.optimizer = torch.optim.Adam([
                        {'params': self.policy.actor.parameters(), 'lr': lr_actor},
                        {'params': self.policy.critic.parameters(), 'lr': lr_critic}
                    ])

        self.policy_old = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init, device=device, res_net=res_net).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()


    def set_action_std(self, new_action_std):
        
        if self.has_continuous_action_space:
            self.action_std = new_action_std
            self.policy.set_action_std(new_action_std)
            self.policy_old.set_action_std(new_action_std)
        
        else:
            print("WARNING : Calling PPO::set_action_std() on discrete action space policy")


    def decay_action_std(self, action_std_decay_rate, min_action_std):

        if self.has_continuous_action_space:
            self.action_std = self.action_std - action_std_decay_rate
            self.action_std = round(self.action_std, 4)
            if (self.action_std <= min_action_std):
                self.action_std = min_action_std
                print("setting actor output action_std to min_action_std : ", self.action_std)
            else:
                print("setting actor output action_std to : ", self.action_std)
            self.set_action_std(self.action_std)

        else:
            print("WARNING : Calling PPO::decay_action_std() on discrete action space policy")

        print("--------------------------------------------------------------------------------------------")


    def select_action(self, state):
        
        if self.has_continuous_action_space:
            with torch.no_grad():
                #state = torch.FloatTensor(state)#.to(self.device)
                action, action_logprob, state_val = self.policy_old.act(state)
            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)
            self.buffer.state_values.append(state_val)
            return action.detach(), action_logprob.detach(), state_val.detach()


    def update(self, buffer=None):
        t0 = time.time()
        # buffer = self.buffer
        if (buffer is not None):
            self.buffer = copy.deepcopy(buffer)
        # self.buffer.print_buffer()
        t1 = time.time()
        print('copy time', t1 - t0)
        if DEBUG:
            # find nans in buffer state, action, logprobs, state_values
            for i in range(len(self.buffer.states)):
                if torch.isnan(self.buffer.states[i]).any():
                    print("nan in buffer.states[{}]".format(i))
                if torch.isnan(self.buffer.actions[i]).any():
                    print("nan in buffer.actions[{}]".format(i))
                if torch.isnan(self.buffer.logprobs[i]).any():
                    print("nan in buffer.logprobs[{}]".format(i))
                if torch.isnan(self.buffer.state_values[i]).any():
                    print("nan in buffer.state_values[{}]".format(i))

        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        t2 = time.time()
        print('Monte Carlo estimate of returns time', t2 - t1)
        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor

        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(self.device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(self.device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(self.device)
        old_state_values = torch.squeeze(torch.stack(self.buffer.state_values, dim=0)).detach().to(self.device)

        # calculate advantages
        advantages = rewards.detach() - old_state_values.detach()
        
        # Optimize policy for K epochs
        for _ in range(self.K_epochs):

            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)
            
            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss   
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

            # final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy
            
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            
        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()
    
    
    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)
   

    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
