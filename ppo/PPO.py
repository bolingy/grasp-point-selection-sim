from torch.distributions import Categorical
from utils import Actor, Critic, ResidualBlock
import numpy as np
import torch
import torchvision.transforms as transforms
import copy
import math
import pickle

def gauss_noise_tensor(sigma_range):
    def gauss_noise_tensor_f(img, sigma_range):
        assert isinstance(img, torch.Tensor)
        dtype = img.dtype
        if not img.is_floating_point():
            img = img.to(torch.float32)

        if sigma_range[0] == sigma_range[1]:
            sigma = sigma_range[0]
        else:
            sigma = torch.rand(1).to(img.device) * (sigma_range[1] - sigma_range[0]) + sigma_range[0]
        out = img + sigma * torch.randn_like(img).to(img.device)

        if out.dtype != dtype:
            out = out.to(dtype)

        return out
    return lambda img: gauss_noise_tensor_f(img, sigma_range)

class PPO_discrete():
    def __init__(self, **kwargs):
        # Init hyperparameters for PPO agent, just like "self.gamma = opt.gamma, self.lambd = opt.lambd, ..."
        self.__dict__.update(kwargs)

        '''Build Actor and Critic'''
        self.actor = Actor(block = ResidualBlock, layers = [3, 4, 6, 3], num_classes=self.num_classes).to(self.train_device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic = Critic(block = ResidualBlock, layers = [3, 4, 6, 3], num_classes=1).to(self.train_device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr)

        '''Build Trajectory holder'''
        self.replay_buffer = ReplayBuffer(self.state_dim, torch.device("cpu"), self.train_device, self.res_net, max_size=int(self.buffer_size), data_augmentation_prob=self.data_augmentation_prob)


    def select_action(self, s, deterministic):
        ne = s.shape[0]
        with torch.no_grad():
            pi = self.actor.pi(s, softmax_dim=1)
            assert pi.shape[0] == ne
            if deterministic:
                a = torch.argmax(pi).item()
                assert a.shape[0] == ne
                a = a.unsqueeze(-1)
                return a, None
            else:
                m = Categorical(pi)
                a = m.sample()
                assert a.shape[0] == ne
                pi_a = pi[:, a]
                assert pi_a.shape[0] == ne
                a = a.unsqueeze(-1)
                return a, pi_a

    def train(self):
        self.entropy_coef *= self.entropy_coef_decay #exploring decay
        '''Prepare PyTorch data from Numpy data'''
        # s = torch.from_numpy(self.s_hoder).to(self.dvc)
        # a = torch.from_numpy(self.a_hoder).to(self.dvc)
        # r = torch.from_numpy(self.r_hoder).to(self.dvc)
        # s_next = torch.from_numpy(self.s_next_hoder).to(self.dvc)
        # old_prob_a = torch.from_numpy(self.logprob_a_hoder).to(self.dvc)
        # done = torch.from_numpy(self.done_hoder).to(self.dvc)
        # dw = torch.from_numpy(self.dw_hoder).to(self.dvc)

        s, a, old_prob_a, r, s_next, dw = self.replay_buffer.sample(len(self.replay_buffer.buffer))
        done = dw.clone().detach()

        ''' Use TD+GAE+LongTrajectory to compute Advantage and TD target'''
        with torch.no_grad():
            vs = self.critic(s)
            vs_ = self.critic(s_next)

            '''dw(dead and win) for TD_target and Adv'''
            deltas = r + self.gamma * vs_ * (~dw) - vs
            deltas = deltas.cpu().flatten().numpy()
            adv = [0]

            '''done for GAE'''
            for dlt, done in zip(deltas[::-1], done.cpu().flatten().numpy()[::-1]):
                advantage = dlt + self.gamma * self.lambd * adv[-1] * (~done)
                adv.append(advantage)
            adv.reverse()
            adv = copy.deepcopy(adv[0:-1])
            adv = torch.tensor(adv).unsqueeze(1).float().to(self.dvc)
            td_target = adv + vs
            if self.adv_normalization:
                adv = (adv - adv.mean()) / ((adv.std() + 1e-4))  #sometimes helps

        """PPO update"""
        #Slice long trajectopy into short trajectory and perform mini-batch PPO update
        optim_iter_num = int(math.ceil(s.shape[0] / self.batch_size))

        for _ in range(self.K_epochs):
            #Shuffle the trajectory, Good for training
            perm = np.arange(s.shape[0])
            np.random.shuffle(perm)
            perm = torch.LongTensor(perm).to(self.dvc)
            s, a, td_target, adv, old_prob_a = \
                s[perm].clone(), a[perm].clone(), td_target[perm].clone(), adv[perm].clone(), old_prob_a[perm].clone()

            '''mini-batch PPO update'''
            for i in range(optim_iter_num):
                index = slice(i * self.batch_size, min((i + 1) * self.batch_size, s.shape[0]))

                '''actor update'''
                prob = self.actor.pi(s[index], softmax_dim=1)
                entropy = Categorical(prob).entropy().sum(0, keepdim=True)
                prob_a = prob.gather(1, a[index])
                ratio = torch.exp(torch.log(prob_a) - torch.log(old_prob_a[index]))  # a/b == exp(log(a)-log(b))

                surr1 = ratio * adv[index]
                surr2 = torch.clamp(ratio, 1 - self.clip_rate, 1 + self.clip_rate) * adv[index]
                a_loss = -torch.min(surr1, surr2) - self.entropy_coef * entropy

                self.actor_optimizer.zero_grad()
                a_loss.mean().backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 40)
                self.actor_optimizer.step()

                '''critic update'''
                c_loss = (self.critic(s[index]) - td_target[index]).pow(2).mean()
                for name, param in self.critic.named_parameters():
                    if 'weight' in name:
                        c_loss += param.pow(2).sum() * self.l2_reg

                self.critic_optimizer.zero_grad()
                c_loss.backward()
                self.critic_optimizer.step()

    def save(self, episode):
        torch.save(self.critic.state_dict(), "./model/ppo_critic{}.pth".format(episode))
        torch.save(self.actor.state_dict(), "./model/ppo_actor{}.pth".format(episode))

    def load(self, episode):
        self.critic.load_state_dict(torch.load("./model/ppo_critic{}.pth".format(episode)))
        self.actor.load_state_dict(torch.load("./model/ppo_actor{}.pth".format(episode)))


class ReplayBuffer(object):
    def __init__(self, state_dim, dvc, train_dvc, res_net, max_size=int(1e6), data_augmentation_prob=0.):
        self.max_size = max_size
        self.buffer_device = dvc
        self.train_device = train_dvc
        self.buffer = []
        self.ptr = 0
        self.size = 0
        if res_net:
            self.s = torch.zeros((max_size, 3, 180, 260),dtype=torch.float,device=self.buffer_device)
        else:
            self.s = torch.zeros((max_size, state_dim),dtype=torch.float,device=self.buffer_device)
        self.a = torch.zeros((max_size, 1),dtype=torch.long,device=self.buffer_device)
        self.r = torch.zeros((max_size, 1),dtype=torch.float,device=self.buffer_device)
        if res_net:
            self.s_next = torch.zeros((max_size, 3, 180, 260),dtype=torch.float,device=self.buffer_device)
        else:
            self.s_next = torch.zeros((max_size, state_dim),dtype=torch.float,device=self.buffer_device)
        self.logprob_a = torch.zeros((max_size, 1),dtype=torch.float,device=self.buffer_device)
        self.dw = torch.zeros((max_size, 1),dtype=torch.bool,device=self.buffer_device)
        self.data_augmentation_prob = data_augmentation_prob
        self.augmentation = transforms.Compose([
            # Rotation
            transforms.Pad((150, 210), padding_mode='edge'),
            transforms.RandomRotation(10, transforms.InterpolationMode.NEAREST),
            transforms.CenterCrop((180, 260)),
            
            # Random Crop
            transforms.RandomCrop((180, 260), padding=(15, 21), padding_mode='edge'),
            
            # Random Resized Crop
            transforms.RandomResizedCrop((180, 260), scale=(0.9, 1.0)),
            
            # Gaussian Blur
            transforms.GaussianBlur((21, 21), sigma=(0.1, 2.0)),
            
            # Salt and Pepper
            gauss_noise_tensor((0, 0.05)),
        ])

    def add(self, s, a, r, s_next, logprob_a, dw):
        for i in range(s.shape[0]):
            if len(self.buffer) < self.max_size:
                self.buffer.append(None)	
            self.buffer[self.ptr] = (s[i], a[i], logprob_a[i], r[i], s_next[i], dw[i])
            self.s[self.ptr] = s[i].to(self.buffer_device)
            self.a[self.ptr] = a[i]
            self.logprob_a[self.ptr] = logprob_a[i]
            self.r[self.ptr] = r[i]
            self.s_next[self.ptr] = s_next[i].to(self.buffer_device)
            self.dw[self.ptr] = dw[i]
            self.ptr = (self.ptr + 1) % self.max_size
            self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = torch.randint(0, self.size, device=self.buffer_device, size=(batch_size,))
        s = self.s[ind].to(self.train_device)
        s_next = self.s_next[ind].to(self.train_device)
        if self.data_augmentation_prob > np.random.rand():
            s_s_next = torch.cat((s, s_next), axis=0)
            assert s_s_next.shape == (s.shape[0]*2, 3, 180, 260)
            s_s_next = self.augmentation(s_s_next)
            s, s_next = s_s_next[:s.shape[0]], s_s_next[s.shape[0]:]
        a = self.a[ind].to(self.train_device)
        logprob_a = self.logprob_a[ind].to(self.train_device)
        r = self.r[ind].to(self.train_device)
        dw = self.dw[ind].to(self.train_device)
        return s, a, logprob_a, r, s_next, dw

    def save_buffer(self, env_name, suffix="", save_path=None):
        if save_path is None:
            save_path = "checkpoints/dqn_buffer_{}_{}".format(env_name, suffix)
        print('Saving buffer to {}'.format(save_path))

        with open(save_path, 'wb') as f:
            pickle.dump(self.buffer, f)
    
    def load_buffer(self, save_path):
        print('Loading buffer from {}'.format(save_path))

        with open(save_path, "rb") as f:
            self.buffer = pickle.load(f)
            self.ptr = len(self.buffer) % self.max_size
            self.size = len(self.buffer)
            for i in range(self.size):
                self.s[i] = self.buffer[i][0].to(self.buffer_device)
                self.a[i] = self.buffer[i][1]
                self.logprob_a[i] = self.buffer[i][2]
                self.r[i] = self.buffer[i][3]
                self.s_next[i] = self.buffer[i][4].to(self.buffer_device)
                self.dw[i] = self.buffer[i][5]

