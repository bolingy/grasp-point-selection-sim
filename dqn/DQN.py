import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch
import torchvision.transforms as transforms
import copy
from rl.rl_utils import *
import pickle

def build_net(layer_shape, activation, output_activation):
    '''build net with for loop'''
    layers = []
    for j in range(len(layer_shape)-1):
        act = activation if j < len(layer_shape)-2 else output_activation
        layers += [nn.Linear(layer_shape[j], layer_shape[j+1]), act()]
    return nn.Sequential(*layers)

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

class ActorTimestepNet(nn.Module):
    def __init__(self, block, layers, num_classes = 10):
        super(ActorTimestepNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Sequential(
                        nn.Conv2d(2, 64, kernel_size = 7, stride = 2, padding = 3),
                        nn.BatchNorm2d(64),
                        nn.ReLU())
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        self.layer0 = self._make_layer(block, 64, layers[0], stride = 3)
        self.avgpool = nn.AvgPool2d(8, stride=2)
        #TODO: removed timestep conditioning for now, add back later
        self.fc = nn.Linear(2548, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, num_classes)
        # self.sigmoid = nn.Sigmoid()
        # self.softmax = nn.Softmax(dim = 1)
        
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    
    
    def forward(self, x):
        # print("x: ", x.shape)
        # last channel is the timestep
        timestep = x[:, -1]
        # x only first 2 channels
        x = x[:, :-1]
        # print("x: ", x.shape)

        # flatten timestep and make it 500 x 1
        timestep = timestep.view(timestep.shape[0], -1)
        timestep = timestep[:, :500]
        # print("timestep: ", timestep.shape)
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.layer0(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = torch.cat((x, timestep), dim = 1)
        x = self.fc(x)
        x = self.relu(x)
        x = self.fc2(x)
        # x = self.softmax(x)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1, downsample = None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU())
        self.conv2 = nn.Sequential(
                        nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1),
                        nn.BatchNorm2d(out_channels))
        self.downsample = downsample
        self.relu = nn.ReLU()
        self.out_channels = out_channels
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out
    
class Q_Net(nn.Module):
    def __init__(self, state_dim, action_dim, hid_shape):
        super(Q_Net, self).__init__()
        layers = [state_dim] + list(hid_shape) + [action_dim]
        self.Q = build_net(layers, nn.ReLU, nn.Identity)

    def forward(self, s):
        q = self.Q(s)
        return q

class DQN_agent(object):
    def __init__(self, **kwargs):
        # Init hyperparameters for agent, just like "self.gamma = opt.gamma, self.lambd = opt.lambd, ..."
        self.__dict__.update(kwargs)
        self.tau = 0.005
        self.replay_buffer = ReplayBuffer(self.state_dim, torch.device("cpu"), self.train_device, self.res_net, max_size=int(self.buffer_size), data_augmentation_prob=self.data_augmentation_prob)
        if self.res_net:
            self.q_net = ActorTimestepNet(block = ResidualBlock, layers = [3, 4, 6, 3], num_classes=self.num_classes).to(self.train_device)
            # self.q_net = nn.DataParallel(self.q_net, device_ids=[0], output_device=self.train_device)
        else:
            self.q_net = Q_Net(self.state_dim, self.action_dim, (self.net_width,self.net_width)).to(self.train_device)
            # self.q_net = nn.DataParallel(self.q_net, device_ids=[0], output_device=self.train_device)
        self.q_net_optimizer = torch.optim.Adam(self.q_net.parameters(), lr=self.lr)
        self.q_target = copy.deepcopy(self.q_net)
        # Freeze target networks with respect to optimizers (only update via polyak averaging)
        for p in self.q_target.parameters(): p.requires_grad = False


    def select_action(self, state, deterministic):#only used when interact with the env
        with torch.no_grad():
            # state = torch.FloatTensor(state.reshape(1, -1).cpu()).to(self.train_device)
            N = state.shape[0]
            if self.res_net:
                assert state.shape == (N, 3, 180, 260)
            else:
                assert state.shape == (N, self.state_dim)
            if deterministic:
                a = self.q_net(state).argmax(dim=1)
            else:
                if np.random.rand() < self.exp_noise:
                    # size of N
                    a = torch.randint(self.action_dim, (N,), device=self.train_device)
                else:
                    a = self.q_net(state).argmax(dim=1)
            if type(a) == int:
                a = torch.tensor(a, dtype=torch.long, device=self.train_device).unsqueeze(-1)
            a = a.unsqueeze(-1)
            assert a.shape == (N, 1), "Expected shape {}, got {}, a = {}".format((N, 1), a.shape, a)
        return a


    def train(self):
        s, a, r, s_next, dw = self.replay_buffer.sample(self.batch_size)
        if self.res_net:
            assert s.shape == s_next.shape == (self.batch_size, 3, 180, 260)
        else:
            assert s.shape == s_next.shape == (self.batch_size, self.state_dim)
        assert a.shape == r.shape == dw.shape == (self.batch_size, 1)

        '''Compute the target Q value'''
        with torch.no_grad():
            if self.DDQN:
                argmax_a = self.q_net(s_next).argmax(dim=1).unsqueeze(-1)
                assert argmax_a.shape == (self.batch_size, 1)
                assert self.q_target(s_next).shape == (self.batch_size, self.action_dim)
                max_q_prime = self.q_target(s_next).gather(1,argmax_a)
                assert max_q_prime.shape == (self.batch_size, 1)
            else:
                max_q_prime = self.q_target(s_next).max(1)[0].unsqueeze(1)
            target_Q = r + (~dw) * self.gamma * max_q_prime #dw: die or win

        # Get current Q estimates
        current_q = self.q_net(s)
        assert current_q.shape == (self.batch_size, self.action_dim)
        assert a.shape == (self.batch_size, 1)
        current_q_a = current_q.gather(1, a)
        assert current_q_a.shape == (self.batch_size, 1)


        q_loss = F.mse_loss(current_q_a, target_Q)
        self.q_net_optimizer.zero_grad()
        q_loss.backward()
        self.q_net_optimizer.step()

        # Update the frozen target models
        for param, target_param in zip(self.q_net.parameters(), self.q_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


    def save(self,algo,EnvName,steps):
        torch.save(self.q_net.state_dict(), "./model/{}_{}_{}.pth".format(algo,EnvName,steps))

    def load(self,algo,EnvName,steps):
        self.q_net.load_state_dict(torch.load("./model/{}_{}_{}.pth".format(algo,EnvName,steps), map_location=self.train_device))
        self.q_target.load_state_dict(torch.load("./model/{}_{}_{}.pth".format(algo,EnvName,steps), map_location=self.train_device))


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

    def add(self, s, a, r, s_next, dw):
        for i in range(s.shape[0]):
            if len(self.buffer) < self.max_size:
                self.buffer.append(None)	
            self.buffer[self.ptr] = (s[i], a[i], r[i], s_next[i], dw[i])
            self.s[self.ptr] = s[i].to(self.buffer_device)
            self.a[self.ptr] = a[i]
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
        r = self.r[ind].to(self.train_device)
        dw = self.dw[ind].to(self.train_device)
        return s, a, r, s_next, dw

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
                self.r[i] = self.buffer[i][2]
                self.s_next[i] = self.buffer[i][3].to(self.buffer_device)
                self.dw[i] = self.buffer[i][4]

