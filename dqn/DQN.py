import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch
import copy
from rl.rl_utils import *

def build_net(layer_shape, activation, output_activation):
	'''build net with for loop'''
	layers = []
	for j in range(len(layer_shape)-1):
		act = activation if j < len(layer_shape)-2 else output_activation
		layers += [nn.Linear(layer_shape[j], layer_shape[j+1]), act()]
	return nn.Sequential(*layers)

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
        self.fc = nn.Linear(2548, num_classes)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim = 1)
        
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
        # concatenate timestep to x
        x = torch.cat((x, timestep), dim = 1)
        x = self.fc(x)
        x = self.softmax(x)
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
		self.replay_buffer = ReplayBuffer(self.state_dim, torch.device("cpu"), self.res_net, max_size=int(self.buffer_size), data_augmentation=self.data_augmentation)
		if self.res_net:
			self.q_net = ActorTimestepNet(block = ResidualBlock, layers = [3, 4, 6, 3], num_classes=6).to(self.train_device)
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
		s = s.to(self.train_device)
		a = a.to(self.train_device)
		r = r.to(self.train_device)
		s_next = s_next.to(self.train_device)
		dw = dw.to(self.train_device)
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
		self.q_net.load_state_dict(torch.load("./model/{}_{}_{}.pth".format(algo,EnvName,steps)))
		self.q_target.load_state_dict(torch.load("./model/{}_{}_{}.pth".format(algo,EnvName,steps)))


class ReplayBuffer(object):
	def __init__(self, state_dim, dvc, res_net, max_size=int(1e6), data_augmentation=False):
		self.max_size = max_size
		self.train_device = dvc
		self.ptr = 0
		self.size = 0
		if res_net:
			self.s = torch.zeros((max_size, 3, 180, 260),dtype=torch.float,device=self.train_device)
		else:
			self.s = torch.zeros((max_size, state_dim),dtype=torch.float,device=self.train_device)
		self.a = torch.zeros((max_size, 1),dtype=torch.long,device=self.train_device)
		self.r = torch.zeros((max_size, 1),dtype=torch.float,device=self.train_device)
		if res_net:
			self.s_next = torch.zeros((max_size, 3, 180, 260),dtype=torch.float,device=self.train_device)
		else:
			self.s_next = torch.zeros((max_size, state_dim),dtype=torch.float,device=self.train_device)
		self.dw = torch.zeros((max_size, 1),dtype=torch.bool,device=self.train_device)
		self.data_augmentation = data_augmentation

	def add(self, s, a, r, s_next, dw):
		# self.s[self.ptr] = s.to(self.train_device)
		# self.a[self.ptr] = a
		# self.r[self.ptr] = r
		# self.s_next[self.ptr] = s_next.to(self.train_device)
		# self.dw[self.ptr] = dw
		for i in range(s.shape[0]):
			self.s[self.ptr] = s[i].to(self.train_device)
			self.a[self.ptr] = a[i]
			self.r[self.ptr] = r[i]
			self.s_next[self.ptr] = s_next[i].to(self.train_device)
			self.dw[self.ptr] = dw[i]
			self.ptr = (self.ptr + 1) % self.max_size
			self.size = min(self.size + 1, self.max_size)

		self.ptr = (self.ptr + 1) % self.max_size
		self.size = min(self.size + 1, self.max_size)

	def sample(self, batch_size):
		ind = torch.randint(0, self.size, device=self.train_device, size=(batch_size,))
		if self.data_augmentation:
			self.s[ind] = data_augmentation(self.s[ind])
		return self.s[ind], self.a[ind], self.r[ind], self.s_next[ind], self.dw[ind]




