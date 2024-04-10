import torch.nn.functional as F
import torch.nn as nn
import torch

class Actor(nn.Module):
    def __init__(self, block, layers, num_classes = 10):
        super(Actor, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Sequential(
                        nn.Conv2d(2, 64, kernel_size = 7, stride = 2, padding = 3),
                        nn.BatchNorm2d(64),
                        nn.ReLU())
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        self.layer0 = self._make_layer(block, 64, layers[0], stride = 3)
        self.avgpool = nn.AvgPool2d(8, stride=2)
        self.fc = nn.Linear(2548, 256) #nn.Linear(2548, 256)
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
        # x = self.softmax(x)
        return x
    
    def pi(self, x, softmax_dim = 0):
        x = self.forward(x)
        x = self.fc2(x)
        prob = F.softmax(x, dim=softmax_dim)
        return prob

class Critic(nn.Module):
    def __init__(self, block, layers, num_classes = 1):
        super(Critic, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Sequential(
                        nn.Conv2d(2, 64, kernel_size = 7, stride = 2, padding = 3),
                        nn.BatchNorm2d(64),
                        nn.ReLU())
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        self.layer0 = self._make_layer(block, 64, layers[0], stride = 3)
        self.avgpool = nn.AvgPool2d(8, stride=2)
        self.fc = nn.Linear(2048, 256) #nn.Linear(2548, 256)
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
        # x = torch.cat((x, timestep), dim = 1)
        x = self.fc(x)
        x = self.relu(x)
        x = self.fc2(x)
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
    
def evaluate_policy(env, agent, turns = 3):
    total_scores = 0
    for j in range(turns):
        s, info = env.reset()
        done = False
        while not done:
            # Take deterministic actions at test time
            a, logprob_a = agent.select_action(s, deterministic=True)
            s_next, r, dw, tr, info = env.step(a)
            done = (dw or tr)

            total_scores += r
            s = s_next
    return int(total_scores/turns)


#You can just ignore this funciton. Is not related to the RL.
def str2bool(v):
    '''transfer str to bool for argparse'''
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'True','true','TRUE', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'False','false','FALSE', 'f', 'n', '0'):
        return False
    else:
        print('Wrong Input.')
        raise