import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import copy
from torch.distributions import MultivariateNormal, Categorical

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6

# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)

class QNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim):
        super(QNetwork, self).__init__()

        # Q1 architecture
        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        # Q2 architecture
        self.linear4 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear5 = nn.Linear(hidden_dim, hidden_dim)
        self.linear6 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state, action):
        xu = torch.cat([state, action], 1)
        x1 = F.relu(self.linear1(xu))
        x1 = F.relu(self.linear2(x1))
        x1 = self.linear3(x1)

        x2 = F.relu(self.linear4(xu))
        x2 = F.relu(self.linear5(x2))
        x2 = self.linear6(x2)

        return x1, x2


class GaussianPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None):
        super(GaussianPolicy, self).__init__()
        
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean_linear = nn.Linear(hidden_dim, num_actions)
        self.log_std_linear = nn.Linear(hidden_dim, num_actions)

        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            high = torch.tensor([1.])
            low = torch.tensor([0.])
            self.action_scale = torch.FloatTensor(
                (high - low) / 2.)
            self.action_bias = torch.FloatTensor(
                (high + low) / 2.)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(GaussianPolicy, self).to(device)


class DeterministicPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None):
        super(DeterministicPolicy, self).__init__()
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean = nn.Linear(hidden_dim, num_actions)
        self.noise = torch.Tensor(num_actions)

        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = 1.
            self.action_bias = 0.
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = torch.tanh(self.mean(x)) * self.action_scale + self.action_bias
        return mean

    def sample(self, state):
        mean = self.forward(state)
        noise = self.noise.normal_(0., std=0.1)
        noise = noise.clamp(-0.25, 0.25)
        action = mean + noise
        return action, torch.tensor(0.), mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        self.noise = self.noise.to(device)
        return super(DeterministicPolicy, self).to(device)
    
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
    

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes = 10):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Sequential(
                        nn.Conv2d(2, 64, kernel_size = 7, stride = 2, padding = 3),
                        nn.BatchNorm2d(64),
                        nn.ReLU())
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        self.layer0 = self._make_layer(block, 64, layers[0], stride = 1)
        self.layer1 = self._make_layer(block, 128, layers[1], stride = 2)
        self.layer2 = self._make_layer(block, 256, layers[2], stride = 2)
        #self.layer3 = self._make_layer(block, 512, layers[3], stride = 2)
        self.avgpool = nn.AvgPool2d(8, stride=2)
        self.fc = nn.Linear(3840, num_classes)
        self.sigmoid = nn.Sigmoid()
        
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
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        # x = self.layer3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.sigmoid(x)
        return x

class QResNetwork(nn.Module):
    def __init__(self, block, layers, num_actions=2 , num_classes = 1, hidden_dim = 64):
        super(QResNetwork, self).__init__()
        self.inplanes = 64
        
        # Q1 architecture
        self.conv1 = nn.Sequential(
                        nn.Conv2d(2, 64, kernel_size = 7, stride = 2, padding = 3),
                        nn.BatchNorm2d(64),
                        nn.ReLU())
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        self.layer0 = self._make_layer(block, 64, layers[0], stride = 1)
        self.layer1 = self._make_layer(block, 128, layers[1], stride = 2)
        self.layer2 = self._make_layer(block, 256, layers[2], stride = 2)
        #self.layer3 = self._make_layer(block, 512, layers[3], stride = 2)
        self.avgpool = nn.AvgPool2d(8, stride=2)
        self.fc_1 = nn.Linear(4340, 8)
        self.linear1 = nn.Linear(8 + num_actions, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, num_classes)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim = 1)
        self.relu = nn.ReLU()

        # Q2 architecture
        self.conv2 = nn.Sequential(
                        nn.Conv2d(2, 64, kernel_size = 7, stride = 2, padding = 3),
                        nn.BatchNorm2d(64),
                        nn.ReLU())
        # self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        self.layer0_2 = copy.deepcopy(self.layer0)
        self.layer1_2 = copy.deepcopy(self.layer1)
        self.layer2_2 = copy.deepcopy(self.layer2)
        #self.layer3_2 = self._make_layer(block, 512, layers[3], stride = 2)
        self.avgpool_2 = nn.AvgPool2d(8, stride=2)
        self.fc_1_2 = nn.Linear(4340, 8)
        self.linear1_2 = nn.Linear(8 + num_actions, hidden_dim)
        self.linear2_2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3_2 = nn.Linear(hidden_dim, 1)
        # self.sigmoid_2 = nn.Sigmoid()
        # self.softmax_2 = nn.Softmax(dim = 1)

        self.apply(weights_init_)

        
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
    
    
    def forward(self, state, action):
        state_dc = copy.deepcopy(state)

        # Q1 architecture
        # last channel is the timestep
        timestep = state_dc[:, -1]
        # state only first 2 channels
        state = state_dc[:, :-1]
        # flatten timestep and make it 500 x 1
        timestep = timestep.view(timestep.shape[0], -1)
        timestep = timestep[:, :500]
        # print("timestep: ", timestep.shape)
        state = self.conv1(state)
        state = self.maxpool(state)
        state = self.layer0(state)
        state = self.layer1(state)
        state = self.layer2(state)
        # state = self.layer3(state)
        state = self.avgpool(state)
        state = state.view(state.size(0), -1)
        # concatenate timestep to state
        state = torch.cat((state, timestep), dim = 1)
        state = self.fc_1(state)
        x1 = torch.cat([state, action], 1)
        x1 = F.relu(self.linear1(x1))
        x1 = F.relu(self.linear2(x1))
        x1 = self.linear3(x1)

        # Q2 architecture
        # last channel is the timestep
        timestep = state_dc[:, -1]
        # state only first 2 channels
        state = state_dc[:, :-1]
        # flatten timestep and make it 500 x 1
        timestep = timestep.view(timestep.shape[0], -1)
        timestep = timestep[:, :500]
        # print("timestep: ", timestep.shape)
        state = self.conv2(state)
        state = self.maxpool(state)
        state = self.layer0_2(state)
        state = self.layer1_2(state)
        state = self.layer2_2(state)
        # state = self.layer3_2(state)
        state = self.avgpool_2(state)
        state = state.view(state.size(0), -1)
        # concatenate timestep to state
        state = torch.cat((state, timestep), dim = 1)
        state = self.fc_1_2(state)
        x2 = torch.cat([state, action], 1)
        x2 = F.relu(self.linear1_2(x2))
        x2 = F.relu(self.linear2_2(x2))
        x2 = self.linear3_2(x2)

        return x1, x2
    

class ActorTimestepPolicy(nn.Module):
    def __init__(self, block, layers, num_classes = 8):
        super(ActorTimestepPolicy, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Sequential(
                        nn.Conv2d(2, 64, kernel_size = 7, stride = 2, padding = 3),
                        nn.BatchNorm2d(64),
                        nn.ReLU())
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        self.layer0 = self._make_layer(block, 64, layers[0], stride = 1)
        self.layer1 = self._make_layer(block, 128, layers[1], stride = 2)
        self.layer2 = self._make_layer(block, 256, layers[2], stride = 2)
        #self.layer3 = self._make_layer(block, 512, layers[3], stride = 2)
        self.avgpool = nn.AvgPool2d(8, stride=2)
        self.fc = nn.Linear(4340, num_classes)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim = 1)
    
        self.action_scale = torch.tensor(0.5)
        self.action_bias = torch.tensor(0.5)

        self.apply(weights_init_)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
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
        x = self.layer1(x)
        x = self.layer2(x)
        # x = self.layer3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        # concatenate timestep to x
        x = torch.cat((x, timestep), dim = 1)
        x = self.fc(x)
        action_1 = self.softmax(x[:, :6])
        # print("action_1", action_1)
        action_2 = x[:, 6:]
        # clamp log std
        action_2[-1] = torch.clamp(action_2[-1], min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        result = torch.cat((action_1, action_2), dim = 1)
        # print("result", result)
        return result

    
    def sample(self, state):
        action = self.forward(state)
        action_probs = action[:, :6]
        # print("action_probs: ", action_probs.shape)
        dist_1 = Categorical(action_probs)
        action_1 = dist_1.sample()
        action_mean_1 = torch.argmax(action_probs, dim = 1)

        action_mean = action[:, -2]
        action_log_std = action[:, -1]
        action_std = action_log_std.exp()
        # num_continuous_action = action_mean.shape[0]
        dist_2 = Normal(action_mean, action_std)
        x_t = dist_2.rsample()
        y_t = torch.tanh(x_t)
        action_2 = y_t * self.action_scale + self.action_bias
        action_logprob_2 = dist_2.log_prob(x_t)
        # Enforcing Action Bound
        action_logprob_2 -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        action_logprob_2 = action_logprob_2.unsqueeze(dim=-1)

        action_logprob_2 = action_logprob_2.sum(1, keepdim=True)
        action_mean_2 = torch.tanh(action_mean) * self.action_scale + self.action_bias

        # print("action_1: ", action_1.shape)
        # print("action_2: ", action_2.shape)
        action = torch.cat((action_1.unsqueeze(dim=-1), action_2.unsqueeze(dim=-1)), dim=-1)
        action_logprob_1 = dist_1.log_prob(action[:, 0])
        action_logprob = action_logprob_1 + action_logprob_2
        
        mean_action = torch.cat((action_mean_1.unsqueeze(dim=-1), action_mean_2.unsqueeze(dim=-1)), dim=-1)

        return action.detach(), action_logprob.detach(), mean_action.detach()
    
    def to(self, device):
        return super(ActorTimestepPolicy, self).to(device)
    

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
    

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes = 10):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Sequential(
                        nn.Conv2d(2, 64, kernel_size = 7, stride = 2, padding = 3),
                        nn.BatchNorm2d(64),
                        nn.ReLU())
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        self.layer0 = self._make_layer(block, 64, layers[0], stride = 1)
        self.layer1 = self._make_layer(block, 128, layers[1], stride = 2)
        self.layer2 = self._make_layer(block, 256, layers[2], stride = 2)
        #self.layer3 = self._make_layer(block, 512, layers[3], stride = 2)
        self.avgpool = nn.AvgPool2d(8, stride=2)
        self.fc = nn.Linear(3840, num_classes)
        self.sigmoid = nn.Sigmoid()
        
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
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        # x = self.layer3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.sigmoid(x)
        return x

class ActorTimestepStatePolicy(nn.Module):
    def __init__(self, state_dim, hidden_dim=64, output_dim=4):
        super(ActorTimestepStatePolicy, self).__init__()
        # Define the layers
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.tanh1 = nn.Tanh()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.tanh2 = nn.Tanh()
        self.fc3 = nn.Linear(hidden_dim, output_dim)

        self.action_scale = torch.tensor(0.5)
        self.action_bias = torch.tensor(0.5)

    def forward(self, x):
        # Define the forward pass
        x = self.fc1(x)
        x = self.tanh1(x)
        x = self.fc2(x)
        x = self.tanh2(x)
        x = self.fc3(x)
        return x

    def sample(self, state):
        N = state.shape[0]
        assert state.shape == (N, 10)
        action = self.forward(state)
        assert action.shape == (N, 4)
        
        ## Action 1
        action_1_mean_unnorm = action[:, 0]
        action_1_mean = torch.tanh(action_1_mean_unnorm) * self.action_scale + self.action_bias
        assert action_1_mean_unnorm.shape == (N,)
        assert action_1_mean.shape == (N,)

        action_1_log_std = action[:, 1]
        action_1_std = action_1_log_std.exp()
        action_1_dist = Normal(action_1_mean_unnorm, action_1_std)
        x_t = action_1_dist.rsample()
        y_t = torch.tanh(x_t)
        action_1 = y_t * self.action_scale + self.action_bias
        assert action_1.shape == (N,)

        action_1_logprob = action_1_dist.log_prob(x_t)
        # Enforcing Action Bound
        action_1_logprob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        assert action_1_logprob.shape == (N,)

        ## Action 2
        action_2_mean_unnorm = action[:, -2]
        action_2_mean = torch.tanh(action_2_mean_unnorm) * self.action_scale + self.action_bias
        assert action_2_mean_unnorm.shape == (N,)
        assert action_2_mean.shape == (N,)

        action_2_log_std = action[:, -1]
        action_2_std = action_2_log_std.exp()
        action_2_dist = Normal(action_2_mean_unnorm, action_2_std)
        x_t = action_2_dist.rsample()
        y_t = torch.tanh(x_t)
        action_2 = y_t * self.action_scale + self.action_bias
        assert action_2.shape == (N,)

        action_2_logprob = action_2_dist.log_prob(x_t)
        # Enforcing Action Bound
        action_2_logprob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        assert action_2_logprob.shape == (N,)

        ## Get Action and Env Action
        action = torch.cat((action_1.unsqueeze(dim=-1), action_2.unsqueeze(dim=-1)), dim=1)
        assert action.shape == (N, 2)

        env_action = torch.cat((torch.ceil(action[:,0:1]*6-1), action[:,1:2]), dim=1)
        assert env_action.shape == (N, 2)

        action_logprob = action_1_logprob + action_2_logprob
        action_logprob = action_logprob.unsqueeze(dim=-1)
        assert action_logprob.shape == (N, 1)
        
        mean_action = torch.cat((action_1_mean.unsqueeze(dim=-1), action_2_mean.unsqueeze(dim=-1)), dim=-1)
        assert mean_action.shape == (N, 2)

        info = {
            'action_1_logprob_mean': action_1_logprob.mean(dim=0).item(),
            'action_1_logprob_std': action_1_logprob.std(dim=0).item(),
            'action_2_logprob_mean': action_2_logprob.mean(dim=0).item(),
            'action_2_logprob_std': action_2_logprob.std(dim=0).item(),
        }

        return action, action_logprob, mean_action, env_action, info
    
    def to(self, device):
        return super(ActorTimestepStatePolicy, self).to(device)