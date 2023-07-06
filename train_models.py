import os
import glob
import time
from datetime import datetime

import isaacgym
import isaacgymenvs

import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical

import numpy as np

import einops
import gym
# import roboschool
#import pybullet_envs
from rl.ppo import *
from rl.rl_utils import *

# check cuda
device = torch.device('cpu')

if(torch.cuda.is_available()): 
    device = torch.device('cuda:0') 
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")

env_name = "robo_push"
has_continuous_action_space = True

max_ep_len = 2                    # max timesteps in one episode
max_training_timesteps = int(1e5)   # break training loop if timeteps > max_training_timesteps

print_freq = max_ep_len * 5     # print avg reward in the interval (in num timesteps)
log_freq = max_ep_len * 2       # log avg reward in the interval (in num timesteps)
save_model_freq = int(2e4)      # save model frequency (in num timesteps)

action_std = 0.01


#####################################################


## Note : print/log frequencies should be > than max_ep_len


################ PPO hyperparameters ################

update_timestep = max_ep_len * 1      # update policy every n timesteps
K_epochs = 40               # update policy for K epochs
eps_clip = 0.2              # clip parameter for PPO
gamma = 0.99                # discount factor

lr_actor = 0.0003       # learning rate for actor network
lr_critic = 0.001       # learning rate for critic network

random_seed = 0         # set random seed if required (0 = no random seed)


print("training environment name : " + env_name)

env = isaacgymenvs.make(
	seed=0,
	task="RL_UR16eManipulation",
	num_envs=2,
	sim_device="cuda:0", # cpu cuda:0
	rl_device="cuda:0", # cpu cuda:0
	multi_gpu=False,
	graphics_device_id=0
)

# state space dimension
state_dim = env.observation_space.shape[0]

# action space dimension
if has_continuous_action_space:
    action_dim = env.action_space.shape[0]
else:
    action_dim = env.action_space.n

###################### logging ######################

# log files for multiple runs are NOT overwritten

log_dir = "PPO_logs"
if not os.path.exists(log_dir):
      os.makedirs(log_dir)

log_dir = log_dir + '/' + env_name + '/'
if not os.path.exists(log_dir):
      os.makedirs(log_dir)


# get number of log files in log directory
run_num = 0
current_num_files = next(os.walk(log_dir))[2]
run_num = len(current_num_files)


# create new log file for each run 
log_f_name = log_dir + '/PPO_' + env_name + "_log_" + str(run_num) + ".csv"

print("current logging run number for " + env_name + " : ", run_num)
print("logging at : " + log_f_name)


################### checkpointing ###################

run_num_pretrained = 0      #### change this to prevent overwriting weights in same env_name folder

directory = "PPO_preTrained"
if not os.path.exists(directory):
      os.makedirs(directory)

directory = directory + '/' + env_name + '/'
if not os.path.exists(directory):
      os.makedirs(directory)


checkpoint_path = directory + "PPO_{}_{}_{}.pth".format(env_name, random_seed, run_num_pretrained)
print("save checkpoint path : " + checkpoint_path)

#####################################################


############# print all hyperparameters #############

print("--------------------------------------------------------------------------------------------")

print("max training timesteps : ", max_training_timesteps)
print("max timesteps per episode : ", max_ep_len)

print("model saving frequency : " + str(save_model_freq) + " timesteps")
print("log frequency : " + str(log_freq) + " timesteps")
print("printing average reward over episodes in last : " + str(print_freq) + " timesteps")

print("--------------------------------------------------------------------------------------------")

print("state space dimension : ", state_dim)
print("action space dimension : ", action_dim)

print("--------------------------------------------------------------------------------------------")

print("PPO update frequency : " + str(update_timestep) + " timesteps") 
print("PPO K epochs : ", K_epochs)
print("PPO epsilon clip : ", eps_clip)
print("discount factor (gamma) : ", gamma)

print("--------------------------------------------------------------------------------------------")

print("optimizer learning rate actor : ", lr_actor)
print("optimizer learning rate critic : ", lr_critic)

if random_seed:
    print("--------------------------------------------------------------------------------------------")
    print("setting random seed to ", random_seed)
    torch.manual_seed(random_seed)
    env.seed(random_seed)
    np.random.seed(random_seed)

    #####################################################

print("============================================================================================")

################# training procedure ################

# initialize a PPO agent
print(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std)
ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std, device)


# track total training time
start_time = datetime.now().replace(microsecond=0)
print("Started training at (GMT) : ", start_time)

print("============================================================================================")


# logging file
log_f = open(log_f_name,"w+")
log_f.write('episode,timestep,reward\n')


# printing and logging variables
print_running_reward = 0
print_running_episodes = 0

log_running_reward = 0
log_running_episodes = 0

time_step = 0
i_episode = 0

# training loop
while time_step <= max_training_timesteps:

    # state = rearrange_state(env.reset()['obs']) # is this a usable state????
    state, reward, done = step_primitives_env_0(torch.tensor(2 * [[0.11, 0., 0.28, 0.22]]).to(device), env) #env.reset()

    state, reward, done = state[None,:], reward[None, :], done[None, :] # remove when parallelized
    state = rearrange_state(state)
    current_ep_reward = 0
    
    for t in range(1, max_ep_len+1):
        # select action with policy
        action, _, _ = ppo_agent.select_action(state)
        action = clip_actions(action)

        ''' hacky way to demonstrate multi env, but only actuallly training from env 0'''
        # print("action shape", action.shape)
        # copy action from policy to all envs
        action = action.repeat(2, 1)
        # print("action shape", action.shape)

        state, reward, done = step_primitives_env_0(action, env)#env.step(action)
        state, reward, done = state[None,:], reward[None, :], done[None, :] # remove when parallelized
        state = rearrange_state(state)

        # saving reward and is_terminals
        ppo_agent.buffer.rewards.append(reward)
        ppo_agent.buffer.is_terminals.append(done)
        
        time_step +=1
        current_ep_reward += reward

        # update PPO agent
        if time_step % update_timestep == 0:
            print('training ppo')
            ppo_agent.update()

#         # if continuous action space; then decay action std of ouput action distribution
#         # if has_continuous_action_space and time_step % action_std_decay_freq == 0:
#         #     ppo_agent.decay_action_std(action_std_decay_rate, min_action_std)

#         # log in logging file
#         if time_step % log_freq == 0:

#             # log average reward till last episode
#             log_avg_reward = log_running_reward / log_running_episodes
#             log_avg_reward = round(log_avg_reward, 4)

#             log_f.write('{},{},{}\n'.format(i_episode, time_step, log_avg_reward))
#             log_f.flush()

#             log_running_reward = 0
#             log_running_episodes = 0

#         # printing average reward
#         if time_step % print_freq == 0:

#             # print average reward till last episode
#             print_avg_reward = print_running_reward / print_running_episodes
#             print_avg_reward = round(print_avg_reward, 2)

#             print("Episode : {} \t\t Timestep : {} \t\t Average Reward : {}".format(i_episode, time_step, print_avg_reward))

#             print_running_reward = 0
#             print_running_episodes = 0
            
#         # save model weights
#         if time_step % save_model_freq == 0:
#             print("--------------------------------------------------------------------------------------------")
#             print("saving model at : " + checkpoint_path)
#             ppo_agent.save(checkpoint_path)
#             print("model saved")
#             print("Elapsed Time  : ", datetime.now().replace(microsecond=0) - start_time)
#             print("--------------------------------------------------------------------------------------------")
            
#         # break; if the episode is over
#         if done:
#             break

#     print_running_reward += current_ep_reward
#     print_running_episodes += 1

#     log_running_reward += current_ep_reward
#     log_running_episodes += 1

#     i_episode += 1


# log_f.close()
# env.close()
        

