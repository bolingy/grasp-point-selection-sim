import os
import glob
import time
import wandb
from datetime import datetime

import isaacgym
import isaacgymenvs

import torch
torch.cuda.empty_cache()
import gc
gc.collect()
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical
import matplotlib.pyplot as plt

import numpy as np

import einops
import gym
# import roboschool
#import pybullet_envs
from rl.ppo import *
from rl.rl_utils import *
import wandb
wandb.login()

import warnings
warnings.filterwarnings("ignore")

# check cuda
train_device = torch.device('cpu')
sim_device = torch.device('cuda:0')

# if(torch.cuda.is_available()): 
#     device = torch.device('cuda:0') 
#     torch.cuda.empty_cache()
#     print("Device set to : " + str(torch.cuda.get_device_name(device)))
# else:
#     print("Device set to : cpu")

env_name = "bin_picking"
has_continuous_action_space = True

max_ep_len = 2                     # max timesteps in one episode
max_training_timesteps = int(1e5)   # break training loop if timeteps > max_training_timesteps

print_freq = 3                  # print avg reward in the interval (in num timesteps)
log_freq = max_ep_len * 10      # log avg reward in the interval (in num timesteps)
save_model_freq = int(2e4)      # save model frequency (in num timesteps)

action_std = 0.1 


#####################################################


## Note : print/log frequencies should be > than max_ep_len


################ PPO hyperparameters ################

pick_len = 3
update_size = pick_len * 10
K_epochs = 40               # update policy for K epochs
eps_clip = 0.2              # clip parameter for PPO
gamma = 0.99                # discount factor

lr_actor = 1e-8       # learning rate for actor network
lr_critic = 1e-8      # learning rate for critic network

random_seed = 0       # set random seed if required (0 = no random seed)

ne = 20               # number of environments

print("training environment name : " + env_name)

run = wandb.init(
    project='bin_picking', 
    config={
    "ne": ne,
    "pick_len": pick_len,
    "update_size": update_size,
    "max_training_timesteps": max_training_timesteps,
    "action_std": action_std,
    "eps_clip": eps_clip,
    "gamma": gamma,
    "lr_actor": lr_actor,
    "lr_critic": lr_critic,
    }
)


env = isaacgymenvs.make(
	seed=0,
	task="RL_UR16eManipulation",
	num_envs=ne,
	sim_device="cuda:0", # cpu cuda:0
	rl_device="cuda:0", # cpu cuda:0
	multi_gpu=False,
	graphics_device_id=0,
    headless=True
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

print("PPO update frequency : " + str(update_size) + " MPs") 
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
ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std, train_device)


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

buf_envs = [RolloutBuffer() for _ in range(ne)]
buf_central = RolloutBuffer()
actions = torch.tensor(ne * [[0.11, 0., 0.28, 0.22]]).to(sim_device)

state, reward, done, true_indicies = step_primitives(actions, env) #env.reset() by kickstarting w random action
state, reward, done, true_indicies = returns_to_device(state, reward, done, true_indicies, train_device)
# true_idx = torch.nonzero(indicies).squeeze(1)

# print(state.shape, reward.shape, done.shape, indicies)
# state, reward, done = state[None,:], reward[None, :], done[None, :] # remove when parallelized
state = rearrange_state(state)

curr_rewards = 0
# training loop
while time_step <= max_training_timesteps: ## prim_step
    # state = rearrange_state(env.reset()['obs']) # is this a usable state????
    action, action_logprob, state_val = ppo_agent.select_action(state)
    for i, true_i in enumerate(true_indicies):
        buf_envs[true_i].states.append(state[i][:, None].clone().detach())
        buf_envs[true_i].actions.append(action[i].clone().detach())
        buf_envs[true_i].logprobs.append(action_logprob[i].clone().detach().unsqueeze(0))
        buf_envs[true_i].state_values.append(state_val[i].clone().detach())
    action = scale_actions(action).to(sim_device)
    # true indicies to one hot flat vector
    one_hot = torch.zeros(ne).bool().to(sim_device)
    one_hot[true_indicies] = 1
    actions[one_hot] = action

    state, reward, done, true_indicies = step_primitives(actions, env)
    state, reward, done, true_indicies = returns_to_device(state, reward, done, true_indicies, train_device)
    state = rearrange_state(state)
    # true_idx = torch.nonzero(indicies).squeeze(1)
    for i, true_i in enumerate(true_indicies):
        if len(buf_envs[true_i].rewards) != len(buf_envs[true_i].states):
            buf_envs[true_i].rewards.append(reward[i].clone().detach().unsqueeze(0))
            buf_envs[true_i].is_terminals.append(done[i].clone().detach().unsqueeze(0))
            time_step += 1

        # check picking in done
        if buf_envs[true_i].is_done():
            if len(buf_envs[true_i].rewards) == len(buf_envs[true_i].states):
                assert len(buf_envs[true_i].rewards) == len(buf_envs[true_i].states), "rewards and states are not the same length at env {}".format(i)
                buf_central.append(copy.deepcopy(buf_envs[true_i]))
                wandb.log({"Central buffer size": len(buf_central.states)})
                buf_envs[true_i].clear()
            else:
                print("rewards and states are not the same length at env {}".format(true_i))
                buf_envs[true_i].clear()

    if buf_central.size() >= update_size:
        def calc_avg_reward_per_update():
            total_reward = sum(buf_central.rewards)
            num_rewards = sum(buf_central.is_terminals)
            return (total_reward / num_rewards).item()
        curr_rewards = calc_avg_reward_per_update()
        ppo_agent.update(buf_central)
        buf_central.clear()
        #free up memory
        torch.cuda.empty_cache()
        gc.collect()

        print_running_reward += curr_rewards
        print_running_episodes += 1
        i_episode += 1
        wandb.log({"Episodes": i_episode})
        wandb.log({"Average reward in every update": curr_rewards})
        print("Updated at timestep {} with average reward {}".format(time_step, curr_rewards))
    # if time_step % log_freq == 0:
    #     # log average reward till last episode
    #     log_avg_reward = log_running_reward / log_running_episodes
    #     log_avg_reward = round(log_avg_reward, 4)

    #     log_f.write('{},{},{}\n'.format(i_episode, time_step, log_avg_reward))
    #     log_f.flush()

    #     log_running_reward = 0
    #     log_running_episodes = 0

    # printing average reward
    if i_episode % print_freq == 0 and i_episode != 0 and print_running_episodes != 0:

        # print average reward till last episode
        print_avg_reward = print_running_reward / print_running_episodes
        # print_running_reward = round(print_running_reward, 2)
        wandb.log({"Average running reward in 5 updates": print_avg_reward})
        print("Episode : {} \t\t Timestep : {} \t\t Average Reward : {}".format(i_episode, time_step, print_avg_reward))

        print_running_reward = 0
        print_running_episodes = 0
        
    # # save model weights
    # if time_step % save_model_freq == 0:
    #     print("--------------------------------------------------------------------------------------------")
    #     print("saving model at : " + checkpoint_path)
    #     ppo_agent.save(checkpoint_path)
    #     print("model saved")
    #     print("Elapsed Time  : ", datetime.now().replace(microsecond=0) - start_time)
    #     print("--------------------------------------------------------------------------------------------")
        
    # # break; if the episode is over
    # if done:
    #     break


'''

    #for t in range(1, max_ep_len+1): ### batch size ->
    #### parallelize this
    # for i in indicies:
    #     action, action_logprob, state_val = ppo_agent.select_action(state[i][None,:])
    #     buf_envs[i].states.append(state[i][None,:].clone().detach())
    #     buf_envs[i].actions.append(action.clone().detach())
    #     buf_envs[i].logprobs.append(action_logprob.clone().detach())
    #     buf_envs[i].state_values.append(state_val.clone().detach())
    #     action = scale_actions(action)
    #     actions[i] = action
    #     if i == 0:
    #         print("#############action added to buf")
        state, reward, done, indicies = step_primitives(actions, env)#env.step(action)
        state = rearrange_state(state)
        # indicies contains the indicies of the environments that have valid observations
        # saving reward and is_terminals
        for i in indicies:
            buf_envs[i].rewards.append(reward[i].clone().detach().unsqueeze(0))
            buf_envs[i].is_terminals.append(done[i].clone().detach().unsqueeze(0))
            if i == 0:
                print("#############reward added to buf")
        print("buf of env 0", str(buf_envs[0]))
        print("size of buf state of env 0", str(len(buf_envs[0].states)))
        print("size of buf reward of env 0", str(len(buf_envs[0].rewards)))
        print("indicies", indicies)

        # update PPO agent if environment is ready to be updated
        for i in range(ne):
            if buf_envs[i].is_done():
                buf_central[i].append(buf_envs[i])
                buf_envs[i].clear()

        if time_step % update_timestep == 0:
            ppo_agent.update(buf_central)
            buf_central.clear()

    
        # if continuous action space; then decay action std of ouput action distribution
        # if has_continuous_action_space and time_step % action_std_decay_freq == 0:
        #     ppo_agent.decay_action_std(action_std_decay_rate, min_action_std)

        # log in logging file
        if time_step % log_freq == 0:

            # log average reward till last episode
            log_avg_reward = log_running_reward / log_running_episodes
            log_avg_reward = round(log_avg_reward, 4)

            log_f.write('{},{},{}\n'.format(i_episode, time_step, log_avg_reward))
            log_f.flush()

            log_running_reward = 0
            log_running_episodes = 0

        # printing average reward
        if time_step % print_freq == 0:

            # print average reward till last episode
            print_avg_reward = print_running_reward / print_running_episodes
            print_avg_reward = round(print_avg_reward, 2)

            print("Episode : {} \t\t Timestep : {} \t\t Average Reward : {}".format(i_episode, time_step, print_avg_reward))

            print_running_reward = 0
            print_running_episodes = 0
            
        # # save model weights
        # if time_step % save_model_freq == 0:
        #     print("--------------------------------------------------------------------------------------------")
        #     print("saving model at : " + checkpoint_path)
        #     ppo_agent.save(checkpoint_path)
        #     print("model saved")
        #     print("Elapsed Time  : ", datetime.now().replace(microsecond=0) - start_time)
        #     print("--------------------------------------------------------------------------------------------")
            
        # # break; if the episode is over
        # if done:
        #     break

    print_running_reward += current_ep_reward
    print_running_episodes += 1

    log_running_reward += current_ep_reward
    log_running_episodes += 1

    i_episode += 1


log_f.close()
env.close()
        
'''
