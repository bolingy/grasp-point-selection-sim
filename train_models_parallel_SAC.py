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
from rl.sac import *
from rl.rl_utils import *
from rl.sac_replay_memory import *
import wandb
import time
wandb.login(key='0ce23b9ac23d5e7e466e1bb6aa633ff0a3624a7b')

import random

import warnings
warnings.filterwarnings("ignore")

# check cuda
train_device = torch.device('cuda:0')
sim_device = torch.device('cuda:0')

# if(torch.cuda.is_available()): 
#     device = torch.device('cuda:0') 
#     torch.cuda.empty_cache()
#     print("Device set to : " + str(torch.cuda.get_device_name(device)))
# else:
#     print("Device set to : cpu")


has_continuous_action_space = True

#max_ep_len = 2                     # max timesteps in one episode
max_training_timesteps = int(1e5)   # break training loop if timeteps > max_training_timesteps

print_freq = 480                  # print avg reward in the interval (in num timesteps)
log_freq = 90 #max_ep_len * 10      # log avg reward in the interval (in num timesteps)
save_model_freq = 480      # save model frequency (in num timesteps)

## Note : print/log frequencies should be > than max_ep_len
################ SAC hyperparameters ################

pick_len = 3
update_size = pick_len * 90
start_steps = 270
updates_per_step = 1
max_size = 1000000
actor_lr = 1e-7
critic_lr = 3e-7
alpha_lr = 0.00003
gamma = 0.99                # discount factor
tau = 0.005                 # target network update rate
autoentropy = True          # automatically udpates alpha for entropy
alpha = 0.9
init_alpha = 0.2
hidden_size = 64           # size of hidden layers

random_seed = 1       # set random seed if required (0 = no random seed)


'''Training/Evaluation Parameter'''
env_name = "RL_UR16eManipulation_Full_Nocam"
policy_name = "SAC_state_gauss0.2std_attempt2"
ne = 90               # number of environments
head_less = True
EVAL = False #if you want to evaluate the model
if EVAL:
    start_steps = 0
    autoentropy = False
    alpha = 0.0
load_policy = False
policy_name = "{0}_batch_{1}_actorlr_{2}_criticlr_{3}_gamma_{4}_tau_{5}".format(policy_name, update_size, actor_lr, critic_lr, gamma, tau)
# policy_name = "seq_multiobjreachori_SAC_batch_1000_actorlr_0.001_criticlr_0.001_gamma_0.99_tau_0.005_3"
load_policy_version = None                  # specify policy version (i.e. int, 50) when loading a trained policy
res_net = False

print("training environment name : " + env_name)
if not EVAL:
    run = wandb.init(
        project='bin_picking_sac', 
        config={
        "ne": ne,
        "pick_len": pick_len,
        "update_size": update_size,
        "max_training_timesteps": max_training_timesteps,
        "gamma": gamma,
        "tau": tau,
        "actor_lr": actor_lr,
        "critic_lr": critic_lr,
        "alpha_lr": alpha_lr,
        "autoentropy": autoentropy,
        "alpha": alpha,
        "init_alpha": init_alpha,
        "hidden_size": hidden_size,
        "max_size": max_size,

        }
    )


env = isaacgymenvs.make(
	seed=0,
	task=env_name,
	num_envs=ne,
	sim_device='cuda:0', # cpu cuda:0
	rl_device='cuda:0', # cpu cuda:0
	multi_gpu=False,
	graphics_device_id=0,
    headless=head_less
)

# state space dimension
state_dim = env.observation_space.shape[0]

# action space dimension
if has_continuous_action_space:
    action_dim = env.action_space.shape[0] # -1 for poking behavior
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
policy_num = 0

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


directory = "PPO_preTrained" + '/' + env_name + '/'
#checkpoint_path = directory + policy_name
checkpoint_path = model_name(directory, policy_name, load_policy_version, extension=False)

print("save checkpoint path : " + checkpoint_path)

#####################################################


############# print all hyperparameters #############

print("--------------------------------------------------------------------------------------------")

print("max training timesteps : ", max_training_timesteps)
#print("max timesteps per episode : ", max_ep_len)

print("model saving frequency : " + str(save_model_freq) + " timesteps")
print("log frequency : " + str(log_freq) + " timesteps")
print("printing average reward over episodes in last : " + str(print_freq) + " timesteps")

print("--------------------------------------------------------------------------------------------")

print("state space dimension : ", state_dim)
print("action space dimension : ", action_dim)

print("--------------------------------------------------------------------------------------------")

print("PPO update frequency : " + str(update_size) + " MPs") 
print("discount factor (gamma) : ", gamma)

print("--------------------------------------------------------------------------------------------")

if random_seed:
    random_seed = random.randint(1, 10000)
    print("--------------------------------------------------------------------------------------------")
    print("setting random seed to ", random_seed)
    random_seed = 2
    #####################################################

print("============================================================================================")

################# training procedure ################
action_space = torch.zeros(action_dim)

# initialize a PPO agent
sac_agent = SAC(state_dim, action_space, gamma, tau, alpha, init_alpha, "Mixed", 1, autoentropy, train_device, hidden_size, actor_lr, critic_lr, alpha_lr, res_net)
if load_policy:
    checkpoint_path = "checkpoints/sac_checkpoint_{}_{}_{}".format(env_name, policy_name, load_policy_version)
    if os.path.exists(checkpoint_path):
        print("loading network from : " + checkpoint_path)
        sac_agent.load_checkpoint(checkpoint_path, EVAL)
        # if EVAL:
        #     ppo_agent.policy.eval()
        #     ppo_agent.policy_old.eval()
        print("network loaded")
    else:
        print("checkpoint {} does not exist.".format(checkpoint_path))
        print("No preTrained network exists. New network created")




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

updates = 0
i_episode = 0
prev_i_episode = 0

buf_envs = [RolloutBuffer() for _ in range(ne)]
memory = ReplayMemory(max_size, random_seed)
prev_mem_size = 0

actions = torch.tensor(ne * [[0.11, 0., 0.28, 0.22]]).to(sim_device)

state, scrap, reward, done, true_indicies = step_primitives(actions, env) #env.reset() by kickstarting w random action
state, scrap, reward, done, true_indicies = returns_to_device(state, scrap, reward, done, true_indicies, train_device)

# state, reward, done, true_indicies = returns_to_device(state, reward, done, true_indicies, train_device)
# true_idx = torch.nonzero(indicies).squeeze(1)

# print(state.shape, reward.shape, done.shape, indicies)
# state, reward, done = state[None,:], reward[None, :], done[None, :] # remove when parallelized

if res_net:
    real_ys, real_dxs = get_real_ys_dxs(state)
    state = rearrange_state_timestep(state)
else:
    # print("state.shape", state.shape)
    obj_state = state[:, -10:]
    # print("obj_state.shape", obj_state.shape)
    state = state[:, :-10]
    # print("state.shape", state.shape)
    real_ys, real_dxs = get_real_ys_dxs(state)
    state = rearrange_state(state)
obj_state = normalize_state(obj_state)

curr_rewards = 0
score_history = []
total_timesteps = 0
time_step_update = 0
# training loop
while total_timesteps <= max_training_timesteps:
    if total_timesteps < start_steps:
        # randomly sample actions 
        # action 1 is discrete (0-5), action 2 is continuous (0-1)
        action_1 = torch.randint(0, 6, (state.shape[0], 1)).to(sim_device)
        action_2 = torch.rand(state.shape[0], 1).to(sim_device)
        action = torch.cat((action_1, action_2), dim = 1)
    else:
        action = sac_agent.select_action(obj_state, EVAL)
    for i, true_i in enumerate(true_indicies):
        buf_envs[true_i].states.append(obj_state[i][:, None].clone().detach())
        buf_envs[true_i].actions.append(action[i].clone().detach())
        if true_i == 0:
            print("action of env 0 updated", action[i])
    action = convert_actions(action, real_ys, real_dxs, sim_device=sim_device)

    if len(memory) > update_size and len(memory) != prev_mem_size and not EVAL:
        prev_mem_size = len(memory)
        # Number of updates per step in environment
        for i in range(updates_per_step):
            # Update parameters of all the networks
            critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = sac_agent.update_parameters(memory, update_size, updates)
            updates += 1
            wandb.log({'critic_1_loss': critic_1_loss, 'critic_2_loss': critic_2_loss, 'policy_loss': policy_loss, 'ent_loss': ent_loss, 'alpha': alpha})
            print("Updated parameters of sac agent")
            print("critic_1_loss: ", critic_1_loss, "critic_2_loss: ", critic_2_loss, "policy_loss: ", policy_loss, "ent_loss: ", ent_loss, "alpha: ", alpha)

    # action = torch.cat((action, torch.zeros(true_indicies.shape[0], 1).to(sim_device)), dim=1)
    create_env_action_via_true_indicies(true_indicies, action, actions, ne, sim_device)
    state, scrap, reward, done, true_indicies = step_primitives(actions, env) #env.reset() by kickstarting w random action
    
    if EVAL and true_indicies[0] == 0 and res_net:
        imgs = state
        img_x = 260
        img_y = 180
        img = imgs[:, :img_x*img_y*2]
        depth = img[:, :img_x*img_y]
        seg = img[:, img_x*img_y:]
        depth = depth.reshape(-1, img_y, img_x)
        seg = seg.reshape(-1, img_y, img_x)
        depth = depth[0].cpu().numpy()
        seg = seg[0].cpu().numpy()
        plt.imshow(depth)
        plt.show()
        plt.imshow(seg)
        plt.show()
    elif EVAL and true_indicies[0] == 0 and not res_net:
        # state is in the form of (target_x, target_y, target_z, obj1_x, obj1_y, obj1_z, obj2_x, obj2_y, obj2_z)
        # print whether target is left, middle, or right compared to the two objects
        state_env_0 = state[0]
        print("state env 0", state_env_0)
        print("reward env 0", reward[0])
        target_y = state_env_0[1]
        obj1_y = state_env_0[4]
        obj2_y = state_env_0[7]
        if target_y < obj1_y and target_y < obj2_y:
            print("target is right")
        elif target_y > obj1_y and target_y > obj2_y:
            print("target is left")
        else:
            print("target is y-middle")
        
        target_x = state_env_0[0]
        obj1_x = state_env_0[3]
        obj2_x = state_env_0[6]
        if target_x < obj1_x and target_x < obj2_x:
            print("target is front")
        elif target_x > obj1_x and target_x > obj2_x:
            print("target is back")
        else:
            print("target is x-middle")

    state, scrap, reward, done, true_indicies = returns_to_device(state, scrap, reward, done, true_indicies, train_device)
    if res_net:
        real_ys, real_dxs = get_real_ys_dxs(state)
        state = rearrange_state_timestep(state)
    else:
        obj_state = state[:, -10:]
        # print("obj_state.shape", obj_state.shape)
        state = state[:, :-10]
        # print("state.shape", state.shape)
        real_ys, real_dxs = get_real_ys_dxs(state)
        state = rearrange_state(state)
    obj_state = normalize_state(obj_state)

    for i, true_i in enumerate(true_indicies):
        if len(buf_envs[true_i].rewards) != len(buf_envs[true_i].states):
            buf_envs[true_i].rewards.append(reward[i].clone().detach().unsqueeze(0))
            buf_envs[true_i]. scraps.append(scrap[i].clone().detach().unsqueeze(0))
            buf_envs[true_i].is_terminals.append(done[i].clone().detach().unsqueeze(0))
            buf_envs[true_i].new_states.append(obj_state[i][:, None].clone().detach())
            if done[i]:
                score_history.append(reward[i].item())
            i_episode += 1
            total_timesteps += 1
            time_step_update += 1
            print_running_reward += reward[i].item()
            print_running_episodes += 1
            # !!!!!!!!!!!mask (?)
        if(buf_envs[true_i].is_terminals[-1] == True):
            if(1 not in buf_envs[true_i].scraps):
                # memory.push(buf_envs[true_i].states[-1], buf_envs[true_i].actions[-1], buf_envs[true_i].rewards[-1], buf_envs[true_i].new_state[-1], ~ torch.tensor(buf_envs[true_i].is_terminals[-1], dtype=torch.bool))
                for i in range(len(buf_envs[true_i].states)):
                    memory.push(buf_envs[true_i].states[i], buf_envs[true_i].actions[i], buf_envs[true_i].rewards[i], buf_envs[true_i].new_states[i], ~ torch.tensor(buf_envs[true_i].is_terminals[i], dtype=torch.bool))
                wandb.log({"Buffer size": len(memory)})
                buf_envs[true_i].clear()
            else:
                print("Scrapped env {}".format(true_i))
                buf_envs[true_i].clear()


    # printing average reward
    if i_episode % print_freq == 0 and i_episode != 0 and print_running_episodes != 0:

        # print average reward till last episode
        print_avg_reward = print_running_reward / print_running_episodes
        # print_running_reward = round(print_running_reward, 2)
        if not EVAL:
            wandb.log({"Average running reward in every {} episodes".format(print_freq): print_avg_reward})
        print("Episode : {} \t\t Timestep : {} \t\t Average Reward : {}".format(i_episode, total_timesteps, print_avg_reward))

        print_running_reward = 0
        print_running_episodes = 0

    # printing average reward
    if i_episode % print_freq == 0 and i_episode != 0:
        avg_reward = np.mean(score_history[-100:])
        if not EVAL:
            wandb.log({"Average reward in past 100 timesteps": avg_reward})
        print("Episode : {} \t\t Timestep : {} \t\t Average Reward : {}".format(i_episode, total_timesteps, avg_reward))

    # save model weights
    if i_episode % save_model_freq == 0 and prev_i_episode != i_episode and not EVAL:
        prev_i_episode = i_episode
        checkpoint_path = model_name(directory, policy_name, policy_num, extension=False)
        policy_num += 1
        print("--------------------------------------------------------------------------------------------")
        print("saving model at : " + checkpoint_path)
        sac_agent.save_checkpoint(env_name, policy_name, suffix=str(i_episode))
        print("Model Saved - Elapsed Time  : ", datetime.now().replace(microsecond=0) - start_time)
        print("--------------------------------------------------------------------------------------------")
        
