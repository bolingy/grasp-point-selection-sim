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


def rearrange_state(state, b=2, h=480, w=640):
    # state is a tensor of shape (batch_size, 307200 + 307200)
    # 307200 = 640 * 480
    # rearrange such that the depth image and seg mask are separate, and the batch dimension is first
    # (batch_size, 307200 + 307200) -> (batch_size, 2, 480, 640)

    # input: (ne, 614400)
    # output: (ne, 2, 480, 640)
    state = einops.rearrange(state, 'ne (b h w) -> ne b h w', b=b, h=h, w=w)
    return state


# def step_primitives(action, envs):
# 	while True:
# 		obs, reward, done, info = envs.step(action)
# 		imgs = obs['obs'][:, :614400]
# 		info = obs['obs'][:, 614400:]
# 		# if imgs contain nan, then make it 0
# 		# imgs[torch.isnan(imgs)] = 255

# 		# vector  num_envs long of info[:, 0] called indicies
# 		indicies = info[:, 0].bool()
# 		# print('indicies: ', indicies)
# 		# if indicies are not empty, then return the imgs
# 		if torch.sum(indicies) > 0:
# 			reward_temp = info[:, 1]
# 			done_temp = info[:, 2]
# 			# print("observation returned")
# 			# print all observations of env 0
# 			# print('imgs: ', imgs[0])
# 			# if indicies[0] == 1:
# 				# print('reward: ', reward_temp[0])
# 				# print('done: ', done_temp[0])
# 			return imgs, reward_temp, done_temp, indicies

def step_primitives(action, envs):
	while True:
		obs, reward, done, info = envs.step(action)
		# if obs['obs'] is all zeros, then continue
		if torch.sum(obs['obs']) == 0:
			continue
		# print("obs", obs['obs'])
		imgs = obs['obs'][:, :614400]
		info = obs['obs'][:, 614400:]
		# return all imgs that has info[env, 0] == 1
		# if all info[env, 0] == 0, then don't return anything
		indicies = info[:, 2].to(torch.int64)
		# select and return imgs with the selected indicies
		if imgs.shape[0] > 0:
			# print("info indicies", info[indicies])
			reward_temp = info[:, 0]
			done_temp = info[:, 1]
			# print("imgs, reward, done", imgs.shape, reward_temp.shape, done_temp.shape)
			return imgs, reward_temp, done_temp, indicies
		
def step_primitives_env_0(action, envs):
	while True:
		obs, reward, done, info = envs.step(action)
		imgs = obs['obs'][:, :614400]
		info = obs['obs'][:, 614400:]
		take_obs = info[0]
		if take_obs[0] == 1:
			print("@@@@@@@@@@@@observation returned")
			return imgs[0], torch.tensor([info[0][1]]), torch.tensor([info[0][2]])

def scale_actions(action):
	action[:, 0] = action[:, 0] * 0.22 - 0.11
	action[:, 1] = action[:, 1] * 0.12 - 0.02
	action[:, 2] = action[:, 2] * 0.28
	action[:, 3] = action[:, 3] * 0.44 - 0.22
	return action

def returns_to_device(state, reward, done, indicies, device):
	state = state.to(device)
	reward = reward.to(device)
	done = done.to(device)
	indicies = indicies.to(device)
	return state, reward, done, indicies
	