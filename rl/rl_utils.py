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
import matplotlib.pyplot as plt


def rearrange_state(state, b=2, h=180, w=260):
    # state is a tensor of shape (batch_size, 307200 + 307200)
    # 307200 = 640 * 480
    # rearrange such that the depth image and seg mask are separate, and the batch dimension is first
    # (batch_size, 307200 + 307200) -> (batch_size, 2, 480, 640)

    # input: (ne, 614400)
    # output: (ne, 2, 480, 640)
	timestep = state[:, -1]
	state = state[:, :-1]
	state = einops.rearrange(state, 'ne (b h w) -> ne b h w', b=b, h=h, w=w)
	# make timestep of shape (ne, 1, h, w)
	timestep = timestep.unsqueeze(1).unsqueeze(2).unsqueeze(3).repeat(1, 1, h, w)
	# print("timestep: ", timestep.shape)
	state = torch.cat((state, timestep), dim=1)
	# print("state: ", state.shape)
	return state

def rearrange_state_timestep(state, b=2, h=180, w=260):
    # state is a tensor of shape (batch_size, 307200 + 307200)
    # 307200 = 640 * 480
    # rearrange such that the depth image and seg mask are separate, and the batch dimension is first
    # (batch_size, 307200 + 307200) -> (batch_size, 2, 480, 640)

    # input: (ne, 614400)
    # output: (ne, 2, 480, 640)
	timestep = state[:, -1]
	state = state[:, :-1]
	state = einops.rearrange(state, 'ne (b h w) -> ne b h w', b=b, h=h, w=w)
	# make timestep of shape (ne, 1, h, w)
	timestep = timestep.unsqueeze(1).unsqueeze(2).unsqueeze(3).repeat(1, 1, h, w)
	# print("timestep: ", timestep.shape)
	state = torch.cat((state, timestep), dim=1)
	# print("state: ", state.shape)
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
		imgs = obs['obs'][:, :-3]
		info = obs['obs'][:, -3:]
		# return all imgs that has info[env, 0] == 1
		# if all info[env, 0] == 0, then don't return anything
		indicies = info[:, 2].to(torch.int64)
		# select and return imgs with the selected indicies
		if imgs.shape[0] > 0:
			# print("info indicies", info[indicies])
			reward_temp = info[:, 0]
			done_temp = info[:, 1]
			# convert done to uint8
			done_temp = done_temp.to(torch.uint8)
			# print("imgs, reward, done", imgs.shape, reward_temp.shape, done_temp.shape)
			return imgs, reward_temp, done_temp, indicies
		
def step_primitives_env_0(action, envs):
	while True:
		obs, reward, done, info = envs.step(action)
		imgs = obs['obs'][:, :93600]
		info = obs['obs'][:, 93600:]
		take_obs = info[0]
		if take_obs[0] == 1:
			print("@@@@@@@@@@@@observation returned")
			return imgs[0], torch.tensor([info[0][1]]), torch.tensor([info[0][2]])

def scale_actions(action):
	action[:, 0] = action[:, 0] * 0.22 - 0.11
	action[:, 1] = action[:, 1] * 0.12 - 0.02
	action[:, 2] = action[:, 2] * 0.28
	if action.shape[1] == 4:
		action[:, 3] = action[:, 3] * 0.22 - 0.11
	return action

def create_env_action_via_true_indicies(true_indicies, action, actions, ne, sim_device):
	one_hot_vec = torch.zeros(ne).bool().to(sim_device)
	one_hot_vec[true_indicies] = 1
	actions[one_hot_vec] = action

def returns_to_device(state, reward, done, indicies, device):
	state = state.to(device)
	reward = reward.to(device)
	done = done.to(device)
	indicies = indicies.to(device)
	return state, reward, done, indicies
	
def model_name(directory, policy_name, version=None, extension=True):
	if version is None:
		checkpoint_path = directory + policy_name 
	else:
		checkpoint_path = directory + policy_name + '_' + str(version)
	if extension:
		checkpoint_path += '.pth'
	if not os.path.exists(checkpoint_path) and not extension:
		os.makedirs(checkpoint_path)
	return checkpoint_path

def get_real_ys_dxs(imgs, sim_device='cuda:0'):
	# returns a dictionary with each point corresponding to the real y and dx
	# returns 2 dictionaries, one for real y and one for real dx
	# example real y
	# real_y = {'1': 0.01, '2': 0.02, '3': 0.03, '4': -0.05} # where 1, 2 are the left and right sides of a segmask and its values are its corresponding offsets
	# example real dx
	# real_dx = {'1': 0.01, '2': 0.02, '3': 0.03} # where 1 is the object and its value is the dx of the object

	obj_offset = 20

	# get the seg mask and depth image
	img_x = 260
	img_y = 180
	img = imgs[:, :img_x*img_y*2]
	depth = img[:, :img_x*img_y]
	seg = img[:, img_x*img_y:]
	depth = depth.reshape(-1, img_y, img_x)
	seg = seg.reshape(-1, img_y, img_x)

	# print("batch size: ", depth.shape[0])
	# get the real ys
	unique_ids = 3
	# print("unique_ids: ", unique_ids)
	# pick the left and right most points of each seg mask
	real_y = torch.zeros((depth.shape[0], unique_ids, 2))	
	real_dx = torch.zeros((depth.shape[0], unique_ids))

	for i in range(depth.shape[0]):
		unique_ids_env = torch.unique(seg[i])
		unique_ids_env = unique_ids_env[unique_ids_env != 0]
		for j in range(unique_ids_env.shape[0]):
			# get the points of the seg mask
			seg_mask = seg[i] == unique_ids_env[j]
			# get the x and y coordinates of the seg mask
			y, x = torch.where(seg_mask)
			# print("y: ", y)
			# print("x: ", x)
			
			# get the left most point
			left_most = torch.min(x)
			# get the right most point
			right_most = torch.max(x)
			real_y[i, j, 0] = left_most - obj_offset
			real_y[i, j, 1] = right_most + obj_offset

			# get the depth of the center of the seg mask
			real_dx[i, j] = depth[i, y[0], x[0]]

	# print("real_y: ", real_y)
	# print("real_dx: ", real_dx)
	# sort by ascending left most point

	# Extract the first column values
	first_column = real_y[:, :, 0]

	# Get the indices that would sort the first column values
	sorted_indices = torch.argsort(first_column, dim=1)

	# Use the sorted indices to rearrange the original tensor
	real_y = torch.gather(real_y, 1, sorted_indices.unsqueeze(-1).expand(-1, -1, real_y.size(-1)))
	# sort real dx such that it matches the sorted real y
	real_dx = torch.gather(real_dx, 1, sorted_indices)

	# rearrange to batch_size, 2 x num_seg_masks
	real_y = real_y.reshape(-1, 2*unique_ids)
	# print("unique_ids: ", unique_ids)
	# print("real_y: ", real_y)
	# print("Real dx: ", real_dx)
	# plt.imshow(seg[0].cpu())
	# plt.show()
	# plt.imshow(seg[1].cpu())
	# plt.show()
	# plt.imshow(depth[0].cpu())
	# plt.show()
	# plt.imshow(depth[1].cpu())
	# plt.show()
	return real_y.to(sim_device), real_dx.to(sim_device)

def convert_actions(action, real_y, real_dx, sim_device='cuda:0'):
	# action will be in the format of [batchsize, y (1-6), dy]
	# return in the form of [batchsize, real_y[y], z, real_dx[dx], dy]
	# action = torch.tensor([[1, 1, 1], [2, 2, 2]]).to("cuda")
	# action[:, 0] = torch.round(action[:, 0] * 5.0)
	# action[:, 0] = 0
	# print("action: ", action)
	action = action.to(sim_device)
	real_y = real_y.to(sim_device)
	real_dx = real_dx.to(sim_device)
	z = torch.ones((action.shape[0], 1)).to(sim_device) * -0.05
	# select the real y in each batch using the indicies specified in action[:, 0]
	result_y = real_y.gather(1, action[:, 0].long().unsqueeze(1))
	result_dx = (- real_dx.gather(1, (action[:, 0].long().unsqueeze(1)) // 2)) - 0.95
	sign = torch.ones(action.shape[0]).to(sim_device)
	sign[action[:, 0] % 2 == 1] = -1
	action[:, 1] = action[:, 1] * 0.11 * sign
	result_y = (result_y - 40) / 260 * (-0.35) + 0.12
	result = torch.cat((result_y, z, result_dx, action[:, 1].unsqueeze(1)), dim=1)
	# print("result: ", result)
	return result
