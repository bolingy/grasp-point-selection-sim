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


def step_primitives(action, envs):
	while True:
		obs, reward, done, info = envs.step(action)
		# print(obs['obs'][:, 614400:])
		info = (obs['obs'][614400:])
		obs = obs['obs'][:614400]
		take_obs = info[0]
		if take_obs == 1:
			print("observation returned")
			print(obs.shape)
			print(info.shape)
			return obs, reward, done, info

def clip_actions(action):
	action[:, 0] = action[:, 0] * 0.22 - 0.11
	action[:, 1] = action[:, 1] * 0.12 - 0.02
	action[:, 2] = action[:, 2] * 0.28
	action[:, 3] = action[:, 3] * 0.44 - 0.22
	return action