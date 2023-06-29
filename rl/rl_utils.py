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
    state = einops.rearrange(state, 'ne (b h w) -> ne b h w', b=2, h=h, w=w)
    print(state)
    return state