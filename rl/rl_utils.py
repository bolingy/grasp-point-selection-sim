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


def rearrange_state(state, h=480, w=640):
    depth_img, seg_mask = state[:, :307200], state[:, 307200:]
    depth_img = einops.rearrange(depth_img, 'b (h w) -> b h w', h=h, w=w)
    seg_mask = einops.rearrange(seg_mask, 'b (h w) -> b h w', h=h, w=w)
    return torch.cat((depth_img, seg_mask), axis=0)