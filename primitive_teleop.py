import isaacgym
import isaacgymenvs
import torch
import rl.rl_utils

ne = 2

envs = isaacgymenvs.make(
	seed=0,
	task="RL_UR16eManipulation",
	num_envs=ne,
	sim_device="cuda:0",
	rl_device="cuda:0",
	multi_gpu=False,
	graphics_device_id=0
)
# Observation space is eef_pos, eef_quat, q_gripper/q
print("Observation space is", envs.observation_space)
print("Action space is", envs.action_space)

def step_primitives_env_0(action):
	while True:
		obs, reward, done, info = envs.step(action)
		imgs = obs['obs'][:, :614400]
		info = obs['obs'][:, 614400:]
		take_obs = info[0]
		if take_obs[0] == 1:
			print("@@@@@@@@@@@@observation returned")
			return imgs[0], torch.tensor([info[0][1]]), torch.tensor([info[0][2]])

def step_primitives(action):
	while True:
		obs, reward, done, info = self.envs.step(action)
		imgs = obs['obs'][:, :614400]
		info = obs['obs'][:, 614400:]
		# return all imgs that has info[env, 0] == 1
		# if all info[env, 0] == 0, then don't return anything
		# select and return imgs with the selected indicies
		indicies = torch.where(info[:, 0] == 1)
		# if indicies are not empty, then return the imgs
		if indicies[0].shape[0] > 0:
			reward_temp = info[:, 1]
			done_temp = info[:, 2]
			return imgs, reward_temp, done_temp, indicies
			

while True:
	# action is specified by forces (6 dof) and gripper (1 dof)
	# the last two values are motion primitive parameters
	# -2 index: 0 = right, 1 = down, 2 = left, 3 = up
	# -1 index: distance to move (in meters)

	# TODO: Pass RL flag in to envs.step() so that the RL flag is on for i time steps
	action = torch.tensor(ne * [[0.11, 0., 0.28, 0.22]]).to("cuda:0")
	obs, reward, done = step_primitives(action)
	# obs, reward, done, info = envs.step(action)
