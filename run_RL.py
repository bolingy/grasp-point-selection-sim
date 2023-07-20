import isaacgym
import isaacgymenvs
import torch
import rl.rl_utils
import matplotlib.pyplot as plt

ne = 2
img_x = 260
img_y = 180

envs = isaacgymenvs.make(
	seed=0,
	task="RL_UR16eManipulation",
	num_envs=ne,
	sim_device="cuda:0",
	rl_device="cuda:0",
	multi_gpu=False,
	graphics_device_id=0, 
	headless=False
)
# Observation space is eef_pos, eef_quat, q_gripper/q
print("Observation space is", envs.observation_space)
print("Action space is", envs.action_space)

def step_primitives_env_0(action):
	while True:
		obs, reward, done, info = envs.step(action)
		imgs = obs['obs'][:, :img_x*img_y*2]
		info = obs['obs'][:, img_x*img_y*2:]
		take_obs = info[0]
		if take_obs[0] == 1:
			print("@@@@@@@@@@@@observation returned")
			return imgs[0], torch.tensor([info[0][1]]), torch.tensor([info[0][2]])

def step_primitives(action):
	while True:
		obs, reward, done, info = envs.step(action)
		# if obs['obs'] is all zeros, then continue
		if torch.sum(obs['obs']) == 0:
			continue
		# print("obs", obs['obs'])
		imgs = obs['obs'][:, :img_x*img_y*2]
		info = obs['obs'][:, img_x*img_y*2:]
		# return all imgs that has info[env, 0] == 1
		# if all info[env, 0] == 0, then don't return anything
		indicies = info[:, 2]
		# select and return imgs with the selected indicies
		if imgs.shape[0] > 0:
			# print("info indicies", info[indicies])
			reward_temp = info[:, 0]
			done_temp = info[:, 1]
			# print("imgs, reward, done", imgs.shape, reward_temp.shape, done_temp.shape)
			return imgs, reward_temp, done_temp, indicies
			
action = torch.tensor(ne * [[0.12, -0.03, 0.28, 0.]]).to("cuda:0")
# imgs, reward, done = step_primitives(action)
# obs, reward, done, info = envs.step(action)

import time 
start = time.time()
for i in range(1000):
# while True:
	# obs, reward, done, info = envs.step(action)
	# print(obs['obs'][:, 614400:])
	imgs, reward, done, indicies = step_primitives(action)
	# img = imgs[:, :img_x*img_y*2]
	# segmask = imgs[:, img_x*img_y:]
	# depth = imgs[:, :img_x*img_y]
	# print("segmask", segmask.shape)
	# plt.imshow(depth[0].cpu().numpy().reshape(img_y, img_x))
	# plt.show()
	# plt.imshow(segmask[0].cpu().numpy().reshape(img_y, img_x))
	# plt.show()
end = time.time()
print(f"Time taken to run the code was {end-start} seconds")