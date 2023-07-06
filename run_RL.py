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
		obs, reward, done, info = envs.step(action)
		imgs = obs['obs'][:, :614400]
		info = obs['obs'][:, 614400:]
		# return all imgs that has info[env, 0] == 1
		# if all info[env, 0] == 0, then don't return anything
		indicies = torch.where(info[:, 0] == 1)
		# select and return imgs with the selected indicies
		imgs = imgs[indicies]
		if imgs.shape[0] > 0:
			# print("info indicies", info[indicies])
			reward_temp = info[indicies][:, 1]
			done_temp = info[indicies][:, 2]
			# print("@@@@@@@@@@@@@@@")
			# print("imgs, reward, done", imgs.shape, reward_temp.shape, done_temp.shape)
			return imgs, reward_temp, done_temp
			
action = torch.tensor(ne * [[0.11, 0., 0.28, 0.22]]).to("cuda:0")
imgs, reward, done = step_primitives(action)
while True:
	for i in imgs:
		isaacgym.viewer.window.show_img(i)
	isaacgym.viewer.window.render_buffer()
	
	imgs, reward, done = step_primitives(action)
