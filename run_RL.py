import isaacgym
import isaacgymenvs
import torch
from rl.rl_utils import *
import matplotlib.pyplot as plt

ne = 2
res_net = True
envs = isaacgymenvs.make(
	seed=0,
	task="RL_UR16eManipulation_Full",
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
			
action = torch.tensor(ne * [[0.5, 0.5, 0.2, 0.2]]).to("cuda:0")
action = scale_actions(action).to("cuda:0")
print('scaled action', action)

import time 
start = time.time()
# for i in range(1000):
total_act = 0
while True:
	# obs, reward, done, info = envs.step(action)
	# print(obs['obs'][:, 614400:])
	imgs, reward, done, indicies = step_primitives(action, envs)
	total_act += indicies.shape[0]
	#print('indicies: ', indicies.shape[0])
	print('reward: ', reward)
	#print(('--------------------------------------------------'))
	if total_act % 10 == 0:
		print('current done', total_act)
	if total_act >= 100:
		break

	# show imgs from each env in indicies
	# for i in range(indicies.shape[0]):
	if (indicies[0] == 0) and res_net == True:
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
	elif (indicies[0] == 0) and res_net == False:
		state_env_0 = imgs[0]
		print("state env 0", state_env_0)
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

end = time.time()
print('total actions', total_act)
print(f"Time taken to run the code was {end-start} seconds")