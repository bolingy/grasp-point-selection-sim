import isaacgym
import isaacgymenvs
import torch
from rl.rl_utils import *
import matplotlib.pyplot as plt

ne = 50
img_x = 260
img_y = 180

envs = isaacgymenvs.make(
	seed=0,
	task="RL_UR16eManipulation_Nocam",
	num_envs=ne,
	sim_device="cuda:0",
	rl_device="cuda:0",
	multi_gpu=False,
	graphics_device_id=0, 
	headless=True
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
	#print('reward: ', reward)
	#print(('--------------------------------------------------'))
	if total_act % 10 == 0:
		print('current done', total_act)
	if total_act >= 100:
		break

	# show imgs from each env in indicies
	# for i in range(indicies.shape[0]):
	# if (indicies==0).sum():
	# 	i = torch.nonzero(indicies==0).item()
	# 	img = imgs[i, :img_x*img_y*2]
	# 	segmask = imgs[i, img_x*img_y:]
	# 	depth = imgs[i, :img_x*img_y]
	# 	plt.imshow(depth.cpu().numpy().reshape(img_y, img_x))
	# 	plt.show()
	# 	plt.imshow(segmask.cpu().numpy().reshape(img_y, img_x))
	# 	plt.show()

end = time.time()
print('total actions', total_act)
print(f"Time taken to run the code was {end-start} seconds")