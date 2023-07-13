import isaacgym
import isaacgymenvs
import torch

ne = 50

from pathlib import Path
cur_path = str(Path(__file__).parent.absolute())

envs = isaacgymenvs.make(
	seed=0,
	task="UR16eManipulation",
	num_envs=ne,
	sim_device="cuda:0",
	rl_device="cuda:0",
	multi_gpu=True,
	graphics_device_id=0,
	data_path=cur_path+"/../System_Identification_Data/Parallelization-Data/",
)
print("Observation space is", envs.observation_space)
print("Action space is", envs.action_space)
obs = envs.reset()

while True:
	action = torch.tensor(ne * [[0.1, 0, 0, 0, 0, 0, 1]])
	obs, reward, done, info = envs.step(
		action
		)