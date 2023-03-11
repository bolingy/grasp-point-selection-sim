import isaacgym
import isaacgymenvs
import torch

from isaacgym import gymapi

ne = 20

envs = isaacgymenvs.make(
	seed=0,
	task="FrankaCubeStack", # Aurmar, FrankaCabinet
	num_envs=ne,
	sim_device="cuda:0",
	rl_device="cuda:0",
	graphics_device_id=0
)
print("Observation space is", envs.observation_space)
print("Action space is", envs.action_space)
obs = envs.reset()
gym = gymapi.acquire_gym()

while True:
	action = torch.tensor(ne * [[0.5, 0, 0, 0, 0, 0, 1]], device="cuda:0")
	obs, reward, done, info = envs.step(action)
	gymapi.Transform()