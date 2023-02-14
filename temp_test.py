import isaacgym
import isaacgymenvs
import torch

ne = 2

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
while True:
	obs, reward, done, info = envs.step(
		torch.rand((ne,)+envs.action_space.shape, device="cuda:0")
	)