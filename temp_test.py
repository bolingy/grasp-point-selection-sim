import isaacgym
import isaacgymenvs
import torch

ne = 1

envs = isaacgymenvs.make(
	seed=0,
	task="FrankaCubeStack", # Aurmar, FrankaCabinet
	num_envs=ne,
	sim_device="cuda:0",
	rl_device="cuda:0",
	multi_gpu=True,
	headless=True,
	graphics_device_id=0
)
print("Observation space is", envs.observation_space)
print("Action space is", envs.action_space)
obs = envs.reset()
while True:
	action = torch.tensor(ne * [[0.1, 0, 0, 0, 0, 0, 1]])
	obs, reward, done, info = envs.step(
		#torch.rand((ne,)+envs.action_space.shape, device="cuda:0")
		action
		)