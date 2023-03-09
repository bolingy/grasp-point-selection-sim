import isaacgym
import isaacgymenvs
import torch

from isaacgym import gymapi

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
gym = gymapi.acquire_gym()

while True:
	action = torch.tensor([[0.45, -2,  1.4973, -0.0224, -0.6976,  0.0211, -0.7159],
        [0, 0, 0, 0.1150, -0.7673,  0.5651, -0.2806]],
       device='cuda:0')
	# action = torch.tensor([[ 0.0600, -2.5000,  2.0300,  0.5800,  1.6700,  1.7400],
    #     [ 0.500, 0.000,  0.00,  0, 0, 0]],
    #    device='cuda:0')
      
	# obs, reward, done, info = envs.step(
	# 	# torch.rand((ne,)+envs.action_space.shape, device="cuda:0")
	# 	action
	# )
	obs, reward, done, info = envs.step(action)

	gymapi.Transform()