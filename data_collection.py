#!/usr/bin/env python
import isaacgym
import isaacgymenvs
import torch

import click
import os
import subprocess

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

conda_env_path = os.environ.get('CONDA_PREFIX')

if conda_env_path:
    lib_path = os.path.join(conda_env_path, 'lib')
    os.environ["LD_LIBRARY_PATH"] = lib_path
else:
    print("Warning: CONDA_PREFIX is not set. Are you sure Conda environment is activated?")

subprocess.run(f"export LD_LIBRARY_PATH={lib_path}", shell=True, executable="/bin/bash")

def _get_data_path(bin_id):
    from datetime import datetime
    import random
    import string

    datetime_string = datetime.now().isoformat().replace(":", "")[:-7]
    random_string = "".join(random.choice(string.ascii_letters) for _ in range(6))
    temp_path = f"/home/soofiyan_ws/Documents/Issac_gym_ws/System_Identification_Data/{datetime_string}-{random_string}-grasp_data_{bin_id}/"
    return os.path.expanduser(temp_path)


@click.command()
@click.option("--bin-id", type=click.Choice(["3H", "3E", "3F"]), default="3F")
@click.option("--num-envs", default=50)
def generate(bin_id, num_envs):
    envs = isaacgymenvs.make(
        seed=0,
        task="UR16eManipulation",
        num_envs=num_envs,
        sim_device="cuda:0",
        rl_device="cuda:0",
        multi_gpu=True,
        headless=True,
        graphics_device_id=0,
        bin_id=bin_id,
        data_path=_get_data_path(bin_id),
    )
    print("Observation space is", envs.observation_space)
    print("Action space is", envs.action_space)
    obs = envs.reset()

    try:
        while True:
            action = torch.tensor(num_envs * [[0.1, 0, 0, 0, 0, 0, 1]])
            obs, reward, done, info = envs.step(action)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    generate()
