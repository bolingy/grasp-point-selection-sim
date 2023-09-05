#!/usr/bin/env python

import isaacgym
import isaacgymenvs
import torch

import click
import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


def _get_data_path(bin_id,output_path):
    from datetime import datetime
    import random
    import string
    datetime_string = datetime.now().isoformat().replace(":", "")[:-7]
    random_string = ''.join(random.choice(string.ascii_letters)
                            for _ in range(6))

    os.makedirs(f"{output_path}", exist_ok=True)
    temp_path = f"{output_path}/{datetime_string}-{random_string}-grasp_data_v2_{bin_id}/"
    return os.path.expanduser(temp_path)



@click.command()
@click.option('--bin-id', type=click.Choice(['3H', '3E', '3F']), default='3F')
@click.option('--num-envs', default=50)
@click.option('--google-scanned-objects-path', default='assets/')
@click.option('--output-path', default='/tmp/')
def generate(bin_id, num_envs, google_scanned_objects_path, output_path):
    envs = isaacgymenvs.make(
        seed=0,
        task="UR16eManipulation",
        num_envs=num_envs,
        sim_device="cuda:0",
        rl_device="cuda:0",
        multi_gpu=False,
        headless=True,
        graphics_device_id=0,
        bin_id=bin_id,
        data_path=_get_data_path(bin_id, output_path),
        google_scanned_objects_path=google_scanned_objects_path,
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


if __name__ == '__main__':
    generate()
