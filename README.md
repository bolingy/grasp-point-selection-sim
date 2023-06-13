# Project Title

This project is designed to 

## Table of Contents

1. [Installation](#installation)
2. [Running the Project](#running)
3. [Alternate Python Version](#python38)

## Installation

Follow these steps for the installation:

```bash
# Create a conda environment with Python version 3.7
conda create -n myenv python=3.7
conda activate myenv
```

Download the Isaac Gym package: [link](https://developer.nvidia.com/isaac-gym/download)

```bash
# Install Isaac Gym
# In the python sub directory, run:
pip install -e .

# Install Isaac Gym Envs
# clone the repository,
git clone https://github.com/NVIDIA-Omniverse/IsaacGymEnvs

#Install this repo,
pip install -e .

# Install the GQCNN dependencies
# In the gqcnn subdirectory, run:
pip install .

# Uninstall pyglet
pip uninstall pyglet

# Install the following dependencies
pip install open3d pyglet==1.4.10
```

## Running
To run the project, use the data_collection.py script:
```bash
python data_collection.py
```
## python38
If you would prefer to use Python 3.8 for this project, switch to the py38 branch of the repository:
```bash
git checkout py38
```