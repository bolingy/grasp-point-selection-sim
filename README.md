# DYNAMO-GRASP Project

This project introduces DYNAMO-GRASP, a novel approach to the challenge of suction grasp point detection. By leveraging the power of physics-based simulation and data-driven modeling, DYNAMO-GRASP is capable of accounting for object dynamics during the grasping process. This significantly enhances a robot's ability to handle unseen objects in real-world scenarios.
Our method was benchmarked against established approaches through comprehensive evaluations in both simulated and real-world environments. It outperformed the alternatives by achieving a success rate of 98% in diverse simulated picking tasks and 80% in real-world, adversarially-designed scenarios.
DYNAMO-GRASP demonstrates a strong ability to adapt to complex and unexpected object dynamics, offering robust generalization to real-world challenges. The results of this research pave the way for more reliable and resilient robotic manipulation in intricate real-world situations.

This repository provides the codebase for collecting data through simulation. These scripts will help users to generate their own datasets, further enhancing the extensibility and usefulness of DYNAMO-GRASP.

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