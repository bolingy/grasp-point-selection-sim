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
# Create a conda environment with Python version 3.8
conda create -n myenv python=3.8
conda activate myenv
```

Download the Isaac Gym package: [link](https://developer.nvidia.com/isaac-gym/download)

```bash
# Install Isaac Gym
# In the python sub directory, run:
pip install -e .

pip install attrs
pip install openai

# Install Isaac Gym Envs
# clone the repository,
git clone https://github.com/NVIDIA-Omniverse/IsaacGymEnvs

#Install this repo,
pip install -e .

pip install flask

# Install the GQCNN dependencies
# In the gqcnn subdirectory, run:
pip install -e .

pip install open3d

# Install tensorflow 2 for python 3.8 version
pip install tensorflow==2.12.0 tensorflow-estimator==2.12.0 tensorflow-io-gcs-filesystem==0.32.0

# Install cudatoolkit and cudnn libraries
conda install cudatoolkit==11.8.0 -c nvidia
conda install cudnn

pip install --upgrade google-api-python-client google-auth-httplib2 google-auth-oauthlib

# Also export the LD_LIBRARY_PATH for cuda
export LD_LIBRARY_PATH=path/to/miniconda3/envs/isaac_ws_py38/lib

# Outside this repo create two folders for saving the data(grasp point properties, depth image, segmentation mask and rgb image),
mkdir System_Identification_Data && cd System_Identification_Data
mkdir Parallelization-Data
```

## Running
To run the project, use the kernel/object_spawning_kernel.py script for spawning 10 environments and 30 unique objects:
```bash
python kernel/object_spawning_kernel.py --bin-id 3F --num-envs 10 --objects-spawn 30
```

Also download the google scanned objects from this link and paste it outside the `grasp-point-selection-sim` git folder: [google scanned objects link](https://drive.google.com/drive/folders/1uDtTad67tJ3GwaPDPTvR5TWf1iArYeq1?usp=sharing)
Folder name is `Google Scanned Objects`


## python38
If you would prefer to use Python 3.7 for this project, switch to the parallelization branch of the repository:
```bash
git checkout parallelization
```