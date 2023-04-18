# How to traverse the code

## For Ranking method:
RobustGraspingPolicy --> GraspingPolicy --> ImageGraspSamplerFactory --> DepthImageSuctionPointSampler --> ImageGraspSampler

## For Cross Entropy Method:
CrossEntropyRobustGraspingPolicy --> GraspingPolicy --> ImageGraspSamplerFactory --> 

## Installtion changes
Create conda environment with python version 3.7
1. Install the isaac gym using the isaacgym repo
2. Install isaac_gym_envs
3. Again install isaac_gym to build isaac gym
4. Install gqcnn depdencies
5. Uninstall pyglet
6. Installl dependencies: open3d, pyglet==1.4.10

Extras:
1. pip uninstall gqcnn
2. pip install open3d