# import cv2
import glob
from itertools import chain
import os
import random
import zipfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from linformer import Linformer
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from tqdm.notebook import tqdm

from vit_pytorch.efficient import ViT
import json
print(f"Torch: {torch.__version__}")
print(torch.cuda.is_available())


bin_crop_dim = 464
# Training settings
batch_size = 128
epochs = 50000  # 20
lr = 5e-8
gamma = 0.7
seed = 42

def add_padding(image, channels, value):
    # Create padding
    padding = int(bin_crop_dim/2)
    # Define padding for height, width and channels
    if(channels == 3):
        padding_width = ((padding, padding), (padding, padding), (0, 0))
    elif(channels == 1):
        padding_width = ((padding, padding), (padding, padding))
    # Apply padding
    padded_img = np.pad(image, padding_width, mode='constant', constant_values=value)
    return padded_img


def visualize_point_cloud(point_cloud):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = point_cloud[:, :, 0].flatten()
    y = point_cloud[:, :, 1].flatten()
    z = point_cloud[:, :, 2].flatten()
    ax.scatter(x, y, z, s=0.5)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()


def depth_to_point_cloud(depth_image):
    cx = depth_image.shape[1]/2
    cy = depth_image.shape[0]/2
    fx = 914.0148
    fy = 914.0147
    height, width = depth_image.shape
    x = np.linspace(0, width-1, width)
    y = np.linspace(0, height-1, height)
    x, y = np.meshgrid(x, y)
    normalized_x = (x - cx) / fx
    normalized_y = (y - cy) / fy
    z = depth_image
    x = normalized_x * z
    y = normalized_y * z
    point_cloud = np.dstack((x, y, z))
    return point_cloud

def add_flap(image, channels, value, flap_dim, flap_rot, top_left_x_pad, bottom_right_x_pad, crop_coords, center_y):
    
    x, y = np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[0]))
    m = flap_rot/(bottom_right_x_pad-top_left_x_pad)
    c = int((image.shape[1] - (int(bin_crop_dim/2) - (crop_coords[1]-center_y))) - flap_dim) - m*top_left_x_pad
    flap_line = m*x + c
    bottom_bin_line = int((image.shape[1] - (int(bin_crop_dim/2) - (crop_coords[1]-center_y))))

    mask = (y >= flap_line) & (y <= bottom_bin_line)

    if(channels == 3):
        image[mask, :] = value
    image[mask] = value
    return image

path = "/home/aurmr/workspaces/manipulation_policy/src/System_Identification_Data/Parallelization-Data-test-center-new/3F/26/depth_image_26_1.npy"

parts_slashed = path.split("/")
parts_ = parts_slashed[-1].split("_")
parts_dot = parts_[-1].split(".")

print("/home/aurmr/workspaces/manipulation_policy/src/System_Identification_Data/Parallelization-Data-test-center-new/3F/" +
      str(parts_slashed[-2])+"/"+"json_data"+"_*"+"_"+str(parts_dot[0])+".json")
data_list = glob.glob("/home/aurmr/workspaces/manipulation_policy/src/System_Identification_Data/Parallelization-Data-test-center-new/3F/" +
                      str(parts_slashed[-2])+"/"+"json_data"+"_*"+"_"+str(parts_dot[0])+".json", recursive=True)
print(data_list)

with open(data_list[0]) as json_file:
    data_id = json.load(json_file)


id = data_id['object_id']

depth_path = path
parts_slashed = depth_path.split("/")
parts_ = parts_slashed[-1].split("_")
segmask_path = "/home/aurmr/workspaces/manipulation_policy/src/System_Identification_Data/Parallelization-Data-test-center-new/3F/" + \
    str(parts_slashed[-2])+"/segmask_"+parts_[-2]+"_"+parts_[-1]
depth_image = np.load(depth_path)

rgb_path = "/home/aurmr/workspaces/manipulation_policy/src/System_Identification_Data/Parallelization-Data-test-center-new/3F/" + \
    str(parts_slashed[-2])+"/rgb_"+parts_[-2]+"_"+parts_[-1]


segmask = np.load(segmask_path)
rgb = np.load(rgb_path)


plt.imshow(depth_image)
plt.show()

segmask_numpy = np.zeros_like(segmask)
segmask_numpy[segmask == id] = 1

depth_mask = np.zeros_like(segmask)
depth_mask[segmask != 0] = 1


y, x = np.where(segmask_numpy == 1)
try:
    center_x, center_y = np.mean(x), np.mean(y)

    top_left_x = int(center_x)-int(bin_crop_dim/2)
    top_left_y = int(center_y)-int(bin_crop_dim/2)
    bottom_right_x = int(center_x)+int(bin_crop_dim/2)
    bottom_right_y = int(center_y)+int(bin_crop_dim/2)
except:
    print(id)
    plt.imshow(segmask)
    plt.show()
    print(x, y)
    print(path)
    print(data_list)
    print(np.unique(segmask_numpy))
    print(center_x, center_y)


crop_coords = [117, 338, 207, 436]
depth_image[depth_mask != 1] = 0.925

# depth_image[crop_coords[0]:crop_coords[1], crop_coords[2]-2:crop_coords[2]+2] = 0.771
# depth_image[crop_coords[0]-2:crop_coords[0]+2, crop_coords[2]:crop_coords[3]] = 0.771
# depth_image[crop_coords[1]-2:crop_coords[1]+2, crop_coords[2]:crop_coords[3]] = 0.771
# depth_image[crop_coords[0]:crop_coords[1], crop_coords[3]-2:crop_coords[3]+2] = 0.771
depth_image[:crop_coords[0], :] = 0.771
depth_image[:, :crop_coords[2]] = 0.771
depth_image[crop_coords[1]:, :] = 0.771
depth_image[:, crop_coords[3]:] = 0.771

top_left_x_pad = int(center_x)-int(bin_crop_dim/2)+int(bin_crop_dim/2)
top_left_y_pad = int(center_y)-int(bin_crop_dim/2)+int(bin_crop_dim/2)
bottom_right_x_pad = int(center_x)+int(bin_crop_dim/2)+int(bin_crop_dim/2)
bottom_right_y_pad = int(center_y)+int(bin_crop_dim/2)+int(bin_crop_dim/2)


depth_pad = add_padding(depth_image, 1, 0.771)
depth_processed = depth_pad[top_left_y_pad:bottom_right_y_pad,
                                  top_left_x_pad:bottom_right_x_pad]

flap_dim = 36
flap_rot = int(np.random.uniform(-5, 5, 1)[0])

depth_processed = add_flap(depth_processed, 1, 0.771, flap_dim, flap_rot, top_left_x_pad, bottom_right_x_pad, crop_coords, center_y)

plt.imshow(depth_processed)
plt.show()

segmask_pad = add_padding(segmask_numpy, 1, 0)
segmask_processed = segmask_pad[top_left_y_pad:bottom_right_y_pad,
                                  top_left_x_pad:bottom_right_x_pad]

segmask_processed = add_flap(segmask_processed, 1, 0, flap_dim, flap_rot, top_left_x_pad, bottom_right_x_pad, crop_coords, center_y)
rgb_mask = np.zeros_like(rgb)
rgb_mask[crop_coords[0]:crop_coords[1] ,crop_coords[2]:crop_coords[3], :] = rgb[crop_coords[0]:crop_coords[1] ,crop_coords[2]:crop_coords[3], :]

rgb_pad = add_padding(rgb_mask, 3, 0)
rgb_processed = rgb_pad[top_left_y_pad:bottom_right_y_pad,
                                  top_left_x_pad:bottom_right_x_pad]

rgb_processed = add_flap(rgb_processed, 3, 0, flap_dim, flap_rot, top_left_x_pad, bottom_right_x_pad, crop_coords, center_y)
plt.imshow(rgb_processed)
plt.show()

point_cloud = depth_to_point_cloud(depth_processed)
visualize_point_cloud(point_cloud)
segmask_processed = np.expand_dims(segmask_processed, axis=-1)
input_data = np.concatenate((point_cloud, segmask_processed), axis=-1)
input_data = input_data.astype(np.double)
trans = transforms.Compose([transforms.ToTensor()])
input_transformed = trans(input_data)

parts_slashed = path.split("/")
parts_ = parts_slashed[-1].split("_")
parts_dot = parts_[-1].split(".")

visualize_label = segmask[top_left_y:bottom_right_y, top_left_x:bottom_right_x]
visualize_label_raw = segmask_numpy
label_image = np.ones((bin_crop_dim, bin_crop_dim))*(-100)
for json_files in data_list:
    with open(json_files) as json_file:
        grasp_data = json.load(json_file)
    id = grasp_data['object_id']
    print(grasp_data['grasp point'])
    grasp_point = np.array(grasp_data['grasp point'])

    label = int(grasp_data['success'])*2
    print(label)
    if (label == 0):
        label = 1

    if (int(grasp_data['oscillation'])):
        label = 1
    elif (int(grasp_data['penetration'])):
        label = 2

    visualize_label_raw[int(grasp_point[1])][int(
        grasp_point[0])] = label*50 + 100
    grasp_point[0] = bin_crop_dim/2 + (grasp_point[0] - center_x)
    grasp_point[1] = bin_crop_dim/2 + (grasp_point[1] - center_y)

    label_image[int(grasp_point[1])][int(grasp_point[0])] = label
    visualize_label[int(grasp_point[1])][int(grasp_point[0])] = label*50 + 100

visualize_label = add_flap(visualize_label, 1, 0, flap_dim, flap_rot, top_left_x_pad, bottom_right_x_pad, crop_coords, center_y)
label_image = add_flap(label_image, 1, -100, flap_dim, flap_rot, top_left_x_pad, bottom_right_x_pad, crop_coords, center_y)    
plt.imshow(visualize_label)
plt.show()

count = 0