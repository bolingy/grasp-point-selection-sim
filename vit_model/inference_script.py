
from __future__ import print_function

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

from torch import nn, einsum
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

import sys

from vit_pytorch.efficient import ViT
from sklearn.cluster import DBSCAN
import json
print(f"Torch: {torch.__version__}")
print(torch.cuda.is_available())

# Training settings
batch_size = 128
epochs = 1000 #20
lr = 5e-4
gamma = 0.7
seed = 42
image_size_h=256
image_size_w=256
patch_size=16
num_classes=2
channels=4
dim = 128#1024
depth = 3
heads = 16
mlp_dim = 128
dropout = 0
emb_dropout = 0

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

seed_everything(seed)
device = 'cuda:0'

# augmentation
train_transforms = transforms.Compose(
    [
        #transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]
)

test_transforms = transforms.Compose(
    [
        transforms.ToTensor(),
    ]
)

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

class ViT_grasp(nn.Module): 
    def __init__(self, *, image_size_h, image_size_w, patch_size, num_classes, dim,
                  depth, heads, mlp_dim, grasp_point_dim=2, pool = 'cls', channels = 4,
                    dim_head = 64, dropout = 0., emb_dropout = 0., transformer):
        super().__init__()
        image_height, image_width = pair(image_size_h)
        patch_height, patch_width = pair(patch_size)
        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        assert 1 % 1 == 0, 'Frames must be divisible by frame patch size'
        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.transformer = transformer
        self.pool = pool
        self.to_latent = nn.Identity()
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 32*32)
        )
        # Deconvolution layers
        self.deconv1 = nn.ConvTranspose2d(in_channels=1, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv2 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv3 = nn.ConvTranspose2d(in_channels=32, out_channels=1, kernel_size=3, stride=2, padding=1, output_padding=1)
        # Activation and Batchnorm
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(32)


    def forward(self, img):
        img = img.type(torch.float)
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.transformer(x)
        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]
        x = self.to_latent(x)
        x = self.mlp_head(x)
        x = self.relu(x)
        x = x.view(-1, 1, 32, 32)
        x = self.deconv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.deconv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.deconv3(x)
        x = torch.sigmoid(x)
        return x


class GraspDataset(Dataset):
    def __init__(self, file_list, transform=None):
        self.file_list = file_list
        self.transform = transform

    def __len__(self):
        self.filelength = len(self.file_list)
        return self.filelength


class inference:
    def __init__(self):
        efficient_transformer = Linformer(
            dim=dim,
            seq_len=256+1,
            depth=12,
            heads=8,
            k=64
        )
        
        self.model = ViT_grasp(
            image_size_h=image_size_h,
            image_size_w=image_size_w,
            patch_size=patch_size,
            num_classes=num_classes,
            channels=channels,
            dim=dim,
            depth=depth,
            heads=heads,
            mlp_dim=mlp_dim,
            dropout=dropout,
            emb_dropout=emb_dropout,
            transformer=efficient_transformer
        ).to(device)

    def depth_to_point_cloud(self, depth_image):
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
    
    def run_model(self, depth_image, segmask, target_id, model_path):
        try:
            self.model.load_state_dict(torch.load(model_path, map_location='cuda:0')['model_state_dict'])
            self.model.eval()
        except:
            self.model.load_state_dict(torch.load(model_path, map_location='cuda:0'))
            self.model.eval()
      
        id = target_id.item()
        segmask_numpy = np.zeros_like(segmask)
        segmask_numpy[segmask == id] = 1

        depth_mask = np.zeros_like(segmask)
        depth_mask[segmask != 0] = 1

        y, x = np.where(segmask_numpy == 1)

        center_x, center_y = np.mean(x), np.mean(y)

        top_left_x = int(center_x)-128
        top_left_y = int(center_y)-128

        centroid = np.array([top_left_x, top_left_y])
        bottom_right_x = int(center_x)+128
        bottom_right_y = int(center_y)+128

        depth_image[depth_mask != 1] = 1.225

        depth_image[156:346, 235:238] = 1.071
        depth_image[155:158, 236:431] = 1.071
        depth_image[345:348, 235:431] = 1.071
        depth_image[156:346, 430:433] = 1.071
        depth_image[:156, :] = 1.071
        depth_image[:, :236] = 1.071
        depth_image[346:, :] = 1.071
        depth_image[:, 431:] = 1.071

        depth_processed = depth_image[top_left_y:bottom_right_y, top_left_x:bottom_right_x]
        segmask_processed = segmask_numpy[top_left_y:bottom_right_y, top_left_x:bottom_right_x]

        point_cloud = self.depth_to_point_cloud(depth_processed)

        segmask_processed = np.expand_dims(segmask_processed, axis=-1)
        input_data = np.concatenate((point_cloud, segmask_processed), axis=-1)

        trans = transforms.Compose([transforms.ToTensor()])
        input_transformed = trans(input_data)

        input_transformed = input_transformed.unsqueeze(0).type(torch.float).to(device)
        label_image = np.ones((256, 256))*(-100)

        output = self.model(input_transformed)
        output = torch.squeeze(output)

        output = output.unsqueeze(2)
        output = output.cpu().detach().numpy()*segmask_processed

        max_value = np.amax(output)
        grasp_point = None
        if(max_value <= 0.7):
            max_coordinates = np.argwhere(output == max_value)
            grasp_point = np.array([max_coordinates[0][1], max_coordinates[0][0]])
        else:
            try:
                points = np.column_stack(np.where(output > 0.9))
                # Perform DBSCAN on the points
                db = DBSCAN(eps=1.5, min_samples=5).fit(points)  # You may need to adjust the parameters
                # Find the labels of the clusters that each point belongs to
                labels = db.labels_
                # Ignore noises in the cluster computation (noises are denoted by -1)
                core_samples_mask = np.zeros_like(labels, dtype=bool)
                core_samples_mask[db.core_sample_indices_] = True
                labels = db.labels_
                # Number of clusters in labels, ignoring noise if present.
                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                # Print the number of clusters
                print('Estimated number of clusters: %d' % n_clusters)

                score_points = output[output > 0.9]

                # Calculate the mean score for each cluster and identify the cluster with the highest mean score
                best_cluster_id = None
                max_avg_score = 0
                max_cluster_size = 0

                normalize_size = np.array([])
                
                for cluster_id in np.unique(labels):
                    if cluster_id == -1:
                        continue
                    normalize_size = np.append(normalize_size, len(points[labels == cluster_id]))
                
                normalize_size = normalize_size/np.max(normalize_size)*0.025

                # Calculate the mean coordinate for each cluster
                count = 0
                for cluster_id in np.unique(labels):
                    if cluster_id == -1:
                        continue  # Skip noise

                    cluster_points = points[labels == cluster_id]
                    cluster_scores = score_points[labels == cluster_id]
                    
                    mean_coordinate = cluster_points.mean(axis=0)
                    avg_score = cluster_scores.mean()
                    # avg_cluster_size_score = cluster_scores.mean() + normalize_size[count]
                    avg_cluster_size_score = cluster_scores.mean()
                    avg_cluster_size = len(cluster_points)

                    print(f'Cluster {cluster_id}: Mean coordinate: {mean_coordinate}, Average score: {avg_cluster_size_score}, Cluster Size: {len(cluster_points)}')

                    # if avg_score > max_avg_score:
                    #     max_avg_score = avg_score
                    #     best_cluster_id = cluster_id
                    #     grasp_point = np.array([mean_coordinate[1], mean_coordinate[0]]).astype(np.int16)
                    if avg_cluster_size_score > max_cluster_size:
                        max_cluster_size = avg_cluster_size_score
                        best_cluster_id = cluster_id
                        grasp_point = np.array([mean_coordinate[1], mean_coordinate[0]]).astype(np.int16)
                    count += 1
                print(f'Best cluster is {best_cluster_id} with average score {max_cluster_size}.')
            except:
                pass
        try:
            if(grasp_point == None):
                max_coordinates = np.argwhere(output == max_value)
                print(max_coordinates)
                avg_row = 0.0
                avg_col = 0.0
                for i in range(len(max_coordinates)):
                    max_row, max_col = max_coordinates[i][1], max_coordinates[i][0]
                    avg_row += max_row
                    avg_col += max_col

                avg_row /= len(max_coordinates)
                avg_col /= len(max_coordinates)
                
                min_dist = sys.maxsize
                grasp_point = None
                for i in range(len(max_coordinates)):
                    max_row, max_col = max_coordinates[i][1], max_coordinates[i][0]
                    temp_first_point = np.array([max_row, max_col])
                    temp_second_point = np.array([avg_row, avg_col])
                    dist = np.linalg.norm(temp_second_point-temp_first_point)
                    if(dist < min_dist):
                        dist = min_dist
                        grasp_point = temp_second_point.astype(np.int16)
        except:
            pass

        # plt.Circle((grasp_point[1], grasp_point[0]), 10, fill=True)
        # plt.figure(0)
        # plt.imshow(output)
        # plt.show()

        return torch.tensor(grasp_point), centroid


# def visualize_point_cloud(point_cloud):
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     x = point_cloud[:, :, 0].flatten()
#     y = point_cloud[:, :, 1].flatten()
#     z = point_cloud[:, :, 2].flatten()
#     ax.scatter(x, y, z, s=0.5)
#     ax.set_xlabel('X')
#     ax.set_ylabel('Y')
#     ax.set_zlabel('Z')
#     plt.show()