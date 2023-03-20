import math
import numpy as np
import cv2
from matplotlib import pyplot as plt, patches
import open3d as o3d
import os
import torch

class CameraInfo():
    def __init__(self, width, height, fx, fy, cx, cy):
        self.width = width
        self.height = height
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy

class calcualte_suction_score():
    def __init__(self, depth_image, segmask, rgb_img, camera_info, grasps_and_predictions, object_id):
        self.depth_image = depth_image
        self.segmask = segmask
        self.rgb_img = rgb_img
        self.camera_info = camera_info
        self.grasps_and_predictions = grasps_and_predictions
        self.object_id = object_id
        self.device = 'cuda:0'
        '''
        Store the base coordiantes of the suction cup points
        '''
        base_coordinate = torch.tensor([0.02, 0, 0]).to(self.device)
        self.suction_coordinates = base_coordinate.view(1, 3)
        for angle in range(45, 360, 45):
            x = base_coordinate[0]*math.cos(angle*math.pi/180) - \
                base_coordinate[1]*math.sin(angle*math.pi/180)
            y = base_coordinate[0]*math.sin(angle*math.pi/180) + \
                base_coordinate[1]*math.cos(angle*math.pi/180)
            '''
            Appending all the coordiantes in suction_cooridnates and the object_suction_coordinate is the x and y 3D cooridnate of the object suction points
            '''
            self.suction_coordinates = torch.cat((self.suction_coordinates, torch.tensor([[x, y, 0.]]).to(self.device)), dim=0)

    def convert_rgb_depth_to_point_cloud(self):
        camera_u = torch.arange(0, self.camera_info.width, device=self.device)
        camera_v = torch.arange(0, self.camera_info.height, device=self.device)
        camera_v, camera_u = torch.meshgrid(camera_v, camera_u, indexing='ij')

        Z = self.depth_image
        X = (camera_u-self.camera_info.cx)/self.camera_info.fx * Z
        Y = (camera_v-self.camera_info.cy)/self.camera_info.fy * Z

        depth_bar = 10
        Z = Z.view(-1)  
        valid = Z > -depth_bar
        X = X.view(-1)
        Y = Y.view(-1)

        position = torch.vstack((X, Y, Z, torch.ones(len(X), device=self.device)))[:, valid]
        position = position.permute(1, 0)
        # position = position@vinv
        points = position[:, 0:3]
        return points
    
    def convert_uv_point_to_xyz_point(self, u, v):
        Z = self.depth_image[v, u]
        X = (u-self.camera_info.cx)/self.camera_info.fx * Z
        Y = (v-self.camera_info.cy)/self.camera_info.fy * Z

        depth_bar = 10
        Z = Z.view(-1)  
        valid = Z > -depth_bar
        X = X.view(-1)
        Y = Y.view(-1)

        position = torch.vstack((X, Y, Z, torch.ones(len(X), device=self.device)))[:, valid]
        position = position.permute(1, 0)
        # position = position@vinv
        points = position[:, 0:3]
        return points[0]

    def find_nearest(self, centroid, points):
        suction_points = centroid[:2] + self.suction_coordinates[:,:2]
        distances = torch.cdist(points[:,:2], suction_points)
        min_indices = torch.argmin(distances, dim=0)
        point_cloud_suction = points[min_indices]
        return point_cloud_suction


    def calculator(self):
        self.segmask = (self.segmask == self.object_id)
        self.depth_image[self.segmask == 0] = -10001

        noise_image = torch.normal(0, 0.0009, size=self.depth_image.size()).to(self.device)
        # For calculating the suction deformation score no noise is used
        # self.depth_image = self.depth_image + noise_image

        '''
        Centroid method using median of point cloud
        '''
        points = self.convert_rgb_depth_to_point_cloud()
        centroid_point = torch.FloatTensor([torch.median(points[:, 0]), torch.median(points[:, 1]), torch.median(points[:, 2])]).to(self.device)

        '''
        Given sample point convert to xyz point
        '''
        xyz_point = self.convert_uv_point_to_xyz_point(self.grasps_and_predictions[0][1].center.x, self.grasps_and_predictions[0][1].center.y)
        print("centroid", xyz_point)
        '''
        Debugging print statements one for the width and height and other one is the centroid
        '''
        # print(torch.max(points[:,0]) - torch.min(points[:,0]), torch.max(points[:,1]) - torch.min(points[:,1]), torch.max(points[:,2]) - torch.min(points[:,2]))
        # print("centroid", centroid)

        '''
        Finding the suction points accordint to the base coordiante
        '''
        point_cloud_suction = self.find_nearest(centroid_point, points)
        print(point_cloud_suction)
        
        '''
        Calcualte the conical spring score
        '''
        minimum_suction_point = torch.min(point_cloud_suction[:,2]).to(self.device)
        ri = torch.clamp(torch.abs(point_cloud_suction[:,2] - minimum_suction_point) / 0.023, max=1.0)
        print(ri)
        suction_score = 1-torch.max(ri)

        return suction_score

