import math
import numpy as np
import cv2
from matplotlib import pyplot as plt, patches
import open3d as o3d
import os
import torch
from autolab_core import (YamlConfig, Logger, BinaryImage, CameraIntrinsics, ColorImage, DepthImage, RgbdImage)
from homogeneous_trasnformation_and_conversion.rotation_conversions import *

class calcualte_suction_score():
    def __init__(self, depth_image, segmask, rgb_img, camera_intrinsics, grasps_and_predictions, object_id):
        self.depth_image = depth_image
        self.segmask = segmask
        self.rgb_img = rgb_img
        self.camera_intrinsics = camera_intrinsics
        self.grasps_and_predictions = grasps_and_predictions
        self.object_id = object_id
        self.device = 'cuda:0'

        '''
        Calculate the normals of the object
        '''
        self.depth_image_normal = depth_image.clone().cpu().numpy()
        depth_normal = DepthImage(self.depth_image_normal, frame=camera_intrinsics.frame)
        point_cloud_im = camera_intrinsics.deproject_to_image(depth_normal)
        self.normal_cloud_im = point_cloud_im.normal_cloud_im()

        # pc_o3d = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(point_cloud_im.data.reshape(-1,3)))
        # # print('here')
        # pc_o3d.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(100), fast_normal_computation=False)
        # pc_o3d.orient_normals_to_align_with_direction(np.array([0., 0., -1.]))
        # pc_o3d.normalize_normals()
        # self.normals = np.array(pc_o3d.normals).astype(np.float32)

    def convert_rgb_depth_to_point_cloud(self):
        camera_u = torch.arange(0, self.camera_intrinsics.width, device=self.device)
        camera_v = torch.arange(0, self.camera_intrinsics.height, device=self.device)
        camera_v, camera_u = torch.meshgrid(camera_v, camera_u, indexing='ij')

        Z = self.depth_image
        X = (camera_u-self.camera_intrinsics.cx)/self.camera_intrinsics.fx * Z
        Y = (camera_v-self.camera_intrinsics.cy)/self.camera_intrinsics.fy * Z

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
        X = (u-self.camera_intrinsics.cx)/self.camera_intrinsics.fx * Z
        Y = (v-self.camera_intrinsics.cy)/self.camera_intrinsics.fy * Z

        depth_bar = 10
        Z = Z.view(-1)  
        valid = Z > -depth_bar
        X = X.view(-1)
        Y = Y.view(-1)

        position = torch.vstack((X, Y, Z, torch.ones(len(X), device=self.device)))[:, valid]
        position = position.permute(1, 0)
        # position = position@vinv
        points = position[:, 0:3]
        if(len(points) == 0):
            return torch.tensor([0, 0, -1])
        return points[0]
    
    def convert_xyz_point_to_uv_point(self, xyz_point):
        X = xyz_point[:, 0]
        Y = xyz_point[:, 1]
        Z = xyz_point[:, 2]
        u = X*self.camera_intrinsics.fx/Z + self.camera_intrinsics.cx
        v = Y*self.camera_intrinsics.fy/Z + self.camera_intrinsics.cy
        
        return u, v

    def find_nearest(self, centroid, points):
        suction_points = centroid[:2] + self.suction_coordinates[:,:2]
        distances = torch.cdist(points[:,:2].type(torch.float64), suction_points.type(torch.float64))
        min_indices = torch.argmin(distances, dim=0)
        point_cloud_suction = points[min_indices]
        return point_cloud_suction


    def calculator(self):
        self.segmask = (self.segmask == self.object_id)
        self.depth_image[self.segmask == 0] = -10001
        centroid_angle = torch.tensor([0, 0, 0]).to(self.device)

        '''
        Centroid method using median of point cloud
        '''
        points = self.convert_rgb_depth_to_point_cloud()
        centroid_point = torch.FloatTensor([torch.median(points[:, 0]), torch.median(points[:, 1]), torch.median(points[:, 2])]).to(self.device)
        if(centroid_point.any() == float('nan')):
            return 0, torch.tensor([0, 0, 0]), torch.tensor([centroid_angle[0], centroid_angle[1], centroid_angle[2]])
        '''
        Given sample point convert to xyz point
        '''
        if(self.grasps_and_predictions == None):
            xyz_point = self.convert_uv_point_to_xyz_point(int(self.segmask.shape[1]/2), int(self.segmask.shape[0]/2))
        else:
            xyz_point = self.convert_uv_point_to_xyz_point(self.grasps_and_predictions[0][0].center.x, self.grasps_and_predictions[0][0].center.y)
        if(xyz_point[2] < 0):
            return 0, torch.tensor([0, 0, 0]), torch.tensor([centroid_angle[0], centroid_angle[1], centroid_angle[2]])
        
        '''
        Debugging print statements one for the width and height and other one is the centroid
        '''
        # print(torch.max(points[:,0]) - torch.min(points[:,0]), torch.max(points[:,1]) - torch.min(points[:,1]), torch.max(points[:,2]) - torch.min(points[:,2]))
        # print("centroid", centroid)

        '''
        Store the base coordiantes of the suction cup points
        '''
        if(self.grasps_and_predictions != None):
            centroid_angle = torch.tensor(self.normal_cloud_im[int(self.grasps_and_predictions[0][0].center.y)][int(self.grasps_and_predictions[0][0].center.x)]).to(self.device)
            centroid_angle[2] = 0

            base_coordinate = torch.tensor([0.02, 0, 0]).to(self.device)
            self.suction_coordinates = base_coordinate.view(1, 3)
            # |x --y
            for angle in range(45, 360, 45):
                x = base_coordinate[0]*math.cos(angle*math.pi/180) - \
                    base_coordinate[1]*math.sin(angle*math.pi/180)
                y = base_coordinate[0]*math.sin(angle*math.pi/180) + \
                    base_coordinate[1]*math.cos(angle*math.pi/180)
                '''
                Appending all the coordiantes in suction_cooridnates and the object_suction_coordinate is the x and y 3D cooridnate of the object suction points
                '''
                self.suction_coordinates = torch.cat((self.suction_coordinates, torch.tensor([[x, y, 0.]]).to(self.device)), dim=0).type(torch.float64)
        
            rotation_matrix_normal = euler_angles_to_matrix(torch.tensor([0, 0, -90]).to(self.device), "XYZ", degrees=True)
            null_translation = torch.tensor([0, 0, 0]).to(self.device)
            T_normal = transformation_matrix(rotation_matrix_normal, null_translation)
            rotation_matrix_suction = euler_angles_to_matrix(centroid_angle.clone().detach().to(self.device), "XYZ", degrees=False)
            
            null_translation = torch.tensor([0, 0, 0]).to(self.device)
            T_suction = transformation_matrix(rotation_matrix_suction, null_translation)
            '''
            The trasnformation of the suction is inverse thus trasnpose is applied (ws --> wn*(sn)^-1)
            '''
            T_suction_to_normal = torch.matmul(T_normal, T_suction.T)
            centroid_angle_suction = matrix_to_euler_angles(T_suction_to_normal[:3, :3], "XYZ")
            centroid_angle_suction[2] = 0.0
            self.suction_coordinates = torch.mm(self.suction_coordinates, euler_angles_to_matrix(centroid_angle_suction.clone().detach().to(self.device), "XYZ", degrees=False).type(torch.float64), out=None)
        
        '''
        Finding the suction points accordint to the base coordiante
        '''
        point_cloud_suction = self.find_nearest(xyz_point, points)
        # u, v = self.convert_xyz_point_to_uv_point(point_cloud_suction)
        
        thresh = torch.sum(torch.sum(point_cloud_suction[:,:2] - (self.suction_coordinates[:, :2] + xyz_point[:2]), 1))
        if(abs(thresh) > 0.005):
            return 0, torch.tensor([0, 0, 0]), torch.tensor([centroid_angle[0], centroid_angle[1], centroid_angle[2]])

        '''
        Calcualte the conical spring score
        '''
        point_cloud_suction[:,2] = point_cloud_suction[:, 2]*torch.cos(centroid_angle[0])*torch.cos(centroid_angle[1]) - self.suction_coordinates[:,2]

        minimum_suction_point = torch.min(point_cloud_suction[:, 2]).to(self.device)
        ri = torch.clamp(torch.abs(point_cloud_suction[:, 2] - minimum_suction_point) / 0.023, max=1.0)
        suction_score = 1-torch.max(ri)

        return suction_score, torch.tensor([xyz_point[2], -xyz_point[0], -xyz_point[1]]), torch.tensor([centroid_angle[0], centroid_angle[1], centroid_angle[2]])

