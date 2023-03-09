import math
import numpy as np
import cv2
from matplotlib import pyplot as plt, patches
import open3d as o3d
import os

class CameraInfo():
    def __init__(self, width, height, fx, fy, cx, cy, scale):
        self.width = width
        self.height = height
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.scale = scale

class calcualte_suction_score():
    def convert_rgb_depth_to_point_cloud(self, depth_img, camera_info):
        xmap = np.arange(camera_info.width)
        ymap = np.arange(camera_info.height)
        xmap, ymap = np.meshgrid(xmap, ymap)
        points_z = depth_img
        points_x = -(xmap - camera_info.cx)/camera_info.width*points_z*camera_info.fx
        points_y = (ymap - camera_info.cy)/camera_info.height*points_z*camera_info.fy
        return np.array([points_x, points_y, points_z])

    def find_nearest(self, array, value, camera_info):
        diff = 999.
        point_cloud_index = np.array([0, 0])
        for j in range(camera_info.height):
            for k in range(camera_info.width):
                compare = abs(array[0][j][k]-value[0]) + abs(array[1][j][k]-value[1])
                if(diff > compare):
                    diff = compare
                    point_cloud_index = np.array([k, j])
        return point_cloud_index

    def convert_pixel_to_point_cloud(self, depth_point, rgb_point, camera_info):
        point_z = depth_point
        point_x = (rgb_point[0] - camera_info.cx) * point_z / camera_info.fx
        point_y = (rgb_point[1] - camera_info.cy) * point_z / camera_info.fy
        return np.array([point_x, point_y, point_z])

    def calculator(self, depth_image):
        current_directory = os.getcwd()
        # depth_image = np.load(f'{current_directory}/../Data/depth_frame_5_0.npy')
        P = np.load(f'{current_directory}/camera_properties/projection_matrix.npy')

        cam_width = 1920
        cam_height = 1080
        fx, fy = 2/P[0, 0], 2/P[1, 1]
        cx, cy = cam_width/2, cam_height/2
        s = 1000.0
        camera_info = CameraInfo(cam_width, cam_height, fx, fy, cx, cy, s)
        points = self.convert_rgb_depth_to_point_cloud(depth_image, camera_info)
        score = 10
        '''
        For visualization purpose 
        '''
        # points = points.transpose(1,2,0).reshape(-1,3)
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(points)
        # o3d.visualization.draw_geometries([pcd],
        #                                 zoom=0.3412,
        #                                 front=[0.4257, -0.2125, -0.8795],
        #                                 lookat=[2.6172, 2.0475, 1.532],
        #                                 up=[-0.0694, -0.9768, 0.2024])
        return score

