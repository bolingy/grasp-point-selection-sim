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
    def convert_rgb_depth_to_point_cloud(self, depth_img, camera_info, Vinv):
        xmap = np.arange(camera_info.width)
        ymap = np.arange(camera_info.height)
        xmap, ymap = np.meshgrid(xmap, ymap)
        points = np.zeros((camera_info.width, camera_info.height, 3))
        for i in range(camera_info.width):
            for j in range(camera_info.height):
                u = -(i-camera_info.cx)/(camera_info.width)  # image-space coordinate
                v = (j-camera_info.cy)/(camera_info.height)  # image-space coordinate
                d = depth_img[j, i]  # depth buffer value
                X2 = [d*camera_info.fx*u, d*camera_info.fy*v, d, 1]  # deprojection vector
                p2 = X2*Vinv  # Inverse camera view to get world coordinates
                points[i][j] = [p2[0, 2], p2[0, 0], p2[0, 1]]
        # points_z = depth_img
        # points_x = ((xmap - camera_info.cx)/camera_info.width)*points_z*camera_info.fx
        # points_y = (-(ymap - camera_info.cy)/camera_info.height)*points_z*camera_info.fy
        # X2 = np.array([points_x, points_y, points_z, np.ones(depth_img.shape)])
        # # X2 = np.reshape(X2, (camera_info.width*camera_info.height, 4))
        # # transformed_point = np.dot(X2, Vinv)
        # points = X2.T@Vinv
        # points = points[:, :, :3]
        # points = points.reshape(-1,3)
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(points)
        # o3d.visualization.draw_geometries([pcd],
        #                                 zoom=0.3412,
        #                                 front=[0.4257, -0.2125, -0.8795],
        #                                 lookat=[2.6172, 2.0475, 1.532],
        #                                 up=[-0.0694, -0.9768, 0.2024])
        return points

    def find_nearest(self, array, value, camera_info):
        diff = 999.
        point_cloud_index = np.array([0, 0])
        for j in range(camera_info.height):
            for k in range(camera_info.width):
                compare = abs(array[k][j][0]-value[0]) + abs(array[k][j][1]-value[1])
                if(diff > compare):
                    diff = compare
                    point_cloud_index = np.array([k, j])
        return point_cloud_index

    def convert_pixel_to_point_cloud(self, depth_point, rgb_point, camera_info, Vinv):
        point_z = depth_point
        point_x = ((rgb_point[0] - camera_info.cx)/camera_info.width)*point_z*camera_info.fx
        point_y = (-(rgb_point[1] - camera_info.cy)/camera_info.height)*point_z*camera_info.fy
        X2 = np.array([point_x, point_y, point_z, 1])
        transformed_point = X2*Vinv
        points = np.array([transformed_point[0, 2], transformed_point[0, 0], transformed_point[0, 1]])
        return points

    def calculator(self, depth_image, seg_mask, rgb_img, cam_proj, cam_vinv, bin_id):
        print(cam_vinv)
        current_directory = os.getcwd()
        # depth_image = np.load(f'{current_directory}/../Data/depth_frame_5_0.npy')
        cam_proj = cam_proj.cpu().numpy()
        fu = 2/cam_proj[0, 0]
        fv = 2/cam_proj[1, 1]
        Vinv = cam_vinv.cpu().numpy()
        print(Vinv)
        # Ignore any points which originate from ground plane or empty space
        depth_image[seg_mask == 0] = -10001
        cam_width = 1920
        cam_height = 1080
        centerU = cam_width/2
        centerV = cam_height/2
        points = []
        for i in range(cam_width):
            for j in range(cam_height):
                if depth_image[j, i] < -10000:
                    continue
                if seg_mask[j, i] > 0:
                    u = -(i-centerU)/(cam_width)  # image-space coordinate
                    v = (j-centerV)/(cam_height)  # image-space coordinate
                    d = depth_image[j, i]  # depth buffer value
                    X2 = [d*fu*u, d*fv*v, d, 1]  # deprojection vector
                    p2 = X2*Vinv  # Inverse camera view to get world coordinates
                    points.append([p2[0, 2], p2[0, 0], p2[0, 1]])
        # score = 10
        plt.imshow(rgb_img, aspect='auto')
        plt.show()
        '''
        For visualization purpose 
        '''
        # points = points.transpose(1,2,0).reshape(-1,3)
        print(points)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        o3d.visualization.draw_geometries([pcd],
                                        zoom=0.3412,
                                        front=[0.4257, -0.2125, -0.8795],
                                        lookat=[2.6172, 2.0475, 1.532],
                                        up=[-0.0694, -0.9768, 0.2024])
        return 10

