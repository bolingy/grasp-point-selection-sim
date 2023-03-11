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
        points_z = depth_img
        points_x = ((xmap - camera_info.cx)/camera_info.width)*points_z*camera_info.fx
        points_y = (-(ymap - camera_info.cy)/camera_info.height)*points_z*camera_info.fy
        X2 = np.array([points_x, points_y, points_z, np.ones(depth_img.shape)])
        print(X2.T.shape)
        X2 = np.reshape(X2, (camera_info.width*camera_info.height, 4))
        transformed_point = np.dot(X2, Vinv)
        print("trasnformed pont shape", transformed_point.shape)
        points = np.array([transformed_point[0, 2], transformed_point[0, 0], transformed_point[0, 1]])
        return points

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

    def convert_pixel_to_point_cloud(self, depth_point, rgb_point, camera_info, Vinv):
        point_z = depth_point
        point_x = ((rgb_point[0] - camera_info.cx)/camera_info.width)*point_z*camera_info.fx
        point_y = (-(rgb_point[1] - camera_info.cy)/camera_info.height)*point_z*camera_info.fy
        X2 = np.array([point_x, point_y, point_z, 1])
        print(X2.shape)
        transformed_point = X2*Vinv
        points = np.array([transformed_point[0, 2], transformed_point[0, 0], transformed_point[0, 1]])
        return points

    def calculator(self, depth_image, seg_mask, rgb_img, cam_proj, cam_vinv, bin_id):
        current_directory = os.getcwd()
        # depth_image = np.load(f'{current_directory}/../Data/depth_frame_5_0.npy')
        P = cam_proj.cpu().numpy()
        Vinv = cam_vinv.cpu().numpy()
        cam_width = 1920
        cam_height = 1080
        fx, fy = 2/P[0, 0], 2/P[1, 1]
        cx, cy = cam_width/2, cam_height/2
        s = 1000.0
        camera_info = CameraInfo(cam_width, cam_height, fx, fy, cx, cy, s)
        points = self.convert_rgb_depth_to_point_cloud(depth_image, camera_info, Vinv)
        print(points)
        seg_mask = np.asarray((seg_mask == 1), dtype=np.uint8)
        seg_mask1 = seg_mask.astype(np.uint8)
        
        plt.imshow(seg_mask*30, aspect='auto')
        plt.show()
        # applying contour to get the edges of the mask
        contours = cv2.findContours(
            seg_mask1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[0] if len(contours) == 2 else contours[1]
        big_contour = max(contours, key=cv2.contourArea)
        rotrect = cv2.minAreaRect(big_contour)
        (center), (width, height), angle = rotrect
        center3D_depth_value = depth_image[int(center[1]), int(center[0])]
        centroid_world_coordinate = self.convert_pixel_to_point_cloud(center3D_depth_value, [center[0], center[1]], camera_info, Vinv)
        
        image_crop_sizes = np.array([[417, 124, 691, 308],
                                    [691, 124, 968, 308],
                                    [968, 124, 1247, 308],
                                    [1247, 124, 1522, 308],
                                    [417, 308, 691, 461],
                                    [691, 308, 968, 461],
                                    [968, 308, 1247, 461],
                                    [1247, 308, 1522, 461],
                                    [417, 461, 691, 730],
                                    [691, 461, 968, 730],
                                    [968, 461, 1247, 730],
                                    [1247, 461, 1522, 730],
                                    [417, 730, 691, 869],
                                    [691, 730, 968, 869],
                                    [968, 730, 1247, 869],
                                    [1247, 730, 1522, 869],], dtype=int)
        image_bin = ["1H", "2H", "3H", "4H", "1G", "2G", "3G", "4G", "1F", "2F", "3F", "4F", "1E", "2E", "3E", "4E"]
        count = 0
        image_bin_index = image_bin.index(bin_id)
        image_crop_ind = image_crop_sizes[image_bin_index]
        cam_width = (image_crop_ind[2]-image_crop_ind[0])
        cam_height = (image_crop_ind[3]-image_crop_ind[1])
        fx, fy = 2/P[0, 0], 2/P[1, 1]
        cx, cy = cam_width/2, cam_height/2
        s = 1000.0
        camera_info = CameraInfo(cam_width, cam_height, fx, fy, cx, cy, s)
        depth = depth_image.copy()
        depth = depth.astype(np.float32)
        depth = depth[image_crop_ind[1]:image_crop_ind[3], image_crop_ind[0]:image_crop_ind[2]]
        seg_mask = seg_mask[image_crop_ind[1]:image_crop_ind[3], image_crop_ind[0]:image_crop_ind[2]]
        rgb_img = rgb_img[image_crop_ind[1]:image_crop_ind[3], image_crop_ind[0]:image_crop_ind[2]]
        # print(center)
        center = np.array([int(round(center[0] - image_crop_ind[0])), int(round(center[1] - image_crop_ind[1]))])
        print(cam_width, cam_height)
        plt.imshow(seg_mask*30, aspect='auto')
        plt.show()
        centroid = np.array([center[0], center[1]])
        '''
        This is the depth value of the centroid which was obtained through mask
        Here the depth image reads in reverse order i.e. the column and row are inverted
        '''
        # print(center)
        center_depth = depth[center[1], center[0]]
        '''
        Here the centroid of the object is converted to world coordinate
        '''
        object_centroid = self.convert_pixel_to_point_cloud(
            center_depth, center, camera_info, Vinv)
        print(object_centroid)
        '''
        The z coordinate of object centroid is of no use because in reprojection this depth value is not useful
        '''
        object_centroid[2] = 0.0
        '''
        Base coordinate array is the radius of suction cup and this is basically the right point of the suction cup
        '''
        base_coordinate = np.array([0.02, 0, 0], dtype=np.float32)
        suction_coordinates = [base_coordinate]
        '''
        Adding the right point of the suction cup to the object centroid as we want to obtain all the points around the object centroid
        '''
        object_base_coordinate = object_centroid + base_coordinate
        object_suction_coordinate = [object_base_coordinate]

        '''
        This loop is used to obtain all the 8 points of the suction cup by rotating the first vector
        '''
        for angle in range(45, 360, 45):
            x = base_coordinate[0]*math.cos(angle*math.pi/180) - \
                base_coordinate[1]*math.sin(angle*math.pi/180)
            y = base_coordinate[0]*math.sin(angle*math.pi/180) + \
                base_coordinate[1]*math.cos(angle*math.pi/180)
            '''
            Appending all the coordiantes in suction_cooridnates and the object_suction_coordinate is the x and y 3D cooridnate of the object suction points
            '''
            suction_coordinates = np.append(
                suction_coordinates, np.array([[x, y, 0.]]), axis=0)
            object_suction_coordinate = np.append(object_suction_coordinate, np.array(
                [[x+object_centroid[0], y+object_centroid[1], 0.]]), axis=0)

        '''
        Here we convert the depth image to point cloud and the format of the point cloud is channels, height and width
        '''
        point_cloud = self.convert_rgb_depth_to_point_cloud(depth, camera_info, Vinv)

        center_transform = self.find_nearest(point_cloud, np.array([object_centroid[0], object_centroid[1]]), camera_info)
        rgb_img = np.array(rgb_img)
        cv2.circle(rgb_img, (int(round(center_transform[0])), int(
            round(center_transform[1]))), 1, (0, 0, 255), 1)
        # print("center transform", center_transform)
        # print(depth[center_transform[1], center_transform[0]])

        suction_projections = np.empty((0, 3), float)
        for suction_points in suction_coordinates:
            suction_point_mark = self.find_nearest(point_cloud, np.array([object_centroid[0]+suction_points[0], object_centroid[1]+suction_points[1]]), camera_info)
            # print("point_cloud cooridnates", self.find_nearest(point_cloud, np.array([object_centroid[0]+suction_points[0], object_centroid[1]+suction_points[1]]), camera_info))
            '''
            here for each desired suction point coordinate which we saved in the suction_cooridantes is used to find the nearest cooridnate of the suction cup
            This object_centroid[0]+suction_point is same as cooridnates appended in the object_suction_coorindates
            '''
            suction_point = self.find_nearest(point_cloud, [
                                              object_centroid[0]+suction_points[0], object_centroid[1]+suction_points[1]], camera_info)
            cv2.circle(rgb_img, (int(round(suction_point_mark[0])), int(
                round(suction_point_mark[1]))), 1, (255, 255, 0), 1)
            # print(int(round(suction_point[0])), int(round(suction_point[1])))
            # print("desired", object_centroid[0]+suction_points[0], object_centroid[1]+suction_points[1])
            # print("actual", point_cloud[0][suction_point[1]][suction_point[0]], point_cloud[1][suction_point[1]][suction_point[0]])
            '''
            Now the suction_point is basically the actual point on the poit cloud whihc is the closest point to the desired coordinates
            Now after getting the pixel cooridnates, here the world cooridnates of each point is getting appended in the suction_projections 
            '''
            suction_projections = np.vstack((suction_projections, np.array(
                [point_cloud[0][suction_point[1]][suction_point[0]], point_cloud[1][suction_point[1]][suction_point[0]], depth[suction_point[1], suction_point[0]]])))

        
        print("without transformation", suction_projections)
        
        minimum_suction_point = np.min(suction_projections[:,2])
        ri = np.array([])
        for i in range(len(suction_projections)):
            value = min(1, np.abs(
                (suction_projections[i][2]-minimum_suction_point))/(0.023))
            ri = np.append(ri, value)

        
        # print(ri)
        score_wo_trasnform = 1-np.max(ri)
        print("Conical Springs score without transformation",score_wo_trasnform)
        # score = 10
        plt.imshow(rgb_img, aspect='auto')
        plt.show()
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
        return score_wo_trasnform

