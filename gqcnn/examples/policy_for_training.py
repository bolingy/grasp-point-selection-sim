# -*- coding: utf-8 -*-
import argparse
import json
import os
import time

import numpy as np

from autolab_core import (YamlConfig, Logger, BinaryImage, CameraIntrinsics,
                          ColorImage, DepthImage, RgbdImage)
from visualization import Visualizer2D as vis

from gqcnn.gqcnn.grasping import (RobustGraspingPolicy,
                            CrossEntropyRobustGraspingPolicy, RgbdImageState,
                            FullyConvolutionalGraspingPolicyParallelJaw,
                            FullyConvolutionalGraspingPolicySuction)
from gqcnn.gqcnn.utils import GripperMode

from pathlib import Path
cur_path = str(Path(__file__).parent.absolute())

class dexnet3():
    def __init__(self, depth_image, segmask, rgb_img, cam_intr):
        self.depth_im = depth_image
        self.segmask = segmask
        self.rgb_im_filename = rgb_img
        self.camera_intr = cam_intr
        self.config_filename = "/home/soofiyan_ws/Documents/Issac_gym_ws/grasp-point-selection-sim/gqcnn/cfg/examples/gqcnn_suction.yaml"
        self.model_dir = "/home/soofiyan_ws/Documents/Issac_gym_ws/grasp-point-selection-sim/gqcnn/models"
        self.model_name = "GQCNN-3.0"

        self.logger = Logger.get_logger("/home/soofiyan_ws/Documents/Issac_gym_ws/grasp-point-selection-sim/gqcnn/examples/policy_for_training.py")
    
    def inference(self):
        # Get configs
        model_path = os.path.join(self.model_dir, self.model_name)
        
        # Read config.
        config = YamlConfig(self.config_filename)
        inpaint_rescale_factor = config["inpaint_rescale_factor"]
        policy_config = config["policy"]

        # Make relative paths absolute.
        if "gqcnn_model" in policy_config["metric"]:
            policy_config["metric"]["gqcnn_model"] = model_path
            if not os.path.isabs(policy_config["metric"]["gqcnn_model"]):
                policy_config["metric"]["gqcnn_model"] = os.path.join(
                    os.path.dirname(os.path.realpath(__file__)), "..",
                    policy_config["metric"]["gqcnn_model"])

        # Read images.
        color_im = ColorImage(np.zeros([self.depth_im.height, self.depth_im.width,
                                        3]).astype(np.uint8),
                            frame=self.camera_intr.frame)

        # Optionally read a segmask.
        valid_px_mask = self.depth_im.invalid_pixel_mask().inverse()
        if self.segmask is None:
            self.segmask = valid_px_mask
        else:
            self.segmask = self.segmask.mask_binary(valid_px_mask)

        # Inpaint.
        self.depth_im = self.depth_im.inpaint(rescale_factor=inpaint_rescale_factor)

        if "input_images" in policy_config["vis"] and policy_config["vis"][
                "input_images"]:
            vis.figure(size=(10, 10))
            num_plot = 1
            if self.segmask is not None:
                num_plot = 2
            vis.subplot(1, num_plot, 1)
            vis.imshow(self.depth_im)
            if self.segmask is not None:
                vis.subplot(1, num_plot, 2)
                vis.imshow(self.segmask)
            vis.show()
        
        # Create state.
        rgbd_im = RgbdImage.from_color_and_depth(color_im, self.depth_im)
        state = RgbdImageState(rgbd_im, self.camera_intr, segmask=self.segmask)

        policy_type = "ranking"
        if "type" in policy_config:
            policy_type = policy_config["type"]
        if policy_type == "ranking":
            policy = RobustGraspingPolicy(policy_config)
        elif policy_type == "cem":
            policy = CrossEntropyRobustGraspingPolicy(policy_config)
        else:
            raise ValueError("Invalid policy type: {}".format(policy_type))

        # Query policy.
        policy_start = time.time()
        # action = policy(state)

        action, grasps, q_values = policy(state)
        num_grasps = len(grasps)
        grasps_and_predictions = zip(np.arange(num_grasps), grasps, q_values)
        grasps_and_predictions = sorted(grasps_and_predictions,
                                        key=lambda x: x[2],
                                        reverse=True)
        # std_dev_np = np.array([])
        for i in range(num_grasps):
            print(f"action coordinates --> ({grasps_and_predictions[i][1].center.x}, {grasps_and_predictions[i][1].center.y}) score --> {grasps_and_predictions[i][2]}")
            # std_dev_np = np.append(std_dev_np, grasps_and_predictions[i][2])

        # print("std_dev: ", np.std(std_dev_np))
        # print("mean: ", np.mean(std_dev_np))

        self.logger.info("Planning took %.3f sec" % (time.time() - policy_start))

        # Vis final grasp.
        # policy_config["vis"]["final_grasp"] = 0
        if policy_config["vis"]["final_grasp"]:
            vis.figure(size=(10, 10))
            vis.imshow(rgbd_im.depth,
                    vmin=policy_config["vis"]["vmin"],
                    vmax=policy_config["vis"]["vmax"])
            vis.grasp(action.grasp, scale=2.5, show_center=False, show_axis=True)
            vis.title("Planned grasp at depth {0:.3f}m with Q={1:.3f}".format(
                action.grasp.depth, action.q_value))
            vis.show()
        
        return action, grasps_and_predictions

import cv2
import random
import imutils
if __name__ == "__main__":
    depth_im_filename = np.load("/home/soofiyan_ws/Documents/Issac_gym_ws/grasp-point-selection-sim/gqcnn/data/examples/single_object/primesense/depth_2.npy")
    segmask = cv2.imread("/home/soofiyan_ws/Documents/Issac_gym_ws/grasp-point-selection-sim/gqcnn/data/examples/single_object/primesense/segmask_2.png", cv2.IMREAD_GRAYSCALE)
    # camera_intr_filename = "/home/soofiyan_ws/catkin_ws/src/gqcnn/data/calib/primesense/primesense.intr"
    segmentation = np.where(segmask == 255)
    bbox = 0, 0, 0, 0
    if len(segmentation) != 0 and len(segmentation[1]) != 0 and len(segmentation[0]) != 0:
        x_min = int(np.min(segmentation[1]))
        x_max = int(np.max(segmentation[1]))
        y_min = int(np.min(segmentation[0]))
        y_max = int(np.max(segmentation[0]))

        bbox = x_min, x_max, y_min, y_max

    segmask = cv2.rectangle(segmask, (x_min, y_min), (x_max, y_max), (255, 0, 0), 1)
    
    print(bbox)

    depth_left = depth_im_filename[y_min, x_min]
    depth_right = depth_im_filename[y_max, x_max]
    print(depth_left, depth_right)
    x_world_left = (x_min-319.5)/525.0 * depth_left
    x_world_right = (x_max-319.5)/525.0 * depth_right
    y_world_left = (y_min-239.5)/525.0 * depth_left
    y_world_right = (y_max-239.5)/525.0 * depth_right

    division_x = ((x_world_right-x_world_left)/0.005)[0]
    division_y = ((y_world_right-y_world_left)/0.005)[0]

    range_x = x_max - x_min
    range_y = y_max - y_min
    prev_x_coordinate = x_min
    prev_y_coordinate = y_min
    print(range_x/division_x)
    for i in range(x_min, x_max, round(range_x/division_x)):
        for j in range(y_min, y_max, round(range_y/division_y)):
            x = random.randint(prev_x_coordinate, prev_x_coordinate+round(range_x/division_x))
            y = random.randint(prev_y_coordinate, prev_y_coordinate+round(range_y/division_y))
            # print(x, y, segmask[y,x], depth_im_filename[y,x])
            if(segmask[y,x]):
                cv2.circle(segmask, (x,y), radius=1, color=(0, 0, 0), thickness=-1)
            prev_x_coordinate = i
            prev_y_coordinate = j

    cv2.imshow("segmask", segmask)
    cv2.waitKey(0)

    # camera_intrinsics = CameraIntrinsics(frame="camera", fx=525.0, fy=525.0, cx=319.5, cy=239.5, skew=0.0, height=480, width=640)
    
    # segmask_dexnet = segmask
    # segmask_numpy = segmask_dexnet.astype(np.uint8)
    # segmask_dexnet = BinaryImage(segmask_numpy, frame=camera_intrinsics.frame)
    
    # depth_image_dexnet = depth_im_filename
    # depth_numpy = depth_image_dexnet
    # depth_img_dexnet = DepthImage(depth_numpy, frame=camera_intrinsics.frame)

    # dexnet_object = dexnet3(depth_img_dexnet, segmask_dexnet, None, camera_intrinsics)
    # action = dexnet_object.inference()
    # print(action.q_value)
    # print(action.grasp.center.x, action.grasp.center.y)
