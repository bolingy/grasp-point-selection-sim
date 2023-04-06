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
        self.config_filename = cur_path+"/../cfg/examples/gqcnn_suction.yaml"
        self.model_dir = cur_path+"/../models"
        self.model_name = "GQCNN-3.0"

        self.logger = Logger.get_logger("../examples/policy_for_training.py")
    
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
        for i in range(num_grasps):
            print(f"action coordinates --> ({grasps_and_predictions[i][1].center.x}, {grasps_and_predictions[i][1].center.y}) score --> {grasps_and_predictions[i][2]}")
        
        self.logger.info("Planning took %.3f sec" % (time.time() - policy_start))

        # Vis final grasp.
        policy_config["vis"]["final_grasp"] = 1
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

# if __name__ == "__main__":
#     depth_im_filename = "/home/soofiyan_ws/catkin_ws/src/gqcnn/data/examples/clutter/primesense/depth_0.npy"
#     segmask_filename = "/home/soofiyan_ws/catkin_ws/src/gqcnn/data/examples/clutter/primesense/segmask_0.png"
#     camera_intr_filename = "/home/soofiyan_ws/catkin_ws/src/gqcnn/data/calib/primesense/primesense.intr"
#     object = dexnet3(depth_im_filename, segmask_filename, None, camera_intr_filename)
#     action = object.inference()
#     print(action.q_value)
#     print(action.grasp.center.x, action.grasp.center.y)
