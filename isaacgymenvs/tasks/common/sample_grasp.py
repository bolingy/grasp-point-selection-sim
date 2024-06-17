import os
import random
import numpy as np
import torch
from autolab_core import BinaryImage, DepthImage
from PIL import Image

class SampleGrasp:
    def dexnet_sample_node(
            self,
            env_count,
            segmask
        ):
            """
            Running DexNet 3.0 and extracting sample points after investigating the pose error after spawning
            """
            if (self.RL_flag[env_count] == torch.tensor(1)):
                random_object_select = random.sample(
                    self.selected_object_env[env_count].tolist(), 1
                )
                self.object_target_id[env_count] = torch.tensor(11)
                self.env_list_reset_arm_pose[env_count] = torch.tensor(1)
                self.env_list_reset_objects[env_count] = torch.tensor(1)
            else:
                rgb_image = self.torch_rgb_cameras[env_count]
                depth_image = -self.torch_depth_cameras[env_count]
                segmask_dexnet = segmask.clone().detach()

                segmask_numpy = np.zeros_like(segmask_dexnet.cpu().numpy().astype(np.uint8))
                segmask_numpy_temp = np.zeros_like(
                    segmask_dexnet.cpu().numpy().astype(np.uint8)
                )
                segmask_numpy_temp[
                    segmask_dexnet.cpu().numpy().astype(np.uint8)
                    == self.object_target_id[env_count].cpu().numpy()
                ] = 1
                segmask_numpy[
                    segmask_dexnet.cpu().numpy().astype(np.uint8)
                    == self.object_target_id[env_count].cpu().numpy()
                ] = 255
                segmask_dexnet = BinaryImage(
                    segmask_numpy,
                    frame=self.camera_intrinsics_back_cam.frame,
                )
                depth_image_dexnet = depth_image.clone().detach()
                noise_image = torch.normal(0, 0.0005, size=depth_image_dexnet.size()).to(
                    self.device
                )
                depth_image_dexnet = depth_image_dexnet + noise_image
                depth_image_save_temp = depth_image_dexnet.clone().detach().cpu().numpy()
                self.depth_image_save[env_count] = depth_image_save_temp

                # cropping the image and modifying depth to match the DexNet 3.0 input configuration
                # Adjusting depth for DexNet: adding dexnet_thresh_offset to ensure depth is between 0.5 and 0.7 m
                dexnet_thresh_offset = 0.2
                depth_image_dexnet += dexnet_thresh_offset
                pod_back_panel_distance = torch.max(
                    depth_image
                ).item()
                # To make the depth of the image ranging from 0.5m to 0.7m for valid configuration for DexNet 3.0
                depth_numpy = depth_image_dexnet.cpu().numpy()
                depth_numpy_temp = depth_numpy * segmask_numpy_temp
                depth_numpy_temp[depth_numpy_temp == 0] = (
                    pod_back_panel_distance + dexnet_thresh_offset
                )
                depth_img_dexnet = DepthImage(
                    depth_numpy_temp,
                    frame=self.camera_intrinsics_back_cam.frame,
                )
                max_num_grasps = 0

                # Storing all the sampled grasp point and its properties from DexNet 3.0 in their respective buffers
                try:
                    y, x = np.where(segmask_numpy == 255)
                    center_x, center_y = int(np.mean(x)), int(np.mean(y))
                    self.suction_deformation_score_temp = torch.Tensor()
                    self.xyz_point_temp = torch.empty((0, 3))
                    self.grasp_angle_temp = torch.empty((0, 3))
                    self.grasp_point_temp = torch.empty((0, 2)).to(self.device)
                    self.force_SI_temp = torch.Tensor()
                    max_num_grasps = 1
                    top_grasps = max_num_grasps if max_num_grasps <= 10 else 7
                    for i in range(max_num_grasps):
                        grasp_point = torch.tensor(
                            [center_x, center_y])

                        depth_image_suction = depth_image
                        suction_deformation_score, xyz_point, grasp_angle = self.suction_score_object.calculator(
                            depth_image_suction, segmask, rgb_image, grasp_point, self.object_target_id[env_count])
                        self.suction_deformation_score_temp = torch.cat(
                            (self.suction_deformation_score_temp, torch.tensor([suction_deformation_score]))).type(torch.float)
                        self.xyz_point_temp = torch.cat(
                            [self.xyz_point_temp, xyz_point.unsqueeze(0)], dim=0)
                        self.grasp_angle_temp = torch.cat(
                            [self.grasp_angle_temp, grasp_angle.unsqueeze(0)], dim=0)
                        self.grasp_point_temp = torch.cat(
                            [self.grasp_point_temp, grasp_point.clone().detach().unsqueeze(0).to(self.device)], dim=0)
                        # cv2.circle(self.rgb_save[env_count], (grasp_point.clone().detach().cpu().numpy()[0], grasp_point.clone().detach().cpu().numpy()[1]), 2, (0, 0, 0), 1)
                        # cv2.imshow("rgb_image_vis", self.rgb_save[env_count])
                        # cv2.waitKey(1)
                        # cv2.destroyAllWindows()
                        self.object_coordiante_camera = xyz_point.clone().detach()
                        if (suction_deformation_score > 0):
                            force_SI = self.force_object.regression(
                                suction_deformation_score)
                        else:
                            force_SI = torch.tensor(0).to(self.device)

                        self.force_SI_temp = torch.cat(
                            (self.force_SI_temp, torch.tensor([force_SI])))

                    if top_grasps > 0:
                        self.env_list_reset_arm_pose[env_count] = torch.tensor(1)
                        self.env_list_reset_objects[env_count] = torch.tensor(1)
                    else:
                        print("No sample points")
                        self.env_complete_reset[env_count] = 1
                except Exception as e:
                    print("dexnet error: ", e)
                    self.env_complete_reset[env_count] = 1

            self.suction_deformation_score_env[
                env_count
            ] = self.suction_deformation_score_temp
            self.grasp_angle_env[env_count] = self.grasp_angle_temp
            self.force_SI_env[env_count] = self.force_SI_temp
            self.xyz_point_env[env_count] = self.xyz_point_temp
            self.grasp_point_env[env_count] = self.grasp_point_temp
            self.free_envs_list[env_count] = torch.tensor(0)
            if self.RL_flag[env_count] == torch.tensor(1):
                # print("pass obs 1 in env ", env_count)
                self.finished_prim[env_count] = 1
                self.done[env_count] = 1
                
    def get_suction_and_object_params(self):
        """
        Get the suction and object properties for the environment.
        """
        cooldown_envs = torch.where(self.frame_count > self.COOLDOWN_FRAMES, 1, 0)
        reset_envs = torch.where(self.env_reset_id_env == 1, 1, 0)
        RL_envs = torch.where(self.RL_flag == 1, 1, 0)
        envs = torch.nonzero(
            torch.logical_and(cooldown_envs, torch.logical_and(reset_envs, RL_envs))
        ).flatten()
        if len(envs) > 0:
            for env_count in envs:
                env_count = env_count.item()
                self.env_reset_id_env[env_count] = torch.tensor(0)
        not_RL_envs = torch.where(self.RL_flag == 0, 1, 0)
        envs = torch.nonzero(
            torch.logical_and(cooldown_envs, torch.logical_and(reset_envs, not_RL_envs))
        ).flatten()
        if len(envs) > 0:
            for env_count in envs:
                env_count = env_count.item()
                self.action_env = torch.tensor([[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]], dtype=torch.float)
                self.env_reset_id_env[env_count] = torch.tensor(0)
                if ((env_count in self.grasp_angle_env) and (len(self.grasp_angle_env[env_count]) != 0)):
                    self.suction_deformation_score[env_count] = self.suction_deformation_score_env[env_count][0]
                    self.suction_deformation_score_env[env_count] = self.suction_deformation_score_env[env_count][1:]
                    self.grasp_angle[env_count] = self.grasp_angle_env[env_count][0]
                    self.grasp_angle_env[env_count] = self.grasp_angle_env[env_count][1:]
                    self.xyz_point[env_count] = self.xyz_point_env[env_count][0]
                    self.xyz_point_env[env_count] = self.xyz_point_env[env_count][1:]
                    self.grasp_point[env_count] = self.grasp_point_env[env_count][0]
                    self.grasp_point_env[env_count] = self.grasp_point_env[env_count][1:]
                    self.force_SI[env_count] = self.force_SI_env[env_count][0]
                    self.force_SI_env[env_count] = self.force_SI_env[env_count][1:]
                else:
                    print("Resetting environment", env_count, " due to no sample points")
                    self.env_complete_reset[env_count] = torch.tensor(1)
                try:
                    if (env_count in self.grasp_angle_env) and (
                        torch.count_nonzero(self.xyz_point[env_count]) < 1
                    ):
                        # error due to illegal 3d coordinate
                        print("xyz point error", self.xyz_point[env_count])
                        self.env_list_reset_arm_pose[env_count] = torch.tensor(1)
                        self.env_list_reset_objects[env_count] = torch.tensor(1)

                        self.save_config_grasp_json(env_count, False, torch.tensor(0), True)

                except Exception as error:
                    self.env_complete_reset[env_count] = torch.tensor(1)
                    print("xyz error in env ", env_count, " and the error is ", error)

