import random
import numpy as np
import os
import torch
import yaml
import json

from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgymenvs.utils.torch_jit_utils import *

from isaacgym import gymutil
import math
import cv2
from homogeneous_trasnformation_and_conversion.rotation_conversions import *
import time
import pandas as pd

from isaacgymenvs.tasks.CommonUtils.setup_env import EnvSetup
from isaacgymenvs.tasks.CommonUtils.reset_env import EnvReset
from isaacgymenvs.tasks.CommonUtils.init_variables_configs import InitVariablesConfigs
from isaacgymenvs.tasks.CommonUtils.sample_grasp import SampleGrasp
from isaacgymenvs.tasks.CommonUtils.robot_control import RobotControl
from isaacgymenvs.tasks.CommonUtils.calculate_error import ErrorCalculator


class UR16eManipulation(InitVariablesConfigs, EnvReset, EnvSetup, SampleGrasp, RobotControl, ErrorCalculator):

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render, bin_id, data_path=None):
        InitVariablesConfigs.__init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render, bin_id, data_path)
        EnvReset.__init__(self, self.gym, self.sim, self.physics_engine)
        EnvSetup.__init__(self, self.gym, self.sim, self.physics_engine)
        self.initialize_var()
        self.setup_env()
        SampleGrasp.__init__(self)
        RobotControl.__init__(self)
        ErrorCalculator.__init__(self)

    '''
    Camera access in the pre physics step to compute the force using suction cup deformation score
    '''
    def refresh_real_time_sensors(self):
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)
        # communicate physics to graphics system
        self.gym.step_graphics(self.sim)
        # render the camera sensors
        self.gym.render_all_camera_sensors(self.sim)
        self.gym.start_access_image_tensors(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)

    '''
    Detecting oscillations while grasping due to slippage
    '''
    def detect_oscillation(self, force_list):
        try:
            force = pd.DataFrame(force_list)
            force = force.astype(float)
            force_z_average = force.rolling(window=10).mean()
            force_numpy = force_z_average.to_numpy()
            dx = np.gradient(np.squeeze(force_numpy))
            dx = dx[~np.isnan(dx)]
            if (np.min(dx) < -0.8):
                return True
            else:
                return False
        except:
            return False

    def compute_observations(self):
        self.refresh_env_tensors()
        obs = ["eef_pos", "eef_quat"]
        obs += ["q_gripper"] if self.control_type == "osc" else ["q"]
        self.obs_buf = torch.cat([self.states[ob] for ob in obs], dim=-1)
        return self.obs_buf

    def orientation_error(self, desired, current):
        '''
        input: desired orientation - quaterion, current orientation - quaterion
        output: orientation err - euler
        '''
        cc = quat_conjugate(current)
        q_r = quat_mul(desired, cc)
        return q_r[:3] * torch.sign(q_r[3]).unsqueeze(-1)    

    def get_segmask(self, env_count, camera_id):
        mask_camera_tensor = self.gym.get_camera_image_gpu_tensor(
            self.sim, self.envs[env_count], self.camera_handles[env_count][camera_id], gymapi.IMAGE_SEGMENTATION)
        torch_mask_tensor = gymtorch.wrap_tensor(mask_camera_tensor)
        return torch_mask_tensor.to(self.device)

    def get_rgb_image(self, env_count, camera_id):
        rgb_camera_tensor = self.gym.get_camera_image_gpu_tensor(
            self.sim, self.envs[env_count], self.camera_handles[env_count][camera_id], gymapi.IMAGE_COLOR)
        torch_rgb_tensor = gymtorch.wrap_tensor(rgb_camera_tensor)
        rgb_image_unflattened = torch_rgb_tensor.to(self.device)
        return torch.reshape(
            rgb_image_unflattened, (rgb_image_unflattened.shape[0], -1, 4))[..., :3]

    def get_depth_image(self, env_count, camera_id):
        depth_camera_tensor = self.gym.get_camera_image_gpu_tensor(
            self.sim, self.envs[env_count], self.camera_handles[env_count][camera_id], gymapi.IMAGE_DEPTH)
        torch_depth_tensor = gymtorch.wrap_tensor(
            depth_camera_tensor)
        return -torch_depth_tensor.to(self.device)

    def save_config_grasp_json(self, env_count, save_force_disp_config, push_suction_deform_score, unreachable):
        success = False
        oscillation = False
        penetration = False
        if (not save_force_disp_config):
            end_effector_forces = []
            object_disp_json_save = {}
            self.force_list_save[env_count] = np.array([])
        else:
            end_effector_forces = self.force_list_save[env_count].tolist()
            oscillation = self.detect_oscillation(
                end_effector_forces)
            if (push_suction_deform_score > torch.tensor(0.1) and oscillation == False):
                success = True
            if (push_suction_deform_score == torch.tensor(0)):
                penetration = True

            object_disp_json_save = self.object_disp_save[env_count].copy(
            )
            for object_id in self.selected_object_env[env_count]:
                object_disp_json_save[int(object_id.item(
                ))] = object_disp_json_save[int(object_id.item())].tolist()

        # saving the grasp point ad its properties if it was a successfull grasp
        json_save = {
            "force_array": self.force_list_save[env_count].tolist(),
            "object_disp": object_disp_json_save,
            "grasp point": self.grasp_point[env_count].tolist(),
            "grasp_angle": self.grasp_angle[env_count].tolist(),
            "dexnet_score": self.dexnet_score[env_count].item(),
            "suction_deformation_score": self.suction_deformation_score[env_count].item(),
            "oscillation": oscillation,
            "gripper_score": push_suction_deform_score.item(),
            "success": success,
            "object_id": self.object_target_id[env_count].item(),
            "penetration": penetration,
            "unreachable": unreachable
        }
        new_dir_name = str(
            env_count)+"_"+str(self.track_save[env_count].type(torch.int).item())
        save_dir_json = self.data_path + str(self.bin_id) + "/" +\
            str(env_count)+"/json_data_"+new_dir_name+"_" + \
            str(self.config_env_count[env_count].type(
                torch.int).item())+".json"
        with open(save_dir_json, 'w') as json_file:
            json.dump(json_save, json_file)
        self.track_save[env_count] = self.track_save[env_count] + \
            torch.tensor(1)

    def get_object_current_pose(self, env_count, object_id):
        object_actor_index = self._object_model_id[int(object_id.item())-1]
        return self._root_state[env_count, object_actor_index, :][:7].type(torch.float).clone().detach()

    def store_objects_current_pose(self, env_count):
        bin_objects_current_pose = {}
        for object_id in self.selected_object_env[env_count]:
            bin_objects_current_pose[int(object_id.item())] = self.get_object_current_pose(
                env_count, object_id)
        return bin_objects_current_pose

    def calculate_objects_position_error(self, env_count):
        total_position_error = torch.tensor(0.0).to(self.device)
        for object_id in self.selected_object_env[env_count]:
            curr_pose = self.get_object_current_pose(env_count, object_id)[:3]
            stored_pose = self.object_pose_store[env_count][int(
                object_id.item())][:3]
            total_position_error += torch.sum(stored_pose - curr_pose)
            if curr_pose[2] < torch.tensor(0.5):
                return total_position_error, True
        return total_position_error, False

    

    def min_area_object(self, segmask_check, segmask_bin_crop):
        areas = [(segmask_check == int(object_id.item())).sum()
                 for object_id in torch.unique(segmask_bin_crop)]
        object_mask_area = min(areas, default=torch.tensor(1000))
        return object_mask_area

    

    

    def transformation_static_dynamic(self, env_count):
        '''
        Transformation for static links
        '''
        poses_tensor = self.gym.acquire_rigid_body_state_tensor(
            self.sim)
        self.curr_poses = gymtorch.wrap_tensor(
            poses_tensor).view(self.num_envs, -1, 13)
        # Transformation of base_link from world coordiante frame (wb)
        rotation_matrix_base_link = euler_angles_to_matrix(
            torch.tensor([180, 0, 0]).to(self.device), "XYZ", degrees=True)
        translation_base_link = torch.tensor(
            [0, 0, 2.020]).to(self.device)
        self.T_base_link = transformation_matrix(
            rotation_matrix_base_link, translation_base_link)
        # Transformation for camera (wc --> wb*bc)
        rotation_matrix_camera_offset = euler_angles_to_matrix(
            torch.tensor([180, 0, 0]).to(self.device), "XYZ", degrees=True)
        T_base_link_to_camera = transformation_matrix(
            rotation_matrix_camera_offset, self.camera_base_link_translation)
        self.T_world_to_camera_link = torch.matmul(
            self.T_base_link, T_base_link_to_camera)

        '''
        Transformation for dynamic links
        '''
        # Transformation for object from camera (wo --> wc*co)
        rotation_matrix_camera_to_object = euler_angles_to_matrix(torch.tensor(
            [0, -self.grasp_angle[env_count][1], self.grasp_angle[env_count][0]]).to(self.device), "XYZ", degrees=False)
        T_camera_to_object = transformation_matrix(
            rotation_matrix_camera_to_object, self.xyz_point[env_count])
        # Transformation from base link to object
        self.T_world_to_object = torch.matmul(
            self.T_world_to_camera_link, T_camera_to_object)
        # Transformation for pre grasp pose (wp --> wo*op)
        rotation_matrix_pre_grasp_pose = euler_angles_to_matrix(
            torch.tensor([0, 0, 0]).to(self.device), "XYZ", degrees=True)
        translation_pre_grasp_pose = torch.tensor(
            [-0.25, 0, 0]).to(self.device)
        T_pre_grasp_pose = transformation_matrix(
            rotation_matrix_pre_grasp_pose, translation_pre_grasp_pose)
        # Transformation of object with base link to pre grasp pose
        self.T_world_to_pre_grasp_pose = torch.matmul(
            self.T_world_to_object, T_pre_grasp_pose)
        # Transformation for pre grasp pose (pe --> inv(we)*wp --> ew*wp)
        rotation_matrix_ee_pose = quaternion_to_matrix(
            self.curr_poses[env_count][self.multi_body_idx['ee_link']][3:7])
        translation_ee_pose = self.curr_poses[env_count][self.multi_body_idx['wrist_3_link']][:3]
        self.T_world_to_ee_pose = transformation_matrix(
            rotation_matrix_ee_pose, translation_ee_pose)
        self.T_ee_pose_to_pre_grasp_pose = torch.matmul(
            torch.inverse(self.T_world_to_ee_pose), self.T_world_to_pre_grasp_pose)
        # Orientation error
        self.action_orientation = matrix_to_euler_angles(
            self.T_ee_pose_to_pre_grasp_pose[:3, :3], "XYZ")

    def check_position_error(self, env_count, env_list_reset_arm_pose, env_list_reset_objects, env_complete_reset):
        _all_objects_current_pose = {}
        _all_object_position_error = torch.tensor(0.0).to(self.device)
        _all_object_rotation_error = torch.tensor(0.0).to(self.device)
        # collecting pose of all objects
        for object_id in self.selected_object_env[env_count]:
            _all_objects_current_pose[int(object_id.item())] = self._root_state[env_count, self._object_model_id[int(object_id.item())-1], :][:3].type(
                torch.float).detach().clone()
            _all_object_position_error += torch.sum(self.object_pose_store[env_count][int(object_id.item(
            ))][:3] - self._root_state[env_count, self._object_model_id[int(object_id.item())-1], :][:3])
            q1 = self.object_pose_store[env_count][int(
                object_id.item())][3:7]
            e1 = quaternion_to_euler_angles(
                q1, "XYZ", False)
            q2 = self._root_state[env_count, self._object_model_id[int(object_id.item())-1], :][3:7].type(
                torch.float).detach().clone()
            e2 = quaternion_to_euler_angles(q2, "XYZ", False)
            _all_object_rotation_error += torch.sum(e1-e2)

            if (_all_objects_current_pose[int(object_id.item())][2] < torch.tensor(0.5)):
                env_complete_reset = torch.cat(
                    (env_complete_reset, torch.tensor([env_count])), axis=0)
        _all_object_position_error = torch.abs(
            _all_object_position_error)
        _all_object_rotation_error = torch.abs(
            _all_object_rotation_error)
        if ((_all_object_position_error > torch.tensor(0.0055)) and (self.action_contrib[env_count] == 2)):
            env_list_reset_arm_pose = torch.cat(
                (env_list_reset_arm_pose, torch.tensor([env_count])), axis=0)
            env_list_reset_objects = torch.cat(
                (env_list_reset_objects, torch.tensor([env_count])), axis=0)

        return env_list_reset_arm_pose, env_list_reset_objects, env_complete_reset, _all_objects_current_pose

    def check_other_object_error(self, env_count, env_list_reset_arm_pose, env_list_reset_objects, env_complete_reset, _all_objects_current_pose):
        '''
        detecting contacts with other objects before contact to the target object
        '''
        _all_object_pose_error = torch.tensor(0.0).to(self.device)
        try:
            # estimating movement of other objects
            for object_id in self.selected_object_env[env_count]:
                if (object_id != self.object_target_id[env_count]):
                    _all_object_pose_error += torch.abs(torch.norm(
                        _all_objects_current_pose[int(object_id.item())][:3] - self.all_objects_last_pose[env_count][int(object_id.item())][:3]))
        except Exception as error:
            _all_object_pose_error = torch.tensor(0.0)

        # reset if object has moved even before having contact with the target object
        if (_all_object_pose_error > torch.tensor(0.0075)):
            env_list_reset_arm_pose = torch.cat(
                (env_list_reset_arm_pose, torch.tensor([env_count])), axis=0)
            env_list_reset_objects = torch.cat(
                (env_list_reset_objects, torch.tensor([env_count])), axis=0)
            print(env_count, _all_object_pose_error,
                  "reset because of object collision before contact")

            self.save_config_grasp_json(
                env_count, False, torch.tensor(0), True)

        for object_id in self.selected_object_env[env_count]:
            self.all_objects_last_pose[env_count][int(
                object_id.item())] = _all_objects_current_pose[int(object_id.item())]

        return env_list_reset_arm_pose, env_list_reset_objects, env_complete_reset

    def calculate_grasp_action(self, env_count):
        # Transformation for grasp pose (wg --> wo*og)
        rotation_matrix_grasp_pose = euler_angles_to_matrix(torch.tensor(
            [0, -self.grasp_angle[env_count][1], self.grasp_angle[env_count][0]]).to(self.device), "XYZ", degrees=False).type(torch.float)
        translation_grasp_pose = torch.tensor(
            [self.ee_vel[env_count], 0, 0]).to(self.device).type(torch.float)
        translation_grasp_pose = torch.matmul(
            rotation_matrix_grasp_pose, translation_grasp_pose)

        start_point = self.T_world_to_pre_grasp_pose[:3, 3].clone(
        ).detach()
        end_point = self.T_world_to_object[:3, 3].clone().detach()
        current_point = self.T_world_to_ee_pose[:3, 3].clone(
        ).detach()
        v = torch.tensor([end_point[0] - start_point[0], end_point[1] -
                          start_point[1], end_point[2] - start_point[2]])
        # Calculate the vector connecting p1 to p
        w = torch.tensor([current_point[0] - start_point[0], current_point[1] -
                          start_point[1], current_point[2] - start_point[2]])
        # Calculate the projection of w onto v
        t = (w[0]*v[0] + w[1]*v[1] + w[2]*v[2]) / \
            (v[0]**2 + v[1]**2 + v[2]**2)
        # Calculate the closest point on the line to p
        q = torch.tensor([start_point[0] + t*v[0], start_point[1] +
                          t*v[1], start_point[2] + t*v[2]]).to(self.device)
        # Find the distance between p and q
        distance = current_point - q
        self.action_env = torch.tensor([[self.ee_vel[env_count],
                                         translation_grasp_pose[1] -
                                         distance[1] *
                                         100 *
                                         self.ee_vel[env_count],
                                         translation_grasp_pose[2] -
                                         distance[2] *
                                         100 *
                                         self.ee_vel[env_count],
                                         self.ee_vel[env_count]*100 *
                                         self.action_orientation[0],
                                         self.ee_vel[env_count]*100 *
                                         self.action_orientation[1],
                                         self.ee_vel[env_count]*100*self.action_orientation[2], 1]], dtype=torch.float)

    def update_suction_deformation_score(self, env_count):
        if (not self.count_step_suction_score_calculator[env_count] % 10 and self.suction_deformation_score[env_count] > self.force_threshold and self.force_encounter[env_count] == 0):
            rgb_image_copy_gripper = self.get_rgb_image(
                env_count, camera_id=1)

            segmask_gripper = self.get_segmask(
                env_count, camera_id=1)

            depth_image = self.get_depth_image(
                env_count, camera_id=1)

            depth_numpy_gripper = depth_image.clone().detach()
            offset = torch.tensor(
                [self.crop_coord[2], self.crop_coord[0]])
            self.suction_deformation_score[env_count], _, _ = self.suction_score_object_gripper.calculator(
                depth_numpy_gripper, segmask_gripper, rgb_image_copy_gripper, None, self.object_target_id[env_count], offset)

            if (self.suction_deformation_score[env_count] > self.force_threshold):
                self.force_SI[env_count] = self.force_object.regression(
                    self.suction_deformation_score[env_count])

        self.count_step_suction_score_calculator[env_count] += 1

    

    def evaluate_suction_grasp_success(self, env_count, env_list_reset_arm_pose, env_list_reset_objects):
        # If arm cross the force required to grasp the object
        if (self.frame_count[env_count] > torch.tensor(self.COOLDOWN_FRAMES) and self.frame_count_contact_object[env_count] == torch.tensor(0)):
            if ((torch.max(torch.abs(self.action_env[0][:3]))) <= 0.001 and (torch.max(torch.abs(self.action_env[0][3:6]))) <= 0.001):
                self.action_contrib[env_count] -= 1

                rgb_image_copy_gripper = self.get_rgb_image(
                    env_count, camera_id=1)

                segmask_gripper = self.get_segmask(
                    env_count, camera_id=1)

                depth_image = self.get_depth_image(
                    env_count, camera_id=1)

                depth_numpy_gripper = depth_image.clone().detach()
                offset = torch.tensor(
                    [self.crop_coord[2], self.crop_coord[0]])
                self.suction_deformation_score[env_count], temp_xyz_point, temp_grasp = self.suction_score_object_gripper.calculator(
                    depth_numpy_gripper, segmask_gripper, rgb_image_copy_gripper, None, self.object_target_id[env_count], offset)

                if (self.suction_deformation_score[env_count] > self.force_threshold):
                    self.force_SI[env_count] = self.force_object.regression(
                        self.suction_deformation_score[env_count])
                else:
                    self.force_SI[env_count] = torch.tensor(
                        1000).to(self.device)
                if (self.action_contrib[env_count] == 1):
                    self.xyz_point[env_count][0] += temp_xyz_point[0]
                    self.grasp_angle[env_count] = temp_grasp

            if (self.force_pre_physics > torch.max(torch.tensor([self.force_threshold, self.force_SI[env_count]])) and self.action_contrib[env_count] == 0):
                self.force_encounter[env_count] = 1
                '''
                Gripper camera
                '''
                rgb_image_copy_gripper = self.get_rgb_image(
                    env_count, camera_id=1)

                segmask_gripper = self.get_segmask(
                    env_count, camera_id=1)

                depth_image = self.get_depth_image(
                    env_count, camera_id=1)

                depth_numpy_gripper = depth_image.clone().detach()

                offset = torch.tensor(
                    [self.crop_coord[2], self.crop_coord[0]])
                score_gripper, _, _ = self.suction_score_object_gripper.calculator(
                    depth_numpy_gripper, segmask_gripper, rgb_image_copy_gripper, None, self.object_target_id[env_count], offset)
                print(env_count, " force: ", self.force_pre_physics)
                print(env_count, " suction gripper ", score_gripper)

                self.frame_count_contact_object[env_count] = 1

                self.save_config_grasp_json(
                    env_count, True, score_gripper, False)

            # If the arm collided with the environment
            elif (self.force_pre_physics > torch.tensor(10) and self.action_contrib[env_count] == 1):
                print(env_count, " force due to collision: ",
                      self.force_pre_physics)
                env_list_reset_arm_pose = torch.cat(
                    (env_list_reset_arm_pose, torch.tensor([env_count])), axis=0)
                env_list_reset_objects = torch.cat(
                    (env_list_reset_objects, torch.tensor([env_count])), axis=0)

                self.save_config_grasp_json(
                    env_count, False, torch.tensor(0), True)

        elif (self.frame_count_contact_object[env_count] == torch.tensor(1) and self.frame_count[env_count] > torch.tensor(self.COOLDOWN_FRAMES)):
            env_list_reset_arm_pose = torch.cat(
                (env_list_reset_arm_pose, torch.tensor([env_count])), axis=0)
            env_list_reset_objects = torch.cat(
                (env_list_reset_objects, torch.tensor([env_count])), axis=0)

        return env_list_reset_arm_pose, env_list_reset_objects

    def detect_target_object_movement(self, env_count, _all_objects_current_pose):
        current_object_pose = self._root_state[env_count, self._object_model_id[self.object_target_id[env_count]-1], :][:3].type(
            torch.float).detach().clone()
        try:
            object_pose_error = torch.abs(torch.norm(
                current_object_pose - self.last_object_pose[env_count]))
        except:
            object_pose_error = torch.tensor(0)
        if (object_pose_error >= 0.0003):
            self.object_movement_enabled = 1

        segmask_gripper = self.get_segmask(env_count, camera_id=1)

        depth_numpy_gripper = self.get_depth_image(
            env_count, camera_id=1)

        # Calculate the contact existance of the suction gripper and the target object
        try:
            contact_exist = self.suction_score_object_gripper.calculate_contact(
                depth_numpy_gripper, segmask_gripper, None, None, self.object_target_id[env_count])
        except Exception as e:
            print(e)
            contact_exist = torch.tensor(0)
        # center pixel of the griper camera
        mask_point_cam = segmask_gripper[int(
            self.height_gripper/2), int(self.width_gripper/2)]
        if (mask_point_cam == self.object_target_id[env_count]):
            depth_point_cam = depth_numpy_gripper[int(
                self.height_gripper/2), int(self.width_gripper/2)]
        else:
            depth_point_cam = torch.tensor(10.)

        # If the object is moving then increase the ee_vel else go to the default value of 0.1
        # print(depth_point_cam, object_pose_error, self.action_contrib[env_count], contact_exist)
        if ((depth_point_cam < torch.tensor(0.03)) and (self.action_contrib[env_count] == torch.tensor(0)) and (object_pose_error <= torch.tensor(0.001)) and (contact_exist == torch.tensor(1))):
            self.ee_vel[env_count] += torch.tensor(0.025)
            self.ee_vel[env_count] = torch.min(
                torch.tensor(1.), self.ee_vel[env_count])
            self.cmd_limit = to_torch(
                [0.25, 0.25, 0.25, 0.75, 0.75, 0.75], device=self.device).unsqueeze(0)
        else:
            self.ee_vel[env_count] = self.DEFAULT_EE_VEL
            self.cmd_limit = to_torch(
                [0.1, 0.1, 0.1, 0.5, 0.5, 0.5], device=self.device).unsqueeze(0)

        self.last_object_pose[env_count] = current_object_pose

        for object_id in self.selected_object_env[env_count]:
            self.all_objects_last_pose[env_count][int(
                object_id.item())] = _all_objects_current_pose[int(object_id.item())]

        return contact_exist

    

    def store_force_and_displacement(self, env_count):
        # force sensor update

        _fsdata = self.gym.acquire_force_sensor_tensor(self.sim)
        fsdata = gymtorch.wrap_tensor(_fsdata)
        self.force_pre_physics = - \
            fsdata[env_count][2].detach().cpu().numpy()

        if (self.action_contrib[env_count] <= 1):
            retKey = self.force_list_save.get(env_count)
            if (retKey == None):
                self.force_list_save[env_count] = torch.tensor(
                    [self.force_pre_physics])
            else:
                force_list_env = self.force_list_save[env_count]
                force_list_env = torch.cat(
                    (force_list_env, torch.tensor([self.force_pre_physics])))
                self.force_list_save[env_count] = force_list_env

            object_disp_env = self.object_disp_save[env_count].copy()
            for object_id in self.selected_object_env[env_count]:
                object_current_pose = self._root_state[env_count, self._object_model_id[int(object_id.item())-1], :][:7].type(
                    torch.float).detach().clone()
                if (int(object_id.item()) not in object_disp_env):
                    object_disp_env[int(object_id.item())] = torch.empty(
                        (0, 7)).to(self.device)
                object_disp_env[int(object_id.item())] = torch.cat(
                    [object_disp_env[int(object_id.item())], object_current_pose.unsqueeze(0)], dim=0)
            self.object_disp_save[env_count] = object_disp_env

    def get_suction_and_object_param(self, env_count, env_list_reset_arm_pose, env_list_reset_objects, env_complete_reset):
        self.env_reset_id_env[env_count] = torch.tensor(0)
        self.action_env = torch.tensor(
            [[0, 0, 0, 0, 0, 0, 1]], dtype=torch.float)

        if ((env_count in self.grasp_angle_env) and (len(self.grasp_angle_env[env_count]) != 0)):
            self.suction_deformation_score[env_count] = self.suction_deformation_score_env[env_count][0]
            self.suction_deformation_score_env[env_count] = self.suction_deformation_score_env[env_count][1:]
            self.grasp_angle[env_count] = self.grasp_angle_env[env_count][0]
            self.grasp_angle_env[env_count] = self.grasp_angle_env[env_count][1:]
            self.xyz_point[env_count] = self.xyz_point_env[env_count][0]
            self.xyz_point_env[env_count] = self.xyz_point_env[env_count][1:]
            self.grasp_point[env_count] = self.grasp_point_env[env_count][0]
            self.grasp_point_env[env_count] = self.grasp_point_env[env_count][1:]
            self.dexnet_score[env_count] = self.dexnet_score_env[env_count][0]
            self.dexnet_score_env[env_count] = self.dexnet_score_env[env_count][1:]
            self.force_SI[env_count] = self.force_SI_env[env_count][0]
            self.force_SI_env[env_count] = self.force_SI_env[env_count][1:]
        else:
            env_complete_reset = torch.cat(
                (env_complete_reset, torch.tensor([env_count])), axis=0)
        try:
            if ((env_count in self.grasp_angle_env) and (torch.count_nonzero(self.xyz_point[env_count]) < 1)):
                # error due to illegal 3d coordinate
                print("xyz point error", self.xyz_point[env_count])
                env_list_reset_arm_pose = torch.cat(
                    (env_list_reset_arm_pose, torch.tensor([env_count])), axis=0)
                env_list_reset_objects = torch.cat(
                    (env_list_reset_objects, torch.tensor([env_count])), axis=0)

                self.save_config_grasp_json(
                    env_count, False, torch.tensor(0), True)

        except Exception as error:
            env_complete_reset = torch.cat(
                (env_complete_reset, torch.tensor([env_count])), axis=0)
            print("xyz error in env ", env_count,
                  " and the error is ", error)

        return env_list_reset_arm_pose, env_list_reset_objects, env_complete_reset

    def pre_physics_step(self, actions):
        self.refresh_real_time_sensors()

        '''
        Commands to the arm for eef control
        '''
        self.actions = torch.zeros(0, 7)
        # Variables to track environments where reset conditions have been met
        env_list_reset_objects = torch.tensor([])
        env_list_reset_arm_pose = torch.tensor([])
        env_complete_reset = torch.tensor([])
        # TODO: what this loop does
        for env_count in range(self.num_envs):
            self.cmd_limit = to_torch(
                [0.1, 0.1, 0.1, 0.5, 0.5, 0.5], device=self.device).unsqueeze(0)
            # Checking reset conditions
            env_list_reset_arm_pose, env_list_reset_objects, env_complete_reset = self.reset_until_valid(
                env_count, env_list_reset_arm_pose, env_list_reset_objects, env_complete_reset)
            # reset storage tensor
            self.action_env = torch.tensor(
                [[0, 0, 0, 0, 0, 0, 1]], dtype=torch.float)
            # Checking condition for grasp point sampling
            if ((self.env_reset_id_env[env_count] == 1) and (self.frame_count[env_count] > self.COOLDOWN_FRAMES)):
                # Sample grasp points and calculate correponding suciton deformation score and required force to grasp the object
                env_list_reset_arm_pose, env_list_reset_objects, env_complete_reset = self.get_suction_and_object_param(
                    env_count, env_list_reset_arm_pose, env_list_reset_objects, env_complete_reset)
            # Checking condition for executing a suction grasp
            elif (self.env_reset_id_env[env_count] == 0 and self.frame_count[env_count] > torch.tensor(self.COOLDOWN_FRAMES)
                  and self.free_envs_list[env_count] == torch.tensor(0)):
                self.store_force_and_displacement(env_count)
                # Calculate transformations for robot control
                self.transformation_static_dynamic(env_count)
                
                env_list_reset_arm_pose, env_list_reset_objects, env_complete_reset, _all_objects_current_pose = self.check_position_error(
                    env_count, env_list_reset_arm_pose, env_list_reset_objects, env_complete_reset)
                
                # Setting the control input for pregrasp pose
                if (self.action_contrib[env_count] >= torch.tensor(1)):
                    pose_factor, ori_factor = 1., 0.3
                    self.action_env = torch.tensor([[pose_factor*self.T_ee_pose_to_pre_grasp_pose[0][3], pose_factor*self.T_ee_pose_to_pre_grasp_pose[1][3],
                                                    pose_factor *
                                                    self.T_ee_pose_to_pre_grasp_pose[2][3], ori_factor *
                                                    self.action_orientation[0],
                                                    ori_factor*self.action_orientation[1], ori_factor*self.action_orientation[2], 1]], dtype=torch.float)
                    # Reset environment if environment is unstable
                    env_list_reset_arm_pose, env_list_reset_objects, env_complete_reset = self.check_other_object_error(
                        env_count, env_list_reset_arm_pose, env_list_reset_objects, env_complete_reset, _all_objects_current_pose)
                # Execute the grasp action
                else:
                    self.calculate_grasp_action(env_count)

                    self.update_suction_deformation_score(env_count)

                    contact_exist = self.detect_target_object_movement(
                        env_count, _all_objects_current_pose)

                    env_list_reset_arm_pose, env_list_reset_objects = self.calculate_angle_error(
                        env_count, env_list_reset_arm_pose, env_list_reset_objects)

                    self.detect_contact_non_target_object(
                        env_count, _all_objects_current_pose, contact_exist)

                env_list_reset_arm_pose, env_list_reset_objects = self.evaluate_suction_grasp_success(
                    env_count, env_list_reset_arm_pose, env_list_reset_objects)

            if (self.frame_count[env_count] <= torch.tensor(self.COOLDOWN_FRAMES)):
                self.action_env = torch.tensor(
                    [[0, 0, 0, 0, 0, 0, 1]], dtype=torch.float)

            self.actions = torch.cat([self.actions, self.action_env])
            self.frame_count[env_count] += torch.tensor(1)

        self.reset_env_conditions(
            env_list_reset_arm_pose, env_list_reset_objects, env_complete_reset)

        self.execute_control_actions()

    def post_physics_step(self):
        self.progress_buf += 1
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)

        # check if there is a timeout
        if len(env_ids) > 0:
            for env_id in env_ids:
                env_count = env_id.item()
                if (self.force_list_save[env_count] != None and len(self.force_list_save[env_count]) > 10):
                    self.save_config_grasp_json(
                        env_count, True, torch.tensor(0), False)
                else:
                    self.save_config_grasp_json(
                        env_count, False, torch.tensor(0), True)

            print(f"timeout reset for environment {env_ids}")
            pos = self.reset_pre_grasp_pose(env_ids)
            self.deploy_actions(env_ids, pos)
            self.reset_object_pose(env_ids)

        self.compute_observations()
        # Compute resets

        self.reset_buf = torch.where(
            (self.progress_buf >= self.max_episode_length - 1), torch.ones_like(self.reset_buf), self.reset_buf)
