import torch
from PIL import Image
from isaacgym import gymtorch
from homogeneous_trasnformation_and_conversion.rotation_conversions import *


class Swipe:
    def swipe_routine(
            self,
            envs, 
            actions
        ):
        actions = actions.clone().detach()[envs]
        if len(envs) > 0:
            for env_count in envs:
                env_count = env_count.item()
                action_temp = actions.clone().detach()
                # Split arm and gripper command
                u_arm_temp, u_gripper = action_temp[env_count, :6], action_temp[env_count, 6]
                u_arm_temp = u_arm_temp.to(self.device)
                u_arm_temp = u_arm_temp * self.cmd_limit / self.action_scale
                # print("u_arm_temp: ", u_arm_temp)
                # print("u arm temp shape: ", u_arm_temp.shape)
                # Get target dist and prim
                self.prim_y = action_temp[env_count, 7]
                self.prim_z = action_temp[env_count, 8]
                self.prim_target_dist_x = action_temp[env_count, 9]
                self.prim_target_dist_y = action_temp[env_count, 10]
                if(self.go_to_start[env_count]):
                    if self.init_go_to_start[env_count] and self.primitive_count[env_count] > 1:
                        _all_objects_current_pose = {}
                        _all_object_position_error = torch.tensor(0.0).to(self.device)
                        pass_obs = True
                        # collecting pose of all objects
                        for object_id in self.selected_object_env[env_count]:
                            _all_objects_current_pose[int(object_id.item())] = self._root_state[env_count, self._object_model_id[int(object_id.item())-1], :][:3].type(
                                torch.float).detach().clone()
                            _all_object_position_error += torch.sum(self.object_pose_store[env_count][int(object_id.item(
                            ))][:3] - self._root_state[env_count, self._object_model_id[int(object_id.item())-1], :][:3])
                            if (_all_objects_current_pose[int(object_id.item())][2] < torch.tensor(1.3) or _all_objects_current_pose[int(object_id.item())][2] > torch.tensor(1.7)
                                or _all_objects_current_pose[int(object_id.item())][0] < torch.tensor(0.2) or _all_objects_current_pose[int(object_id.item())][0] > torch.tensor(0.95)
                                or _all_objects_current_pose[int(object_id.item())][1] < torch.tensor(-0.18) or _all_objects_current_pose[int(object_id.item())][1] > torch.tensor(0.10)):
                                    self.env_complete_reset[env_count] = 1
                                    print("Object out of bin 2")
                                    pass_obs = False
                        if pass_obs:
                            if self.return_pre_grasp[env_count] == 0:
                                # print("pass obs 2 in env ", env_count)
                                self.finished_prim[env_count] = 1
                                self.min_distance = torch.ones(self.num_envs).to(self.device)*100
                                self.return_pre_grasp[env_count] = 1
                            else:
                                self.env_list_reset_arm_pose[env_count] = 1
                                self.return_pre_grasp[env_count] = 0
                                self.init_go_to_start[env_count] = False
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
                        [0, 0, 0]).to(self.device), "XYZ", degrees=False)
                    T_camera_to_object = transformation_matrix(
                        rotation_matrix_camera_to_object, torch.tensor([0.900, self.prim_y, self.prim_z]).to(self.device)) 
                    # Transformation from base link to object
                    T_world_to_object = torch.matmul(
                        self.T_world_to_camera_link, T_camera_to_object)
                    # Transformation for pre grasp pose (wp --> wo*op)
                    rotation_matrix_pre_grasp_pose = euler_angles_to_matrix(
                        torch.tensor([0, 0, 0]).to(self.device), "XYZ", degrees=True)
                    translation_pre_grasp_pose = torch.tensor(
                        [-0.28, 0, 0]).to(self.device)
                    T_pre_grasp_pose = transformation_matrix(
                        rotation_matrix_pre_grasp_pose, translation_pre_grasp_pose)
                    # Transformation of object with base link to pre grasp pose
                    T_world_to_pre_grasp_pose = torch.matmul(
                        T_world_to_object, T_pre_grasp_pose)
                    # Transformation for pre grasp pose (pe --> inv(we)*wp --> ew*wp)
                    rotation_matrix_ee_pose = quaternion_to_matrix(
                        self.curr_poses[env_count][self.multi_body_idx['ee_link']][3:7])
                    translation_ee_pose = self.curr_poses[env_count][self.multi_body_idx['wrist_3_link']][:3]
                    T_world_to_ee_pose = transformation_matrix(
                        rotation_matrix_ee_pose, translation_ee_pose)
                    try:
                        T_ee_pose_to_pre_grasp_pose = torch.matmul(
                            torch.inverse(T_world_to_ee_pose), T_world_to_pre_grasp_pose)
                    except torch.linalg.LinAlgError:
                        print("T_world_to_ee_pose is singular ", T_world_to_ee_pose)
                        env_complete_reset = torch.cat(
                            (env_complete_reset, torch.tensor([env_count])), axis=0)
                        continue

                    # Orientation error
                    action_orientation = matrix_to_euler_angles(
                        T_ee_pose_to_pre_grasp_pose[:3, :3], "XYZ")

                    pose_factor, ori_factor = 1., 0.3
                    self.action_env = torch.tensor([[pose_factor*T_ee_pose_to_pre_grasp_pose[0][3], pose_factor*T_ee_pose_to_pre_grasp_pose[1][3],
                                                    pose_factor *
                                                    T_ee_pose_to_pre_grasp_pose[2][3], ori_factor *
                                                    action_orientation[0],
                                                    ori_factor*action_orientation[1], ori_factor*action_orientation[2], 0, 0, 0, 0, 0]], dtype=torch.float).to(self.device)
                    if ((torch.max(torch.abs(self.action_env[0][:3]))) <= 0.005 and (torch.max(torch.abs(self.action_env[0][3:6]))) <= 0.005):
                        self.go_to_start[env_count] = False
                    
                else:
                    # go in by self.prim_target_dist_x -> swipe by self.prim_target_dist_y -> out by self.prim_target_dist_x
                    # Set primitive to opposite if target dist is negative
                    act = [0, 1, 3]
                    curr_prim = int(self.prim[env_count].item())
                    if curr_prim == 1:
                        self.true_target_dist = self.prim_target_dist_y
                        
                    else:
                        self.true_target_dist = self.prim_target_dist_x
                    if self.true_target_dist < 0:
                        act[curr_prim] = act[curr_prim] + 3
                        self.true_target_dist = abs(self.true_target_dist)
                    
                    new_target_dist = (torch.stack((torch.tensor([self.true_target_dist, 0., 0.], device=self.device),
                        torch.tensor([0., -self.true_target_dist, 0.], device=self.device),
                        torch.tensor([0., 0., self.true_target_dist], device=self.device),
                        torch.tensor([-self.true_target_dist, 0., 0.], device=self.device),
                        torch.tensor([0., self.true_target_dist, 0.], device=self.device),
                        torch.tensor([0., 0., -self.true_target_dist], device=self.device))))
                    
                    self.target_dist[env_count] = new_target_dist

                    action_str = self.action[env_count][act[curr_prim]]
                    curr_eef_pos = self.states["eef_pos"][env_count].unsqueeze(0)
                    curr_eef_quat = self.states["eef_quat"][env_count].unsqueeze(0)
                    move_dist = self.target_dist[env_count][act[curr_prim]]
                    u_arm_temp[0:6], action_str = self.primitives[env_count].move_w_ori(action_str, curr_eef_pos, curr_eef_quat, move_dist)
                    if self.true_target_dist <= 0.05:
                        self.temp_action = action_str

                    if action_str == "done":
                        action_str = self.temp_action
                        self.prim[env_count] += 1
                        # print("#############NEXT")
                        if curr_prim == 2:
                            self.prim[env_count] = 0
                            # reset arm to init pose
                            self.env_list_reset_default_arm_pose[env_count] = 1
                            self.frame_count[env_count] = 0
                            self.progress_buf[env_count] = 0

                            # print("#############RESET ARM")
                            self.primitive_count[env_count] += 1
                            if self.primitive_count[env_count] >= (self.num_primitive_actions + 1):
                            ######## False if min dist
                            # if False:
                                self.RL_flag[env_count] = 0
                                self.run_dexnet[env_count] = 1
                                self.free_envs_list[env_count] = torch.tensor(1)
                                bin_objects_current_pose = {}
                                for object_id in self.selected_object_env[env_count]:
                                    bin_objects_current_pose[int(object_id.item())] = self._root_state[env_count, self._object_model_id[int(object_id.item())-1], :][:7].type(
                                        torch.float).clone().detach()
                                self.object_pose_store[env_count] = bin_objects_current_pose
                                # print("#############FINISH PRIMITIVE AND RESET COUNT")
                            else:
                                self.run_dexnet[env_count] = 0
                                # print("#############WAIT NEXT PRIM IN SEQUENCE")
                                self.init_go_to_start[env_count] = True
                                self.go_to_start[env_count] = True
                    self.temp_action = action_str

                    self.action_env = u_arm_temp.to(self.device)
                    self.action_env = torch.cat((self.action_env,  torch.tensor([0, 0, 0, 0, 0]).unsqueeze(0).to(self.device) ), dim=1)# .repeat(self.action_env.shape[0], 1)), dim=1)      
                    # force sensor update
                    self.gym.refresh_force_sensor_tensor(self.sim)

                    _fsdata = self.gym.acquire_force_sensor_tensor(self.sim)
                    fsdata = gymtorch.wrap_tensor(_fsdata)
                    self.force_pre_physics = - \
                        fsdata[env_count][2].detach()
                    if (self.RL_flag[env_count] == torch.tensor(1)):
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