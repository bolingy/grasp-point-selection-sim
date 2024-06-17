import torch
from isaacgym import gymtorch
from homogeneous_trasnformation_and_conversion.rotation_conversions import *


class Utils:
    def reset_until_valid(self):
            """
            Reset environments until a valid configuration is found.
            """
            cooldown_envs = torch.where(self.frame_count == self.COOLDOWN_FRAMES, 1, 0)
            obj_check_envs = torch.where(self.object_pose_check_list == 1, 1, 0)
            # if index in both cooldown_envs and obj_check_envs then reset the object
            reset_obj_envs = torch.nonzero(
                torch.logical_and(cooldown_envs, obj_check_envs)
            ).flatten()
            if len(reset_obj_envs) > 0:
                for env in reset_obj_envs:
                    env = env.item()
                    # setting the pose of the object after cool down period
                    self.object_pose_store[env] = self.store_objects_current_pose(env)
                self.env_list_reset_objects[reset_obj_envs] = torch.tensor(1)
                self.object_pose_check_list[reset_obj_envs] -= torch.tensor(1)

     
            # Spawning objects until they acquire stable pose and also doesn't falls down
            obj_check_envs = torch.where(self.object_pose_check_list == 0, 1, 0)
            free_envs = torch.where(self.free_envs_list == 1, 1, 0)
            # if index in cooldown_envs, obj_check_envs and free_envs then 
            check_stable_pose_envs = torch.nonzero(
                torch.logical_and(
                    torch.logical_and(cooldown_envs, obj_check_envs), free_envs
                )
            ).flatten()
            if len(check_stable_pose_envs) > 0:
                if not self.rendered_this_step:
                    self.render_cameras()
                    self.rendered_this_step = True
                segmask = self.check_reset_conditions(check_stable_pose_envs)

                # Check if the environment returned from reset and the frame for that enviornment is 
                # equal to the COOLDOWN_FRAMES
                not_in_complete_reset = torch.where(self.env_complete_reset[check_stable_pose_envs] == 0)[0]                
                for env_count in not_in_complete_reset:
                    # Running DexNet 3.0 after investigating the pose error of the objects for sampling 
                    # grasp points
                    env_count = env_count.item()
                    self.dexnet_sample_node(env_count, segmask)

    def store_objects_current_pose(self, env):
        bin_objects_current_pose = {}
        for object_id in self.selected_object_env[env]:
            bin_objects_current_pose[
                int(object_id.item())
            ] = self.get_object_current_pose(env, object_id)
        return bin_objects_current_pose
    
    def get_object_current_pose(self, env_count, object_id):
        """
        Retrieve the current pose of a specific object.
        """
        object_actor_index = self._object_model_id[int(object_id.item()) - 1]
        return self._root_state[env_count, object_actor_index, :][:7].type(torch.float).clone().detach()
    
    
    def calculate_objects_position_error(self, env_count):
        """
        Check objects pose error with respect to the stored pose.
        """
        total_position_error = torch.tensor(0.0).to(self.device)
        for object_id in self.selected_object_env[env_count]:
            curr_pose = self.get_object_current_pose(env_count, object_id)[:3]
            stored_pose = self.object_pose_store[env_count][int(object_id.item())][:3].to(self.device)
            total_position_error += torch.sum(stored_pose - curr_pose)
            if curr_pose[2] < torch.tensor(0.5):
                return total_position_error, True
        return total_position_error, False
    
    def transformation_static_dynamic(self, env_count):
        """
        Transformation for static links
        """
        poses_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.curr_poses = gymtorch.wrap_tensor(poses_tensor).view(self.num_envs, -1, 13)
        # Transformation of base_link from world coordiante frame (wb)
        rotation_matrix_base_link = euler_angles_to_matrix(
            torch.tensor([180, 0, 0]).to(self.device), "XYZ", degrees=True
        )
        translation_base_link = torch.tensor([0, 0, 2.020]).to(self.device)
        self.T_base_link = transformation_matrix(
            rotation_matrix_base_link, translation_base_link
        )
        # Transformation for camera (wc --> wb*bc)
        rotation_matrix_camera_offset = euler_angles_to_matrix(
            torch.tensor([180, 0, 0]).to(self.device), "XYZ", degrees=True
        )
        T_base_link_to_camera = transformation_matrix(
            rotation_matrix_camera_offset, self.camera_base_link_translation
        )
        self.T_world_to_camera_link = torch.matmul(
            self.T_base_link, T_base_link_to_camera
        )

        """
        Transformation for dynamic links
        """
        # Transformation for object from camera (wo --> wc*co)
        rotation_matrix_camera_to_object = euler_angles_to_matrix(
            self.grasp_angle[env_count].to(self.device),
            "XYZ",
            degrees=False,
        )
        T_camera_to_object = transformation_matrix(
            rotation_matrix_camera_to_object, self.xyz_point[env_count]
        )
        # Transformation from base link to object
        self.T_world_to_object = torch.matmul(
            self.T_world_to_camera_link, T_camera_to_object
        )
        # Transformation for pre grasp pose (wp --> wo*op)
        rotation_matrix_pre_grasp_pose = euler_angles_to_matrix(
            torch.tensor([0, 0, 0]).to(self.device), "XYZ", degrees=True
        )
        translation_pre_grasp_pose = torch.tensor([-0.25, 0, 0]).to(self.device)
        T_pre_grasp_pose = transformation_matrix(
            rotation_matrix_pre_grasp_pose, translation_pre_grasp_pose
        )
        # Transformation of object with base link to pre grasp pose
        self.T_world_to_pre_grasp_pose = torch.matmul(
            self.T_world_to_object, T_pre_grasp_pose
        )
        # Transformation for pre grasp pose (pe --> inv(we)*wp --> ew*wp)
        rotation_matrix_ee_pose = quaternion_to_matrix(
            self.curr_poses[env_count][self.multi_body_idx["ee_link"]][3:7]
        )
        translation_ee_pose = self.curr_poses[env_count][
            self.multi_body_idx["wrist_3_link"]
        ][:3]
        self.T_world_to_ee_pose = transformation_matrix(
            rotation_matrix_ee_pose, translation_ee_pose
        )
        self.T_ee_pose_to_pre_grasp_pose = torch.matmul(
            torch.inverse(self.T_world_to_ee_pose), self.T_world_to_pre_grasp_pose
        )
        # Orientation error
        self.action_orientation = matrix_to_euler_angles(
            self.T_ee_pose_to_pre_grasp_pose[:3, :3], "XYZ"
        )