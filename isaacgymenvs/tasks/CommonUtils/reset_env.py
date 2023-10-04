import random
import numpy as np
import torch
from homogeneous_trasnformation_and_conversion.rotation_conversions import *
from isaacgym import gymtorch
from isaacgym.torch_utils import *
from isaacgymenvs.tasks.CommonUtils.setup_env import EnvSetup

class EnvReset:
    def __init__(self, gym, sim):
        self.gym = gym
        self.sim = sim
        self.env_setup_instance = EnvSetup(gym, sim)
        self.ur16e_dof_lower_limits = self.env_setup_instance.ur16e_dof_lower_limits

    '''
    Reset the arm pose for heading towards pre grasp pose
    '''
    def random_number_with_probabilities(self, probabilities):
        random_number = random.random()
        for i, probability in enumerate(probabilities):
            if random_number < probability:
                return i
        return len(probabilities) - 1

    def reset_init_arm_pose(self, env_ids):
        for env_count in env_ids:
            env_count = env_count.item()
            # How many objects should we spawn 2 or 3
            probabilities = self.object_bin_prob_spawn[self.bin_id]
            random_number = self.random_number_with_probabilities(
                probabilities)
            # random_number = random.choice([1, 2, 3])
            random_number += 1
            object_list_env = {}
            object_set = range(1, self.object_count_unique+1)

            if (self.smaller_bin):
                selected_object = np.random.choice(
                    object_set, p=self.object_prob/np.sum(self.object_prob), size=random_number, replace=False)
            else:
                selected_object = np.random.choice(
                    object_set, p=None, size=random_number, replace=False)

            list_objects_domain_randomizer = torch.tensor([])
            for object_count in selected_object:
                domain_randomizer = random_number = random.choice(
                    [1, 2, 3, 4, 5])
                offset_object = np.array([np.random.uniform(0.67, 0.7, 1).reshape(
                    1,)[0], np.random.uniform(-0.22, -0.12, 1).reshape(1,)[0], self.object_height_spawn[self.bin_id], np.random.uniform(0.0, 6.28, 1).reshape(1,)[0],
                    np.random.uniform(0.0, 6.28, 1).reshape(1,)[0], np.random.uniform(0.0, 6.28, 1).reshape(1,)[0]])

                quat = euler_angles_to_quaternion(
                    torch.tensor(offset_object[3:6]), "XYZ", degrees=False)
                offset_object = np.concatenate(
                    [offset_object[:3], quat.cpu().numpy()])
                item_config = (object_count-1)*5 + domain_randomizer
                object_list_env[item_config] = torch.tensor(offset_object)
                list_objects_domain_randomizer = torch.cat(
                    (list_objects_domain_randomizer, torch.tensor([item_config])))
            self.selected_object_env[env_count] = list_objects_domain_randomizer
            self.object_pose_store[env_count] = object_list_env

        print("Object configuration in each bin: ",
              self.selected_object_env)
        # pos = torch.tensor(np.random.uniform(low=-6.2832, high=6.2832, size=(6,))).to(self.device).type(torch.float)
        # pos = tensor_clamp(pos.unsqueeze(0), self.ur16e_dof_lower_limits.unsqueeze(0), self.ur16e_dof_upper_limits)
        pos = tensor_clamp(self.ur16e_default_dof_pos.unsqueeze(
            0), self.ur16e_dof_lower_limits.unsqueeze(0), self.ur16e_dof_upper_limits)
        pos = pos.repeat(len(env_ids), 1)

        # reinitializing the variables
        for env_id in env_ids:
            self.action_contrib[env_id] = 2
            self.force_encounter[env_id] = 0
            self.frame_count_contact_object[env_id] = 0
            self.frame_count[env_id] = 0
            self.free_envs_list[env_id] = torch.tensor(1)
            self.object_pose_check_list[env_id] = self.RETRY_OBJECT_STABILITY
            self.ee_vel[env_id] = self.DEFAULT_EE_VEL
            self.count_step_suction_score_calculator[env_id] = 0

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0

        return pos

    '''
    Resetting object poses and quering object pose from the saved object pose
    '''

    def reset_object_pose(self, env_ids):
        for counter in range(len(self.object_models)):
            self.object_poses = torch.zeros(0, 7).to(self.device)
            for env_count in env_ids:
                # resetting force list
                # randomly spawning objects
                retKey = self.object_pose_store[env_count.item()].get(
                    counter+1)
                if retKey is None:
                    object_pose_env = torch.tensor(
                        [[counter/4, 1, 0.5, 0.0, 0.0, 0.0, 1.0]]).to(self.device)
                else:
                    object_pose_env = self.object_pose_store[env_count.item(
                    )][counter+1].clone().detach().to(self.device)
                    object_pose_env = object_pose_env.unsqueeze(0)
                self.object_poses = torch.cat(
                    [self.object_poses, object_pose_env])

            self.reset_init_object_state(
                env_ids=env_ids, object=self.object_models[counter], offset=self.object_poses)

        for env_count in env_ids:
            self.force_list_save[env_count.item()] = None
            self.target_object_disp_save[env_count.item()] = None
            self.object_disp_save[env_count.item()] = {}
            self.all_objects_last_pose[env_count.item()] = {}

        # setting the objects with randomly generated poses
        for counter in range(len(self.object_models)):
            self._object_model_state[counter][env_ids] = self._init_object_model_state[counter][env_ids]

        multi_env_ids_cubes_int32 = self._global_indices[env_ids, -len(
            self.object_models):].flatten()
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self._root_state),
            gymtorch.unwrap_tensor(multi_env_ids_cubes_int32), len(multi_env_ids_cubes_int32))

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0

        # reinitializing the variables
        for env_id in env_ids:
            self.action_contrib[env_id] = 2
            self.force_encounter[env_id] = 0
            self.frame_count_contact_object[env_id] = 0
            self.frame_count[env_id] = 0
            self.env_reset_id_env[env_id] = 1
            self.ee_vel[env_id] = 0.15
            self.force_contact_flag[env_id.item()] = torch.tensor(
                0).type(torch.bool)

        self.object_movement_enabled = 0
        self.cmd_limit = to_torch(
            [0.1, 0.1, 0.1, 0.5, 0.5, 0.5], device=self.device).unsqueeze(0)

    def reset_idx_init(self, env_ids):
        pos = self.reset_init_arm_pose(env_ids)
        self.deploy_actions(env_ids, pos)
        # Update objects states
        self.reset_object_pose(env_ids)