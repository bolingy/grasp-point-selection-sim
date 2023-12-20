from copy import deepcopy
import random
import numpy as np
import os
import torch
import yaml
import json

from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym.torch_utils import *
from isaacgymenvs.utils.torch_jit_utils import *
from isaacgymenvs.tasks.base.vec_task import VecTask

from matplotlib import pyplot as plt

import math

from suction_cup_modelling.suction_score_calcualtor import calcualte_suction_score
from suction_cup_modelling.force_calculator import calcualte_force

# Importing DexNet
# from gqcnn.examples.policy_for_training import dexnet3
from autolab_core import (BinaryImage, CameraIntrinsics, DepthImage)
import assets.urdf_models.models_data as md

from homogeneous_trasnformation_and_conversion.rotation_conversions import *
import pandas as pd
from .motion_primitives import Primitives

from pathlib import Path
cur_path = str(Path(__file__).parent.absolute())

import einops


class RL_UR16eManipulation(VecTask):

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        self.cfg = cfg

        self.temp_flag = 1

        self.max_episode_length = self.cfg["env"]["episodeLength"]

        self.action_scale = self.cfg["env"]["actionScale"]
        self.aggregate_mode = self.cfg["env"]["aggregateMode"]

        # Controller type
        self.control_type = self.cfg["env"]["controlType"]
        assert self.control_type in {"osc", "joint_tor"},\
            "Invalid control type specified. Must be one of: {osc, joint_tor}"

        # image obs include: depth dim = 1, segmask dim = 1
        self.cfg["env"]["numObservations"] = 46800

        # actions include: delta EEF if OSC (6) or joint torques (7) + bool gripper (1)

        self.obj_randomization = self.cfg["env"]["objRandomization"]

        self.retract = self.cfg["env"]["retract"]

        self.num_objects_in_bin = self.cfg["env"]["numObjectsInBin"]

        # Values to be filled in at runtime
        # will be dict filled with relevant states to use for reward calculation
        self.states = {}
        # will be dict mapping names to relevant sim handles
        self.handles = {}
        self.num_dofs = None                    # Total number of DOFs per env
        self.actions = None                     # Current actions to be deployed

        self.models_lib = md.model_lib()

        # Tensor placeholders
        # State of root body        (n_envs, 13)
        self._root_state = None
        self._dof_state = None  # State of all joints       (n_envs, n_dof)
        self._q = None  # Joint positions           (n_envs, n_dof)
        # Joint velocities          (n_envs, n_dof)
        self._qd = None
        # State of all rigid bodies             (n_envs, n_bodies, 13)
        self._rigid_body_state = None
        self._contact_forces = None     # Contact forces in sim
        self._eef_state = None  # end effector state (at grasping point)
        self._j_eef = None  # Jacobian for end effector
        self._mm = None  # Mass matrix
        self._arm_control = None  # Tensor buffer for controlling arm
        self._gripper_control = None  # Tensor buffer for controlling gripper
        self._pos_control = None            # Position actions
        self._effort_control = None         # Torque actions
        self._ur16e_effort_limits = None        # Actuator effort limits for ur16e
        # Unique indices corresponding to all envs in flattened array
        self._global_indices = None

        # camera handles
        self.camera_handles = [[]]
        self.camera_info = None

        self.rgb_camera_tensors = []
        self.depth_camera_tensors = []
        self.mask_camera_tensors = []
        self.eef_rgb_camera_tensors = []
        self.eef_depth_camera_tensors = []
        self.eef_mask_camera_tensors = []

        self.debug_viz = self.cfg["env"]["enableDebugVis"]

        self.up_axis = "z"
        self.up_axis_idx = 2

        self.ur16e_hand = "ee_link"
        self.ur16e_wrist_3_link = "wrist_3_link"
        self.ur16e_base = "base_link"
        self.suction_gripper = "epick_end_effector"

        self.force_threshold = 0.15
        self.object_coordiante_camera = torch.tensor([0, 0, 0])
        

        # Parallelization
        self.init_camera_capture = 1

        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../configs")+'/big_collision_primitives_3d.yml') as file:
            self.world_params = yaml.load(file, Loader=yaml.FullLoader)

        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id,
                         headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)

        if self.viewer != None:
            cam_pos = gymapi.Vec3(-2.0, 0.0, 1.5)
            cam_target = gymapi.Vec3(2.0, 0.0, 1.0)
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

        self.ur16e_default_dof_pos = to_torch(
            [1.57, 0, -1.57, 0, 1.57, 0, 0], device=self.device
        )

        self.cooldown_frames = 150

        # System Identification data results
        self.cur_path = str(Path(__file__).parent.absolute())
        self.force_list = np.array([])
        self.rgb_camera_visualization = None
        self.dexnet_coordinates = np.array([])
        self.grasp_angle = torch.tensor([[0, 0, 0]])
        self.grasps_and_predictions = None
        self.unsorted_grasps_and_predictions = None

        # OSC Gains
        self.kp = to_torch(
            [150., 150., 150., 100., 100., 100.], device=self.device)
        self.kd = 2 * torch.sqrt(self.kp)
        self.kp_null = to_torch([10.]*6, device=self.device)
        self.kd_null = 2 * torch.sqrt(self.kp_null)

        # Parameters of simulator
        # when action contrib is 0, go to pregrasp pose 
        # when action contrib is 1, adjust angle to be same with grasp pose 
        # when action contrib is 2, go to grasp pose by moving in x,y,z direction
        self.action_contrib = torch.ones(self.num_envs).to(self.device)*2
        self.frame_count_contact_object = torch.zeros(self.num_envs).to(self.device)
        self.force_encounter = torch.zeros(self.num_envs).to(self.device)
        self.frame_count = torch.zeros(self.num_envs).to(self.device)
        self.free_envs_list = torch.ones(self.num_envs).to(self.device)
        self.object_pose_check_list = torch.ones(self.num_envs).to(self.device)
        self.object_target_id = torch.ones(self.num_envs).type(torch.int).to(self.device)
        self.speed = torch.ones(self.num_envs).to(self.device)*0.1
        # print("No. of environments: ", self.num_envs)

        # Parameter storage and Trackers for each environments
        self.suction_deformation_score_temp = torch.Tensor()
        self.xyz_point_temp = torch.Tensor([])
        self.grasp_angle_temp = torch.Tensor([])
        self.force_SI_temp = torch.Tensor()
        self.grasp_point_temp = torch.Tensor([])

        self.suction_deformation_score_env = {}
        self.xyz_point_env = {}
        self.grasp_angle_env = {}
        self.force_SI_env = {}
        self.grasp_point_env = {}
        self.force_contact_flag = torch.zeros(self.num_envs).to(self.device)

        self.suction_deformation_score = {}
        self.xyz_point = torch.zeros(self.num_envs, 3).to(self.device)
        self.grasp_angle = {}
        self.force_SI = torch.zeros(self.num_envs).to(self.device)
        self.grasp_point = {}
        self.last_object_pose = {}
        self.all_objects_last_pose = {}
        self.object_pose_store = torch.zeros(self.num_envs, self._root_state.shape[1], 7).to(self.device)
        self.target_object_disp_save = {}
        self.object_disp_save = {}
        self.offset_object_pose_retract = {}
        self.retract_start_pose = torch.zeros(self.num_envs, 3).to(self.device)
        self.suction_score_store_env = {}
        self.retract_up = torch.zeros(self.num_envs).to(self.device)

        self.selected_object_env = torch.zeros(self.num_envs, self.num_objects_in_bin, dtype=torch.int64).to(self.device)
        self.retract_object_state = {}

        self.track_save = torch.zeros(self.num_envs).to(self.device)
        self.config_env_count = torch.zeros(self.num_envs).to(self.device)
        self.force_list_save = {}
        self.force_pre_physics = torch.zeros(self.num_envs).to(self.device)
        self.depth_image_save = {}
        self.segmask_save = {}
        self.rgb_save = {}
        self.grasp_point_save = {}
        self.grasp_angle_save = {}
        self.suction_deformation_score_save = {}
        self.force_require_SI = {}
        self.count_step_suction_score_calculator = torch.zeros(self.num_envs)

        self.env_reset_id_env = torch.ones(self.num_envs).to(self.device)
        self.bootup_reset = torch.ones(self.num_envs).to(self.device)

        self.RL_flag = torch.ones(self.num_envs).to(self.device)
        self.run_dexnet = torch.ones(self.num_envs).to(self.device)

        # Set control limits
        self.cmd_limit = to_torch([0.1, 0.1, 0.1, 0.5, 0.5, 0.5], device=self.device).unsqueeze(0) if \
            self.control_type == "osc" else self._ur16e_effort_limits[:6].unsqueeze(0)
        self.primitive_count = torch.ones(self.num_envs).to(self.device)
        self.done = torch.zeros(self.num_envs).to(self.device)
        self.finished_prim = torch.zeros(self.num_envs).to(self.device)
        self.torch_target_area_tensor = torch.zeros(self.num_envs).to(self.device)


        self.counter = 0
        self.action = np.concatenate(self.num_envs * [np.array(["in", "right", "up", "out", "left", "down"], dtype="U10")])
        self.action = np.reshape(self.action, (self.num_envs, 6))
        self.true_target_dist = torch.zeros(self.num_envs).to(self.device)
        # self.target_dist is a tensor of shape (num_envs, 6, 3)
        # where each row is a target dist for each env
        self.target_dist = torch.zeros(self.num_envs, 6, 3).to(self.device)
        self.target_dist[:, 0, 0] = self.true_target_dist
        self.target_dist[:, 1, 1] = -self.true_target_dist
        self.target_dist[:, 2, 2] = self.true_target_dist
        self.target_dist[:, 3, 0] = -self.true_target_dist
        self.target_dist[:, 4, 1] = self.true_target_dist
        self.target_dist[:, 5, 2] = -self.true_target_dist


        # make a target_dist for each env
        self.prim = torch.zeros(self.num_envs).to(self.device)
        self.num_primitive_actions = self.cfg["env"]["numPrimitiveActions"]
        self.current_directory = os.getcwd()
        self.init_go_to_start = torch.ones(self.num_envs).to(self.device)
        self.return_pre_grasp = torch.zeros(self.num_envs).to(self.device)
        self.go_to_start = torch.ones(self.num_envs).to(self.device)
        self.success = torch.zeros(self.num_envs).to(self.device)
        self.min_distance = torch.ones(self.num_envs).to(self.device) * 100
        self.temp_action = np.empty((self.num_envs), dtype=str)

        self.weight_distance = 1.0

        # Reset all environments
        self.reset_idx(torch.arange(self.num_envs, device=self.device))

        # Refresh tensors
        self._refresh()
        # Primitives
        primitives = Primitives(self.num_envs, self.states['eef_pos'], device=self.device)
        # make an array of deep copies of Primitives for each env
        self.primitives = np.array([deepcopy(primitives) for _ in range(self.num_envs)])
        

    def create_sim(self):
        self.sim_params.up_axis = gymapi.UP_AXIS_Z
        self.sim_params.gravity.x = 0
        self.sim_params.gravity.y = 0
        self.sim_params.gravity.z = -9.81
        self.sim = super().create_sim(
            self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs(
            self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_root = os.path.join(os.path.dirname(
            os.path.abspath(__file__)), "../../assets")
        ur16e_asset_file = "urdf/Aurmar_description/robots/ur16e.urdf"

        # load ur16e asset
        asset_options = gymapi.AssetOptions()
        asset_options.flip_visual_attachments = True
        asset_options.fix_base_link = True
        asset_options.collapse_fixed_joints = False
        asset_options.disable_gravity = True
        asset_options.thickness = 0.001
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_EFFORT
        asset_options.use_mesh_materials = True
        ur16e_asset = self.gym.load_asset(
            self.sim, asset_root, ur16e_asset_file, asset_options)

        ur16e_dof_stiffness = to_torch(
            [0., 0., 0., 0., 0., 0., 0.], dtype=torch.float, device=self.device)
        ur16e_dof_damping = to_torch(
            [0., 0., 0., 0., 0., 0., 0.], dtype=torch.float, device=self.device)

        # Create table asset
        table_pos = [0.0, 0.0, 1.0]
        table_thickness = 0.05
        table_opts = gymapi.AssetOptions()
        table_opts.fix_base_link = True

        asset_options = gymapi.AssetOptions()
        asset_options.flip_visual_attachments = True
        asset_options.fix_base_link = False
        asset_options.collapse_fixed_joints = True
        asset_options.disable_gravity = False
        asset_options.thickness = 0.001
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_EFFORT
        asset_options.use_mesh_materials = True

        self.object_models = []
        objects_file = open('misc/object_list_domain_randomization.txt', 'r')
        objects = objects_file.readlines()

        self.object_count_unique = 0
        # Strips the newline character
        for object in objects:
            self.object_count_unique += 1
            for domain in range(5):
                self.object_models.append(
                    str(object.strip())+"_"+str(domain+1))
        object_model_asset_file = []
        object_model_asset = []
        for counter, model in enumerate(self.object_models):
            object_model_asset_file.append(
                "urdf_models/models/"+model+"/model.urdf")
            object_model_asset.append(self.gym.load_asset(
                self.sim, asset_root, object_model_asset_file[counter], asset_options))

        self.num_ur16e_bodies = self.gym.get_asset_rigid_body_count(
            ur16e_asset)
        self.num_ur16e_dofs = self.gym.get_asset_dof_count(ur16e_asset)

        # print("num ur16e bodies: ", self.num_ur16e_bodies)
        # print("num ur16e dofs: ", self.num_ur16e_dofs)

        # set ur16e dof properties
        ur16e_dof_props = self.gym.get_asset_dof_properties(ur16e_asset)
        self.ur16e_dof_lower_limits = []
        self.ur16e_dof_upper_limits = []
        self._ur16e_effort_limits = []
        for i in range(self.num_ur16e_dofs):
            ur16e_dof_props['driveMode'][i] = gymapi.DOF_MODE_POS if i > 6 else gymapi.DOF_MODE_EFFORT
            if self.physics_engine == gymapi.SIM_PHYSX:
                ur16e_dof_props['stiffness'][i] = ur16e_dof_stiffness[i]
                ur16e_dof_props['damping'][i] = ur16e_dof_damping[i]
            else:
                ur16e_dof_props['stiffness'][i] = 7000.0
                ur16e_dof_props['damping'][i] = 50.0

            self.ur16e_dof_lower_limits.append(ur16e_dof_props['lower'][i])
            self.ur16e_dof_upper_limits.append(ur16e_dof_props['upper'][i])
            self._ur16e_effort_limits.append(ur16e_dof_props['effort'][i])

        self.ur16e_dof_lower_limits = to_torch(
            self.ur16e_dof_lower_limits, device=self.device)
        self.ur16e_dof_upper_limits = to_torch(
            self.ur16e_dof_upper_limits, device=self.device)
        self._ur16e_effort_limits = to_torch(
            self._ur16e_effort_limits, device=self.device)
        self.ur16e_dof_speed_scales = torch.ones_like(
            self.ur16e_dof_lower_limits)

        # Define start pose for ur16e
        ur16e_start_pose = gymapi.Transform()
        # gymapi.Vec3(-0.45, 0.0, 1.0 + table_thickness / 2 + table_stand_height)
        ur16e_start_pose.p = gymapi.Vec3(-0.22, 0, 2.020)

        quat = euler_angles_to_quaternion(torch.tensor(
            [180, 0, 0]).to(self.device), "XYZ", degrees=True)
        ur16e_start_pose.r = gymapi.Quat(quat[0], quat[1], quat[2], quat[3])

        # Define start pose for table
        table_start_pose = gymapi.Transform()
        table_start_pose.p = gymapi.Vec3(*table_pos)
        table_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
        self._table_surface_pos = np.array(
            table_pos) + np.array([0, 0, table_thickness / 2])

        object_model_start_pose = []
        for counter in range(len(self.object_models)):
            object_model_start_pose.append(gymapi.Transform())
            object_model_start_pose[counter].p = gymapi.Vec3(0.0, -0.1, -10.0)
            object_model_start_pose[counter].r = gymapi.Quat(
                0.0, 0.0, 0.0, 1.0)

        # compute aggregate size
        num_ur16e_bodies = self.gym.get_asset_rigid_body_count(ur16e_asset)
        num_ur16e_shapes = self.gym.get_asset_rigid_shape_count(ur16e_asset)
        if ('cube' in self.world_params['world_model']['coll_objs']):
            cube = self.world_params['world_model']['coll_objs']['cube']
            self.num_pod_bodies = len(cube.keys())
        max_agg_bodies = num_ur16e_bodies + self.num_pod_bodies + len(self.object_models)
        max_agg_shapes = num_ur16e_shapes + self.num_pod_bodies + len(self.object_models)


        self.ur16es = []
        self.envs = []

        self.multi_body_idx = {
            "base_link": self.gym.find_asset_rigid_body_index(ur16e_asset, "base_link"),
            "wrist_3_link": self.gym.find_asset_rigid_body_index(ur16e_asset, "wrist_3_link"),
            "ee_link": self.gym.find_asset_rigid_body_index(ur16e_asset, "ee_link"),
            "epick_end_effector": self.gym.find_asset_rigid_body_index(ur16e_asset, "epick_end_effector"),
        }

        # force sensor
        sensor_pose = gymapi.Transform()
        self.gym.create_asset_force_sensor(
            ur16e_asset, self.multi_body_idx["wrist_3_link"], sensor_pose)

        # Create environments
        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)

            # Create actors and define aggregate group appropriately depending on setting
            # NOTE: ur16e should ALWAYS be loaded first in sim!
            if self.aggregate_mode >= 3:
                self.gym.begin_aggregate(
                    env_ptr, max_agg_bodies, max_agg_shapes, True)

            # Create ur16e
            self.ur16e_actor = self.gym.create_actor(
                env_ptr, ur16e_asset, ur16e_start_pose, "ur16e", i, 0, 0)
            self.gym.set_actor_dof_properties(
                env_ptr, self.ur16e_actor, ur16e_dof_props)
            self.gym.enable_actor_dof_force_sensors(env_ptr, self.ur16e_actor)

            if self.aggregate_mode == 2:
                self.gym.begin_aggregate(
                    env_ptr, max_agg_bodies, max_agg_shapes, True)

            # Create pod
            if ('cube' in self.world_params['world_model']['coll_objs']):
                cube = self.world_params['world_model']['coll_objs']['cube']
                for obj in cube.keys():
                    dims = cube[obj]['dims']
                    pose = cube[obj]['pose']
                    self.add_table(dims, pose, ur16e_start_pose,
                                   env_ptr, i, color=[0.6, 0.6, 0.6])

            if self.aggregate_mode == 1:
                self.gym.begin_aggregate(
                    env_ptr, max_agg_bodies, max_agg_shapes, True)

            # Set urdf objects
            self._object_model_id = []
            for counter in range(len(self.object_models)):
                self._object_model_id.append(self.gym.create_actor(
                    env_ptr, object_model_asset[counter], object_model_start_pose[counter], self.object_models[counter], i, 0, counter+1))

            if self.aggregate_mode > 0:
                self.gym.end_aggregate(env_ptr)

            # Store the created env pointers
            self.envs.append(env_ptr)
            self.ur16es.append(self.ur16e_actor)

            # Addign friction to the suction cup
            ur16e_handle = 0
            suction_gripper_handle = self.gym.find_actor_rigid_body_handle(
                env_ptr, ur16e_handle, self.suction_gripper)
            suction_gripper_shape_props = self.gym.get_actor_rigid_shape_properties(
                env_ptr, suction_gripper_handle)
            suction_gripper_shape_props[0].friction = 1.0
            self.gym.set_actor_rigid_shape_properties(
                env_ptr, suction_gripper_handle, suction_gripper_shape_props)

            '''
            Camera Setup
            '''
            # Camera environment setup (Back cam)
            self.camera_handles.append([])
            self.body_states = []
            self.camera_properties_back_cam = gymapi.CameraProperties()
            self.camera_properties_back_cam.enable_tensors = True
            self.camera_properties_back_cam.horizontal_fov = 70.0
            self.camera_properties_back_cam.width = 1280
            self.camera_properties_back_cam.height = 786
            camera_handle = self.gym.create_camera_sensor(
                env_ptr, self.camera_properties_back_cam)
            self.camera_base_link_translation = torch.tensor(
                [-0.48, 0.05, 0.6]).to(self.device)
            local_transform = gymapi.Transform()
            local_transform.p = gymapi.Vec3(
                self.camera_base_link_translation[0], self.camera_base_link_translation[1], self.camera_base_link_translation[2])
            camera_rotation_x = gymapi.Quat.from_axis_angle(
                gymapi.Vec3(1, 0, 0), np.deg2rad(180))
            local_transform.r = camera_rotation_x

            ur16e_base_link_handle = self.gym.find_actor_rigid_body_handle(
                env_ptr, 0, self.ur16e_base)
            self.gym.attach_camera_to_body(
                camera_handle, env_ptr, ur16e_base_link_handle, local_transform, gymapi.FOLLOW_TRANSFORM)
            self.camera_handles[i].append(camera_handle)

            # Embedded camera at gripper for real time feedback 
            ur16e_hand_link_handle = self.gym.find_actor_rigid_body_handle(
                env_ptr, 0, "ee_link")
            self.camera_gripper_link_translation = []
            self.camera_properties_gripper = gymapi.CameraProperties()
            self.camera_properties_gripper.enable_tensors = True
            self.camera_properties_gripper.horizontal_fov = 150.0
            self.camera_properties_gripper.width = 1920
            self.camera_properties_gripper.height = 1080
            camera_handle_gripper = self.gym.create_camera_sensor(
                env_ptr, self.camera_properties_gripper)
            self.camera_gripper_link_translation.append(
                torch.tensor([0.0, 0, 0]).to(self.device))
            local_transform = gymapi.Transform()
            local_transform.p = gymapi.Vec3(self.camera_gripper_link_translation[0][0],
                                            self.camera_gripper_link_translation[0][1],
                                            self.camera_gripper_link_translation[0][2])
            self.gym.attach_camera_to_body(
                camera_handle_gripper, env_ptr, ur16e_hand_link_handle, local_transform, gymapi.FOLLOW_TRANSFORM)
            self.camera_handles[i].append(camera_handle_gripper)
            ############
            rgb_camera_tensor = self.gym.get_camera_image_gpu_tensor(
                        self.sim, self.envs[i], self.camera_handles[i][0], gymapi.IMAGE_COLOR)
            torch_rgb_cam_tensor = gymtorch.wrap_tensor(rgb_camera_tensor)
            self.rgb_camera_tensors.append(torch_rgb_cam_tensor)

            mask_camera_tensor = self.gym.get_camera_image_gpu_tensor(
                    self.sim, self.envs[i], self.camera_handles[i][0], gymapi.IMAGE_SEGMENTATION)
            torch_mask_tensor = gymtorch.wrap_tensor(mask_camera_tensor)
            self.mask_camera_tensors.append(torch_mask_tensor)

            depth_camera_tensor = self.gym.get_camera_image_gpu_tensor(
                            self.sim, self.envs[i], self.camera_handles[i][0], gymapi.IMAGE_DEPTH)
            torch_depth_tensor = gymtorch.wrap_tensor(depth_camera_tensor)
            self.depth_camera_tensors.append(torch_depth_tensor)

            eef_rgb_camera_tensor = self.gym.get_camera_image_gpu_tensor(
                        self.sim, self.envs[i], self.camera_handles[i][1], gymapi.IMAGE_COLOR)
            eef_torch_rgb_cam_tensor = gymtorch.wrap_tensor(eef_rgb_camera_tensor)
            self.eef_rgb_camera_tensors.append(eef_torch_rgb_cam_tensor)

            eef_mask_camera_tensor = self.gym.get_camera_image_gpu_tensor(
                    self.sim, self.envs[i], self.camera_handles[i][1], gymapi.IMAGE_SEGMENTATION)
            eef_torch_mask_tensor = gymtorch.wrap_tensor(eef_mask_camera_tensor)
            self.eef_mask_camera_tensors.append(eef_torch_mask_tensor)

            eef_depth_camera_tensor = self.gym.get_camera_image_gpu_tensor(
                            self.sim, self.envs[i], self.camera_handles[i][1], gymapi.IMAGE_DEPTH)
            eef_torch_depth_tensor = gymtorch.wrap_tensor(eef_depth_camera_tensor)
            self.eef_depth_camera_tensors.append(eef_torch_depth_tensor)


        # Setup data
        self.init_data()

        # reset array for retract env ids
        self.env_reset_retract = torch.tensor([]).to(self.device)

        # List for storing the object poses
        self._init_object_model_state = []
        for counter in range(len(self.object_models)):
            self._init_object_model_state.append(
                torch.zeros(self.num_envs, 13, device=self.device))

    # For pod
    def add_table(self, table_dims, table_pose, robot_pose, env_ptr, env_id, color=[1.0, 0.0, 0.0]):

        table_dims = gymapi.Vec3(table_dims[0], table_dims[1], table_dims[2])

        asset_options = gymapi.AssetOptions()
        asset_options.armature = 0.001
        asset_options.fix_base_link = True
        asset_options.thickness = 0.002
        obj_color = gymapi.Vec3(color[0], color[1], color[2])
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(table_pose[0], table_pose[1], table_pose[2])
        pose.r = gymapi.Quat(
            table_pose[3], table_pose[4], table_pose[5], table_pose[6])
        table_asset = self.gym.create_box(self.sim, table_dims.x, table_dims.y, table_dims.z,
                                          asset_options)

        table_pose = robot_pose * pose
        table_handle = self.gym.create_actor(
            env_ptr, table_asset, table_pose, 'table', env_id, 0)
        self.gym.set_rigid_body_color(
            env_ptr, table_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, obj_color)
        table_shape_props = self.gym.get_actor_rigid_shape_properties(
            env_ptr, table_handle)
        table_shape_props[0].friction = 0.4
        self.gym.set_actor_rigid_shape_properties(
            env_ptr, table_handle, table_shape_props)

    def init_data(self):
        # Setup sim handles
        env_ptr = self.envs[0]
        ur16e_handle = 0
        self.handles = {
            # ur16e
            "base_link": self.gym.find_actor_rigid_body_handle(env_ptr, ur16e_handle, self.ur16e_base),
            "wrist_3_link": self.gym.find_actor_rigid_body_handle(env_ptr, ur16e_handle, self.ur16e_wrist_3_link),
            "hand": self.gym.find_actor_rigid_body_handle(env_ptr, ur16e_handle, self.ur16e_hand),
        }

        for counter in range(len(self.object_models)):
            self.handles[self.object_models[counter]+"_body_handle"] = self.gym.find_actor_rigid_body_handle(
                self.envs[0], self._object_model_id[counter], self.object_models[counter])
            object_shape_props = self.gym.get_actor_rigid_shape_properties(
                env_ptr, self._object_model_id[counter])
            object_shape_props[0].friction = 0.2
            self.gym.set_actor_rigid_shape_properties(
                env_ptr, self._object_model_id[counter], object_shape_props)

        # Get total DOFs
        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs

        # Setup tensor buffers
        _actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(
            self.sim)
        _dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        _rigid_body_state_tensor = self.gym.acquire_rigid_body_state_tensor(
            self.sim)
        self._root_state = gymtorch.wrap_tensor(
            _actor_root_state_tensor).view(self.num_envs, -1, 13)
        self._dof_state = gymtorch.wrap_tensor(
            _dof_state_tensor).view(self.num_envs, -1, 2)
        self._rigid_body_state = gymtorch.wrap_tensor(
            _rigid_body_state_tensor).view(self.num_envs, -1, 13)
        self._q = self._dof_state[..., 0]
        self._qd = self._dof_state[..., 1]
        self._eef_state = self._rigid_body_state[:, self.handles["hand"], :]
        self._base_link = self._rigid_body_state[:,
                                                 self.handles["base_link"], :]
        self._wrist_3_link = self._rigid_body_state[:,
                                                    self.handles["wrist_3_link"], :]
        _jacobian = self.gym.acquire_jacobian_tensor(self.sim, "ur16e")
        jacobian = gymtorch.wrap_tensor(_jacobian)
        hand_joint_index = self.gym.get_actor_joint_dict(
            env_ptr, ur16e_handle)['gripper_joint']
        self._j_eef = jacobian[:, hand_joint_index, :, :6]
        _massmatrix = self.gym.acquire_mass_matrix_tensor(self.sim, "ur16e")
        mm = gymtorch.wrap_tensor(_massmatrix)
        self._mm = mm[:, :6, :6]

        self._object_model_state = []
        for counter in range(len(self.object_models)):
            self._object_model_state.append(
                self._root_state[:, self._object_model_id[counter], :])

        # Initialize actions
        self._pos_control = torch.zeros(
            (self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
        self._effort_control = torch.zeros_like(self._pos_control).to(self.device)
        # Initialize control
        self._arm_control = self._effort_control[:, :]

        # Initialize indices    ------ > self.num_envs * num of actors
        self._global_indices = torch.arange(self.num_envs * (self.num_pod_bodies + 1 + len(self.object_models)), dtype=torch.int32,
                                            device=self.device).view(self.num_envs, -1)

        '''
        camera intrinsics for back cam and gripper cam
        '''
        # cam_vinv_back_cam = torch.inverse((torch.tensor(self.gym.get_camera_view_matrix(self.sim, self.envs[i], self.camera_handles[i][0])))).to(self.device)
        cam_proj_back_cam = torch.tensor(self.gym.get_camera_proj_matrix(
            self.sim, self.envs[0], self.camera_handles[0][0]), device=self.device)
        self.width_back_cam = self.camera_properties_back_cam.width
        self.height_back_cam = self.camera_properties_back_cam.height
        self.fx_back_cam = self.width_back_cam/(2/cam_proj_back_cam[0, 0])
        self.fy_back_cam = self.height_back_cam/(2/cam_proj_back_cam[1, 1])
        self.cx_back_cam = self.width_back_cam/2
        self.cy_back_cam = self.height_back_cam/2
        self.camera_intrinsics_back_cam = CameraIntrinsics(frame="camera_back", fx=self.fx_back_cam, fy=self.fy_back_cam,
                                                           cx=self.cx_back_cam, cy=self.cy_back_cam, skew=0.0,
                                                           height=self.height_back_cam, width=self.width_back_cam)
        self.suction_score_object = calcualte_suction_score(
            self.camera_intrinsics_back_cam, self.device)
        # self.dexnet_object = dexnet3(self.camera_intrinsics_back_cam)
        # self.dexnet_object.load_dexnet_model()

        # print("focal length in x axis: ", self.fx_back_cam)
        # print("focal length in y axis: ", self.fy_back_cam)
        # cam_vinv_gripper = torch.inverse((torch.tensor(self.gym.get_camera_view_matrix(self.sim, self.envs[i], self.camera_handles[i][0])))).to(self.device)
        cam_proj_gripper = torch.tensor(self.gym.get_camera_proj_matrix(
            self.sim, self.envs[0], self.camera_handles[0][1]), device=self.device)
        self.width_gripper = self.camera_properties_gripper.width
        self.height_gripper = self.camera_properties_gripper.height
        self.fx_gripper = self.width_gripper/(2/cam_proj_gripper[0, 0])
        self.fy_gripper = self.height_gripper/(2/cam_proj_gripper[1, 1])
        self.cx_gripper = self.width_gripper/2
        self.cy_gripper = self.height_gripper/2
        self.camera_intrinsics_gripper = CameraIntrinsics(frame="camera_gripper", fx=self.fx_gripper, fy=self.fy_gripper,
                                                          cx=self.cx_gripper, cy=self.cy_gripper, skew=0.0,
                                                          height=self.height_gripper, width=self.width_gripper)
        self.suction_score_object_gripper = calcualte_suction_score(
            self.camera_intrinsics_gripper, self.device)
        self.force_object = calcualte_force()

    def _update_states(self):
        self.states.update({
            # ur16e
            "base_link": self._base_link[:, :7],
            "wrist_3_link": self._wrist_3_link[:, :7],
            "wrist_3_link_vel": self._wrist_3_link[:, 7:],
            "q": self._q[:, :],
            "q_gripper": self._q[:, -2:],
            "eef_pos": self._wrist_3_link[:, :3],
            "eef_quat": self._wrist_3_link[:, 3:7],
            "eef_vel": self._wrist_3_link[:, 7:],
        })

    def _refresh(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)
        self.gym.refresh_mass_matrix_tensors(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)

        # Refresh states
        self._update_states()

    def deploy_actions(self, env_ids, pos):
        # Reset the internal obs accordingly
        self._q[env_ids, :] = pos.to(self.device)
        self._qd[env_ids, :] = torch.zeros_like(self._qd[env_ids])
        # Set any position control to the current position, and any vel / effort control to be 0
        # NOTE: Task takes care of actually propagating these controls in sim using the SimActions API
        self._pos_control[env_ids, :] = pos
        self._effort_control[env_ids, :] = torch.zeros_like(pos)
        # Deploy updates
        multi_env_ids_int32 = self._global_indices[env_ids, 0].flatten()
        self.gym.set_dof_position_target_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(
                                                            self._pos_control),
                                                        gymtorch.unwrap_tensor(
                                                            multi_env_ids_int32),
                                                        len(multi_env_ids_int32))
        self.gym.set_dof_actuation_force_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(
                                                            self._effort_control),
                                                        gymtorch.unwrap_tensor(
                                                            multi_env_ids_int32),
                                                        len(multi_env_ids_int32))
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(
                                                  self._dof_state),
                                              gymtorch.unwrap_tensor(
                                                  multi_env_ids_int32),
                                              len(multi_env_ids_int32))

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
            # self.RL_flag[env_count] = 0
            # How many objects should we spawn 2 or 3
            if self.obj_randomization:
                probabilities = [0.0, 0.0, 1.0]
            else:
                probabilities = [0.0, 0.0, 1.0]
            # randomly select number of objs
            random_number = self.random_number_with_probabilities(probabilities)
            random_number += 1
            object_list_env = torch.zeros(self._root_state.shape[1], 7)   
            # Sample objects from object set
            object_set = range(1, self.object_count_unique+1)
            if self.obj_randomization:
                selected_object = random.sample(object_set, random_number)
            else:
                selected_object = [1, 5, 3]
            # store the sampled weight for each obj
            list_objects_domain_randomizer = torch.tensor([])
            # Fixed obj pose for fixed scene
            offset_object1 = np.array([0.55, 0., 1.45, 0.0, 0.0, 0.0])
            offset_object2 = np.array([0.6, 0., 1.45, 0.0, 0.0, 0.0])
            offset_object3 = np.array([0.62, -0.1, 1.45, 0.0, 0.0, 0.0])
            
            offset_objects = [offset_object2, offset_object3, offset_object1]
            # add gaussian noise to the first 2 elements of the 3 offsets
            mean = 0
            std_dev = 0.0
            offset_objects = [np.concatenate([offset[:2] + np.random.normal(mean, std_dev, 2), offset[2:]]) for offset in offset_objects]
            # print("noisy offset object", noisy_offset_objects)

            # apply sampled object pose and weight
            for object_count in selected_object:
                domain_randomizer = random_number = random.choice(
                    [1, 2, 3, 4, 5])
                if self.obj_randomization:
                    # pose
                    offset_object = np.array([np.random.uniform(0.57, 0.7, 1).reshape(
                        1,)[0], np.random.uniform(-0.15, 0.10, 1).reshape(1,)[0], 1.55, np.random.uniform(-math.pi/2, math.pi/2, 1).reshape(1,)[0],
                        np.random.uniform(-math.pi/2, math.pi/2, 1).reshape(1,)[0], np.random.uniform(-math.pi/2, math.pi/2, 1).reshape(1,)[0]])
                    domain_randomizer = random_number = random.choice(
                    [1, 2, 3, 4, 5])
                else:
                    # apply fixed poses and weights
                    offset_object = np.array([np.random.uniform(0.57, 0.7, 1).reshape(
                        1,)[0], np.random.uniform(-0.15, 0.10, 1).reshape(1,)[0], 1.55, np.random.uniform(-math.pi/2, math.pi/2, 1).reshape(1,)[0],
                        np.random.uniform(-math.pi/2, math.pi/2, 1).reshape(1,)[0], np.random.uniform(-math.pi/2, math.pi/2, 1).reshape(1,)[0]])
                    domain_randomizer = random_number = random.choice(
                    [1])
                ##############################################
                
                # print("object count", object_count)
                if object_count == 5 and not self.obj_randomization:
                    offset_object = offset_objects[1]
                elif not self.obj_randomization: 
                    offset_object = offset_objects[object_count-1]
                ##############################################
                # set position and orientation
                quat = euler_angles_to_quaternion(
                    torch.tensor(offset_object[3:6]), "XYZ", degrees=False)
                offset_object = np.concatenate(
                    [offset_object[:3], quat.cpu().numpy()])
                # calculate buffer index
                item_config = (object_count-1)*5 + domain_randomizer
                # change buffer values
                object_list_env[item_config] = torch.tensor(offset_object)
                list_objects_domain_randomizer = torch.cat(
                    (list_objects_domain_randomizer, torch.tensor([item_config])))
            # signify env for selected objs and poses
            self.selected_object_env[env_count] = list_objects_domain_randomizer
            # print("object list env", object_list_env)
            self.object_pose_store[env_count] = object_list_env

        # print(env_count, "objects spawned in each environment",
            #   self.selected_object_env)
        # pos = torch.tensor(np.random.uniform(low=-6.2832, high=6.2832, size=(6,))).to(self.device).type(torch.float)
        # pos = tensor_clamp(pos.unsqueeze(0), self.ur16e_dof_lower_limits.unsqueeze(0), self.ur16e_dof_upper_limits)
        pos = tensor_clamp(self.ur16e_default_dof_pos.unsqueeze(
            0), self.ur16e_dof_lower_limits.unsqueeze(0), self.ur16e_dof_upper_limits)
        pos = pos.repeat(len(env_ids), 1)

        # reinitializing the variables - TODO: Parallelization
        for env_id in env_ids:
            self.action_contrib[env_id] = 2
            self.force_encounter[env_id] = 0
            self.frame_count_contact_object[env_id] = 0
            self.frame_count[env_id] = 0
            self.free_envs_list[env_id] = torch.tensor(1)
            self.object_pose_check_list[env_id] = torch.tensor(3)
            self.speed[env_id] = torch.tensor(0.1)
            self.run_dexnet[env_id] = torch.tensor(1)
            self.RL_flag[env_id] = torch.tensor(1)
            self.go_to_start[env_id] = 1
            self.init_go_to_start[env_id] = torch.tensor(1)
            self.return_pre_grasp[env_id] = torch.tensor(0)
            self.primitive_count[env_id.item()] = torch.tensor(1)
            self.min_distance[env_id] = torch.tensor(100)
            self.torch_target_area_tensor[env_id] = 0
            self.force_encounter[env_id] = 0
            self.suction_score_store_env[env_id] = torch.tensor(0.0)
            self.retract_start_pose[env_id] = torch.zeros(3)
            self.offset_object_pose_retract[env_id] = None
            self.retract_up[env_id] = 0
            self.count_step_suction_score_calculator[env_id] = 0
            self.env_reset_id_env[env_id] = 1



        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0

        return pos

    '''
    Resetting object poses and quering object pose from the saved object pose
    '''

    def reset_object_pose(self, env_ids, reset_retract_object=False):
        if (env_ids != None):
            for counter in range(len(self.object_models)):
                self.object_poses = torch.zeros(0, 7).to(self.device)
                for env_count in env_ids:
                    # resetting force list
                    # randomly spawning objects
                    retKey = self.object_pose_store[env_count.item()][counter + 1]
                    if torch.all(retKey == torch.zeros(7).to(self.device)):
                        object_pose_env = torch.tensor(
                            [[counter/4, 1, 0.5, 0.0, 0.0, 0.0, 1.0]]).to(self.device)
                    else:
                        object_pose_env = self.object_pose_store[env_count.item(
                        )][counter+1].clone().detach().to(self.device)
                        # quat = self.object_pose_store[env_count.item(
                        # )][counter+1][3:7]
                        # object_pose_env = torch.cat([object_pose_env[:3], quat])

                        # quat = euler_angles_to_quaternion(
                        #     object_pose_env[3:6], "XYZ", degrees=False)
                        # object_pose_env = torch.cat([object_pose_env[:3], quat])
                        object_pose_env = object_pose_env.unsqueeze(0)
                    self.object_poses = torch.cat(
                        [self.object_poses, object_pose_env])

                self._reset_init_object_state(
                    env_ids=env_ids, object=self.object_models[counter], offset=self.object_poses)

            for env_count in env_ids:
                # self.force_list_save[env_count.item()] = None
                self.all_objects_last_pose[env_count.item()] = {}
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
                self.speed[env_id] = 0.1
                self.force_contact_flag[env_id.item()] = torch.tensor(
                    0).type(torch.bool)
                # if self.done[env_count] == 1:
                self.run_dexnet[env_id] = torch.tensor(1)
                # self.RL_flag[env_count] = torch.tensor(1)
                self.go_to_start[env_id] = 1
                # self.init_go_to_start[env_id] = torch.tensor(1)
                self.return_pre_grasp[env_id] = torch.tensor(0)
                self.primitive_count[env_id.item()] = torch.tensor(1)
                
                # self.finished_prim[env_id] = torch.tensor(0)

            self.object_movement_enabled = 0
            self.cmd_limit = to_torch(
                [0.1, 0.1, 0.1, 0.5, 0.5, 0.5], device=self.device).unsqueeze(0)
        if (reset_retract_object == True and env_ids == None):
            multi_env_ids_cubes_int32 = self._global_indices[self.env_reset_retract, -len(
                self.object_models):].flatten()
            self.gym.set_actor_root_state_tensor_indexed(
                self.sim, gymtorch.unwrap_tensor(self._root_state),
                gymtorch.unwrap_tensor(multi_env_ids_cubes_int32), len(multi_env_ids_cubes_int32))
        elif (reset_retract_object == True and len(env_ids) != 0):
            env_ids = torch.cat(
                (env_ids, self.env_reset_retract), axis=0)
            multi_env_ids_cubes_int32 = self._global_indices[env_ids, -len(
                self.object_models):].flatten()
            self.gym.set_actor_root_state_tensor_indexed(
                self.sim, gymtorch.unwrap_tensor(self._root_state),
                gymtorch.unwrap_tensor(multi_env_ids_cubes_int32), len(multi_env_ids_cubes_int32))
        elif (reset_retract_object == False and len(env_ids) != 0):
            multi_env_ids_cubes_int32 = self._global_indices[env_ids, -len(
                self.object_models):].flatten()
            self.gym.set_actor_root_state_tensor_indexed(
                self.sim, gymtorch.unwrap_tensor(self._root_state),
                gymtorch.unwrap_tensor(multi_env_ids_cubes_int32), len(multi_env_ids_cubes_int32))

        self.env_reset_retract = torch.tensor([]).to(self.device)
    
    def reset_idx(self, env_ids):
        pos = self.reset_init_arm_pose(env_ids)
        self.deploy_actions(env_ids, pos)
        # Update objects states
        self.reset_object_pose(env_ids)

    def reset_default_arm_pose(self, env_ids):
        pos = tensor_clamp(self.ur16e_default_dof_pos.unsqueeze(
            0), self.ur16e_dof_lower_limits.unsqueeze(0), self.ur16e_dof_upper_limits)
        pos = pos.repeat(len(env_ids), 1)
        return pos
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
            # print("npmindx", np.min(dx))
            if (np.min(dx) < -3.0):
                return True
            else:
                return False
        except:
            return False

    def compute_observations(self):
        self._refresh()
        # pose state observations
        #   pose and quaternion of end effector and respective joint angles

        # obs = ["eef_pos", "eef_quat"]
        # obs += ["q_gripper"] if self.control_type == "osc" else ["q"]
        # self.obs_buf = torch.cat([self.states[ob] for ob in obs], dim=-1)
        # return self.obs_buf

        # image observations
        self.obs_buf = torch.zeros(93600).to(self.device)
        
        # print("object disp save", self.object_disp_save[0])

        # torch_rgb_cameras = torch.FloatTensor(self.rgb_camera_tensors).to(self.device)

        if self.finished_prim.sum() > 0:
            torch_scrap_tensor = torch.tensor([]).to(self.device)
            torch_prim_tensor = self.finished_prim.clone().detach()
            envs_finished_prim = torch.nonzero(torch_prim_tensor).long().squeeze(1)
            for env_count in envs_finished_prim:
                # print("self.force_list_save", self.force_list_save.keys())
                # print("env_count", env_count)
                if env_count.item() in self.force_list_save.keys() and self.force_list_save[env_count.item()] is not None:
                    # print("self.force_list_save.shape", self.force_list_save[env_count.item()].shape)
                    oscillation = self.detect_oscillation(self.force_list_save[env_count.item()])
                    self.force_list_save[env_count.item()] = None
                else:
                    oscillation = False
                torch_scrap_tensor = torch.cat((torch_scrap_tensor, torch.tensor([oscillation]).to(self.device)))

                # print("oscillation", oscillation)
                torch_rgb_tensor = self.rgb_camera_tensors[env_count]
                rgb_image = torch_rgb_tensor.to(self.device)
                rgb_image_copy = torch.reshape(
                    rgb_image, (rgb_image.shape[0], -1, 4))[..., :3]

                # self.rgb_save[env_count] = rgb_image_copy[180:660,
                #                                             410:1050].clone().detach().cpu().numpy()
                # try:
                #     # new_dir_name = str(
                #     #         env_count)+"_"+str(self.track_save[env_count].type(torch.int).item())
                #     # print("new_dir_name", new_dir_name)
                #     os.mkdir(
                #         cur_path+"/../../System_Identification_Data/AutoEncoder/"+self.new_dir_name)
                # except:
                #     pass
                # save_dir_rgb_npy = cur_path+"/../../System_Identification_Data/AutoEncoder/" + \
                #     self.new_dir_name+"/rgb_" + str(int(env_count)) + "_"+str(self.track_save[env_count].type(torch.int).item()) + "_" + str(int(self.primitive_count[env_count].item())) +".npy"
                # if self.save_data:
                #     np.save(save_dir_rgb_npy, self.rgb_save[env_count])
                    # print("saved rgb image")
                

            torch_depth_cameras = torch.stack(self.depth_camera_tensors).to(self.device)
            torch_depth_tensor = torch_depth_cameras[envs_finished_prim]

            torch_segmask_cameras = torch.stack(self.mask_camera_tensors).to(self.device)
            prev_target_area_tensor = self.torch_target_area_tensor
            torch_segmask_tensor = torch_segmask_cameras[envs_finished_prim]
            torch_segmask_tensor = torch_segmask_tensor[:, 280:460, 510:770]
            torch_segmask_tensor = einops.rearrange(torch_segmask_tensor, 'b h w -> b (h w)')
            label = self.object_target_id[envs_finished_prim]
            label = label.unsqueeze(1).expand(torch_segmask_tensor.shape)
            # create torch target area tensor (ne) from torch_segmask_tensor where it counts number of pixels that is 255
            torch_target_area_tensor = torch.sum(torch_segmask_tensor == label, dim=1)
            # print("torch_target_area_tensor", torch_target_area_tensor)
            diff_target_area_tensor = torch.zeros(envs_finished_prim.shape[0]).to(self.device)
            # print(self.primitive_count[envs_finished_prim])
            indicies = torch.where(self.primitive_count[envs_finished_prim] > 1)[0]
            # print("indicies", indicies)
            if indicies.shape[0] > 0:
                diff_target_area_tensor[indicies] = torch_target_area_tensor[indicies] - prev_target_area_tensor[envs_finished_prim][indicies]
                # print("!!!!!diff_target_area_tensor", diff_target_area_tensor)
            self.torch_target_area_tensor[envs_finished_prim] = torch_target_area_tensor.float()
            scaled_diff = torch.tensor(1e-4).to(self.device)
            scaled_diff_tensor = diff_target_area_tensor.to(self.device) * scaled_diff
            scaled_diff_tensor = torch.clamp(scaled_diff_tensor, -0.2, 0.2)
            torch_success_tensor = self.success[envs_finished_prim].clone().detach()
            torch_success_tensor = torch_success_tensor + scaled_diff_tensor

            # reset if success
            self.success[envs_finished_prim] = torch.tensor(0).float().to(self.device)
            torch_done_tensor = self.done[envs_finished_prim].clone().detach()
            # reset if done
            self.done[envs_finished_prim] = torch.tensor(0).float().to(self.device)
            torch_indicies_tensor = envs_finished_prim

            # crop depth image
            torch_depth_tensor = torch_depth_tensor[:, 280:460, 510:770]

            torch_depth_tensor = einops.rearrange(torch_depth_tensor, 'b h w -> b (h w)')

            torch_segmask_tensor = torch.where(torch_segmask_tensor == label, torch.tensor(255).to(self.device), torch_segmask_tensor)

            torch_primcount_tensor = self.primitive_count[envs_finished_prim].clone().detach()

            
            self.obs_buf = torch.cat((torch_depth_tensor, torch_segmask_tensor), dim=1).squeeze(0)

            if torch_indicies_tensor.shape[0] > 1:
                self.obs_buf = torch.cat((self.obs_buf,  torch_primcount_tensor.unsqueeze(0).T.to(self.device)), dim=1)
                self.obs_buf = torch.cat((self.obs_buf,  torch_scrap_tensor.unsqueeze(0).T.to(self.device)), dim=1)
                self.obs_buf = torch.cat((self.obs_buf,  torch_success_tensor.unsqueeze(0).T.to(self.device)), dim=1)
                self.obs_buf = torch.cat((self.obs_buf,  torch_done_tensor.unsqueeze(0).T.to(self.device)), dim=1)
                self.obs_buf = torch.cat((self.obs_buf,  torch_indicies_tensor.unsqueeze(0).T.to(self.device)), dim=1)
            else:
                self.obs_buf = torch.cat((self.obs_buf,  torch_primcount_tensor.to(self.device)), dim=0)
                self.obs_buf = torch.cat((self.obs_buf,  torch_scrap_tensor.to(self.device)), dim=0)
                self.obs_buf = torch.cat((self.obs_buf,  torch_success_tensor.to(self.device)), dim=0)
                self.obs_buf = torch.cat((self.obs_buf,  torch_done_tensor.to(self.device)), dim=0)
                self.obs_buf = torch.cat((self.obs_buf,  torch_indicies_tensor.to(self.device)), dim=0)
            
            if torch_indicies_tensor.shape[0] == 1:
                self.obs_buf = self.obs_buf.unsqueeze(0)    
            return self.obs_buf    
        else:
            return None

    
    def compute_reward(self):
        # suction score
        suction_deformation_score = self.suction_deformation_score_temp
        # suction_deformation_score = suction_deformation_score.unsqueeze(-1)
        # # force score
        # if (suction_deformation_score > 0):
        #     force_SI = self.force_object.regression(
        #         suction_deformation_score)
        # else:
        #     force_SI = torch.tensor(0).to(self.device)
        # force_SI = force_SI.unsqueeze(-1)

        # return reward
        reward = suction_deformation_score
        # if done = self.RL_flag[env_id.item()] = torch.tensor(1)
        return reward

    def _reset_init_object_state(self, object, env_ids, offset):
        """
        Simple method to sample @cube's position based on self.startPositionNoise and self.startRotationNoise, and
        automaticlly reset the pose internally. Populates the appropriate self._init_cubeX_state

        If @check_valid is True, then this will also make sure that the sampled position is not in contact with the
        other cube.

        Args:
            cube(str): Which cube to sample location for. Either 'A' or 'B'
            env_ids (tensor or None): Specific environments to reset cube for
            check_valid (bool): Whether to make sure sampled position is collision-free with the other cube.
        """

        # Initialize buffer to hold sampled values
        num_resets = len(env_ids)
        sampled_object_state = torch.zeros(num_resets, 13, device=self.device)

        # Get correct references depending on which one was selected
        for i in range(len(self.object_models)):
            if object == self.object_models[i]:
                this_object_state_all = self._init_object_model_state[i]

        # For variable height
        # firs row H 1.7, second row G 1.5, third row F 1.35, fourth row E 1.2
        # sampling height at which the cube will be dropped

        # Indexes corresponding to envs we're still actively sampling for
        active_idx = torch.arange(num_resets, device=self.device)
        sampled_object_state[:, 3:7] = offset[:, 3:7]
        sampled_object_state[:, :3] = offset[:, :3]

        # Lastly, set these sampled values as the new init state
        this_object_state_all[env_ids, :] = sampled_object_state

    def _compute_osc_torques(self, dpose):
        # Solve for Operational Space Control # Paper: khatib.stanford.edu/publications/pdfs/Khatib_1987_RA.pdf
        # Helpful resource: studywolf.wordpress.com/2013/09/17/robot-control-4-operation-space-control/
        q, qd = self._q[:, :6], self._qd[:, :6]
        mm_inv = torch.inverse(self._mm)
        m_eef_inv = self._j_eef @ mm_inv @ torch.transpose(self._j_eef, 1, 2)
        m_eef = torch.inverse(m_eef_inv)

        # Transform our cartesian action `dpose` into joint torques `u`
        u = torch.transpose(self._j_eef, 1, 2) @ m_eef @ (
            self.kp * dpose[:, :6] - self.kd * self.states["wrist_3_link_vel"]).unsqueeze(-1)

        # Nullspace control torques `u_null` prevents large changes in joint configuration
        # They are added into the nullspace of OSC so that the end effector orientation remains constant
        # roboticsproceedings.org/rss07/p31.pdf
        j_eef_inv = m_eef @ self._j_eef @ mm_inv
        u_null = self.kd_null * -qd + self.kp_null * (
            (self.ur16e_default_dof_pos[:6] - q + np.pi) % (2 * np.pi) - np.pi)
        u_null[:, self.num_ur16e_dofs:] *= 0
        u_null = self._mm @ u_null.unsqueeze(-1)
        u += (torch.eye((6), device=self.device).unsqueeze(0) -
              torch.transpose(self._j_eef, 1, 2) @ j_eef_inv) @ u_null

        # Clip the values to be within valid effort range
        u = tensor_clamp(u.squeeze(-1),
                         -self._ur16e_effort_limits[:6].unsqueeze(0), self._ur16e_effort_limits[:6].unsqueeze(0))

        return u

    def orientation_error(self, desired, current):
        '''
        input: desired orientation - quaterion, current orientation - quaterion
        output: orientation err - euler
        '''
        cc = quat_conjugate(current)
        q_r = quat_mul(desired, cc)
        return q_r[:3] * torch.sign(q_r[3]).unsqueeze(-1)

    def reset_pre_grasp_pose(self, env_ids):
        pos = torch.zeros(0, self.num_ur16e_dofs).to(self.device)
        for _ in env_ids:
            joint_poses_list = torch.load(
                f"{cur_path}/../../misc/joint_poses.pt")
            temp_pos = joint_poses_list[torch.randint(
                0, len(joint_poses_list), (1,))[0]].to(self.device)
            temp_pos = torch.tensor([-0.2578, -1.9463, 1.7880, 0.1189, 1.4177, -0.4511]).to(self.device)
            temp_pos = torch.reshape(temp_pos, (1, len(temp_pos)))
            temp_pos = torch.cat(
                (temp_pos, torch.tensor([[0]]).to(self.device)), dim=1)
            pos = torch.cat([pos, temp_pos])
        return pos        

    def pre_physics_step(self, actions):
        assert actions.shape == (self.num_envs, 4), "actions must be of shape (num_envs, 4), got {}".format(
            actions.shape)
        actions[:, 0] = torch.clamp(actions[:, 0], -0.11, 0.11)
        actions[:, 1] = torch.clamp(actions[:, 1], -0.02, 0.1)
        actions[:, 2] = torch.clamp(actions[:, 2], 0.0, 0.28)
        actions[:, 3] = torch.clamp(actions[:, 3], -0.22, 0.22)

        '''
        Camera access in the pre physics step to compute the force using suction cup deformation score
        '''
        # communicate physics to graphics system
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)
        self.gym.step_graphics(self.sim)
        # render the camera sensors
        # self.gym.render_all_camera_sensors(self.sim)
        def render_cameras():   
            self.gym.render_all_camera_sensors(self.sim)
            self.gym.start_access_image_tensors(self.sim)
            # Updated in self.rgb_camera_tensors and self.depth_camera_tensors and self.mask_camera_tensors
            self.gym.end_access_image_tensors(self.sim)
        render_cameras()
        self.gym.refresh_dof_state_tensor(self.sim)
        '''
        Commands to the arm for eef control
        '''
        actions = torch.concat((torch.tensor(self.num_envs*[[0.0001, 0., 0., 0., 0., 0., 0.3]]).to(self.device), actions), dim=1)
        assert actions.shape == (self.num_envs, 11), "actions must be of shape (num_envs, 11), got {}".format(
            actions.shape)
        
        self.actions = torch.zeros(self.num_envs, 11).to(self.device)
        # Before each loop this will track all the environments where a condition for reset has been called
        env_list_reset_objects = torch.tensor([]).to(self.device)
        env_list_reset_arm_pose = torch.tensor([]).to(self.device)
        env_complete_reset = torch.tensor([]).to(self.device)
        env_list_reset_default_arm_pose = torch.tensor([]).to(self.device)
        
        self.cmd_limit = to_torch(
            [0.1, 0.1, 0.1, 0.5, 0.5, 0.5], device=self.device).unsqueeze(0)
        
        # Set pose if object cooled down
        # envs where frame count is greater than self.cooldown_frames and object_pose_check_list is greater than 1
        # import pdb; pdb.set_trace()
        obj_cooled_down_envs = torch.where((self.frame_count == self.cooldown_frames) & (self.object_pose_check_list >= 1))[0].to(self.device)
        if len(obj_cooled_down_envs) > 0:
            # store all current poses in self.object_pose_store
            # WARN: WILL NOT WORK FOR ENVS WITH DIFFERENT NUMBER OF OBJECTS
            # bin_objects_current_pose_tensor contains the current poses of the objects in the bin
            # shape (num_envs, num_objects_in_bin (3), 7)
            # import pdb; pdb.set_trace()
            bin_objects_current_pose_tensor = torch.zeros(self.num_envs, self._root_state.shape[1], 13).to(self.device)
            selected_object_env_idx = torch.zeros(self.num_envs, self._root_state.shape[1]).to(self.device)
            selected_object_env_idx.scatter_(1, self.selected_object_env - 1, 1)
            num_objects = self.selected_object_env.shape[1]
            bin_objects_current_pose_tensor = self._root_state[selected_object_env_idx.type(torch.bool)]
            bin_objects_current_pose_tensor = bin_objects_current_pose_tensor[:, :7].to(self.device) 
            # print("bin_objects_current_pose_tensor", bin_objects_current_pose_tensor)
            self.object_pose_store[selected_object_env_idx.type(torch.bool)] = bin_objects_current_pose_tensor
            env_list_reset_objects = torch.cat((env_list_reset_objects, obj_cooled_down_envs), axis=0)
            self.object_pose_check_list[obj_cooled_down_envs] -= 1

        check_fallen_envs = torch.where((self.frame_count == self.cooldown_frames) & (self.object_pose_check_list == 0))[0].to(self.device)
        if len(check_fallen_envs) > 0:
            # check if object fell 
            all_obj_poses = self.object_pose_store[check_fallen_envs]
            object_fell_envs = torch.where((all_obj_poses[:, :, 2] < 1.6) & (all_obj_poses[:, :, 2] > 1.3) & (all_obj_poses[:, :, 1] > 0.10)
                                            & (all_obj_poses[:, :, 1] < -0.18) & (all_obj_poses[:, :, 0] > 0.95) & (all_obj_poses[:, :, 0] < 0.2))[0].to(self.device)
            # self.free_envs_list[check_fallen_envs] = 0
            if len(object_fell_envs) > 0:
                env_complete_reset = torch.cat((env_complete_reset, object_fell_envs), axis=0)
                # print("object fell")
        
        # Store grasp point and grasp angle
        # print("self.frame_count", self.frame_count)
        store_gp_envs = torch.where((self.free_envs_list == 1) & (self.object_pose_check_list == 0) & (self.frame_count == self.cooldown_frames))[0].to(self.device)
        if len(store_gp_envs) > 0:
            RL_ready_envs = torch.where((self.RL_flag == 1) & (self.free_envs_list == 1) & (self.object_pose_check_list == 0) & (self.frame_count == self.cooldown_frames))[0].to(self.device)
            if len(RL_ready_envs) > 0:
                # print("IN STORE GP ENVS, RL_READY ENVS")
                self.object_target_id[RL_ready_envs] = 21
                env_list_reset_arm_pose = torch.cat(
                                    (env_list_reset_arm_pose, RL_ready_envs), axis=0)
                env_list_reset_objects = torch.cat(
                    (env_list_reset_objects, RL_ready_envs), axis=0)


            not_RL_envs = torch.where((self.RL_flag == 0) & (self.free_envs_list == 1) & (self.object_pose_check_list == 0) & (self.frame_count == self.cooldown_frames))[0].to(self.device)
            if len(not_RL_envs) > 0:
                # print("IN STORE GP ENVS, RL NOT READY ENVS")
                mask_tensor = torch.stack(self.mask_camera_tensors)
                segmask_dexnet = mask_tensor[not_RL_envs].clone().detach()
                rgb_tensor = torch.stack(self.rgb_camera_tensors)
                rgb_image = rgb_tensor[not_RL_envs].to(self.device)
                rgb_image_copy = torch.reshape(
                    rgb_image, (rgb_image.shape[0], rgb_image.shape[1], -1, 4))[..., :3]
                depth_tensor = torch.stack(self.depth_camera_tensors)
                depth_image = depth_tensor[not_RL_envs].to(self.device)
                depth_image = -depth_image
                # set segmask target object to 255
                target_obj_masks = torch.where(segmask_dexnet[:, 180:660, 410:1050] == self.object_target_id[not_RL_envs].reshape(-1, 1, 1), 255, 0)

                # Calculate centroids
                nonzero_indices = torch.nonzero(target_obj_masks == 255)
                # nonzero_indices is a tensor of shape (3282, 3) where each row is (batch_idx, y, x)
                # separate out each batch
                batch_indices = nonzero_indices[:, 0]
                batch_indices_unique = torch.unique(batch_indices)
                centroid_x = torch.zeros_like(batch_indices_unique, dtype=torch.float32)
                centroid_y = torch.zeros_like(batch_indices_unique, dtype=torch.float32)
                for i, batch_idx in enumerate(batch_indices_unique):
                    # get all indices in the batch
                    batch_indices_mask = batch_indices == batch_idx
                    # get all x and y indices in the batch
                    x_indices = nonzero_indices[batch_indices_mask, 2]
                    y_indices = nonzero_indices[batch_indices_mask, 1]
                    # calculate centroid
                    centroid_x[i] = torch.mean(x_indices, dtype=torch.float32)
                    centroid_y[i] = torch.mean(y_indices, dtype=torch.float32)

                # Plot center of target object in the first batch
                # plt.imshow(target_obj_masks[0].cpu().numpy())
                # plt.plot(centroid_x[0].cpu().numpy(), centroid_y[0].cpu().numpy(), 'ro')
                # plt.show()

                grasp_point = torch.cat((centroid_x.unsqueeze(-1), centroid_y.unsqueeze(-1)), dim=1).type(torch.int64).to(self.device)
                for env, true_env in enumerate(not_RL_envs):
                    # suction_deformation_score, xyz_point, grasp_angle = self.suction_score_object.calculator(
                    #     depth_image, segmask_dexnet, rgb_image_copy, grasp_point, self.object_target_id[not_RL_envs])
                    try:
                        suction_deformation_score, xyz_point, grasp_angle = self.suction_score_object.calculator(
                            depth_image[env], segmask_dexnet[env], rgb_image_copy[env], grasp_point[env], self.object_target_id[not_RL_envs][env])
                        if torch.any(xyz_point == 0):
                            # print("grasp point not found in env", true_env)
                            env_complete_reset = torch.cat((env_complete_reset, torch.tensor([true_env]).to(self.device)), axis=0)
                        else:
                            self.suction_deformation_score[true_env.item()] = suction_deformation_score
                            self.xyz_point[true_env.item()] = xyz_point
                            self.grasp_angle[true_env.item()] = grasp_angle
                            self.env_reset_id_env[true_env.item()] = 0
                            # print("env_reset_id_env", self.env_reset_id_env[true_env.item()], true_env)
                            suction_deformed_envs = torch.where(self.suction_deformation_score[true_env.item()] > 0.0)[0].to(self.device)
                            if (len(suction_deformed_envs) > 0):
                                self.force_SI_env[suction_deformed_envs.item()] = self.force_object.regression(
                                    self.suction_deformation_score[suction_deformed_envs.item()])
                            suction_not_deformed_envs = torch.where(self.suction_deformation_score[true_env.item()] <= 0.0)[0].to(self.device)
                            if (len(suction_not_deformed_envs) > 0):
                                self.force_SI_env[suction_not_deformed_envs.item()] = 0.0
                            # print("RESET ARM POSE 2")
                            env_list_reset_arm_pose = torch.cat((env_list_reset_arm_pose, torch.tensor([true_env]).to(self.device)), axis=0)
                            env_list_reset_objects = torch.cat((env_list_reset_objects, torch.tensor([true_env]).to(self.device)), axis=0)
                    except Exception as e:
                        # print("grasp point error", true_env, e)
                        env_complete_reset = torch.cat((env_complete_reset, torch.tensor([true_env]).to(self.device)), axis=0)
            # reset to prepare for grasp
            self.free_envs_list[store_gp_envs] = 0
            # except Exception as e:
                # print("grasp point error", e)
                # env_complete_reset = torch.cat((env_complete_reset, store_gp_envs), axis=0)

#################################################################################################
        # Actions
        # Moving the arm to pre grasp pose (for RL swiping envs only)
        # print("self.free_envs_list", self.free_envs_list)
        move_to_pre_grasp_envs = torch.where((self.frame_count.type(torch.int64) >= self.cooldown_frames) & (self.free_envs_list.type(torch.int64) == 0) & (self.RL_flag.type(torch.int64) == 1))[0].type(torch.int64).to(self.device).to(self.device)
        
        if len(move_to_pre_grasp_envs) > 0:
            # print("IN move_to_pre_grasp_envs ENVS")

            action_temp = actions.clone().detach().to(self.device)

            u_arm_temp, _ = action_temp[:, :6], action_temp[:, 6]
            u_arm_temp = u_arm_temp * self.cmd_limit / self.action_scale

            self.prim_y = action_temp[:, 7]

            self.prim_z = action_temp[:, 8]

            self.prim_target_dist_x = action_temp[:, 9]

            self.prim_target_dist_y = action_temp[:, 10]

            # import pdb; pdb.set_trace()
            go_to_start_envs = torch.where((self.go_to_start == 1) & (self.frame_count.type(torch.int64) >= self.cooldown_frames) & (self.free_envs_list.type(torch.int64) == 0) & (self.RL_flag.type(torch.int64) == 1))[0].type(torch.int64).to(self.device).to(self.device)
            if len(go_to_start_envs) > 0:
                print("IN GO TO START ENVS", go_to_start_envs)
                return_pre_grasp_envs = torch.where((self.return_pre_grasp == 0) & (self.init_go_to_start == 1) & (self.go_to_start == 1) & (self.frame_count.type(torch.int64) >= self.cooldown_frames) & (self.free_envs_list.type(torch.int64) == 0) & (self.RL_flag.type(torch.int64) == 1))[0].type(torch.int64).to(self.device).to(self.device)
                if len(return_pre_grasp_envs) > 0:
                    self.finished_prim[return_pre_grasp_envs] = 1
                    self.min_distance = torch.ones(self.num_envs).to(self.device)*100
                    self.return_pre_grasp[return_pre_grasp_envs] = 1
                not_return_pre_grasp_envs = torch.where((self.return_pre_grasp == 1) & (self.init_go_to_start == 1) & (self.go_to_start == 1) & (self.frame_count.type(torch.int64) >= self.cooldown_frames) & (self.free_envs_list.type(torch.int64) == 0) & (self.RL_flag.type(torch.int64) == 1))[0].type(torch.int64).to(self.device).to(self.device)
                if len(not_return_pre_grasp_envs) > 0:
                    env_list_reset_arm_pose = torch.cat(
                        (env_list_reset_arm_pose, not_return_pre_grasp_envs), axis=0)
                    self.return_pre_grasp[not_return_pre_grasp_envs] = 0
                    self.init_go_to_start[not_return_pre_grasp_envs] = 0
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
                    [0, 0, 0]).to(self.device), "XYZ", degrees=False).repeat(self.num_envs, 1, 1)
                translation_vector_x =  torch.ones((self.num_envs)).to(self.device) * 0.9
                translation_vector = torch.cat((translation_vector_x.unsqueeze(1), self.prim_target_dist_y.unsqueeze(1), torch.zeros((self.num_envs, 1)).to(self.device)), dim=1)
                T_camera_to_object = transformation_matrices(
                    rotation_matrix_camera_to_object, translation_vector.to(self.device))
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
                    self.curr_poses[:, self.multi_body_idx['ee_link']][:, 3:7])
                translation_ee_pose = self.curr_poses[:, self.multi_body_idx['wrist_3_link']][:, :3]
                T_world_to_ee_pose = transformation_matrices(
                    rotation_matrix_ee_pose, translation_ee_pose)
                T_ee_pose_to_pre_grasp_pose = torch.matmul(
                    torch.inverse(T_world_to_ee_pose), T_world_to_pre_grasp_pose)
                # Orientation error
                action_orientation = matrix_to_euler_angles(
                    T_ee_pose_to_pre_grasp_pose[:, :3, :3], "XYZ")


                pose_factor, ori_factor = 1., 0.3
                # apply actions to all envs that are in go_to_start_envs
                zero_tensor = torch.zeros((self.num_envs, 1)).to(self.device)
                # print("self.actions", self.actions)
                # print("go_to_start_envs", go_to_start_envs)
                # print("self.actions[go_to_start_envs]", self.actions[go_to_start_envs])
                # if len(go_to_start_envs) == 1:
                #     import pdb; pdb.set_trace()
                print("Go to start moving", go_to_start_envs)
                self.actions[go_to_start_envs] = torch.cat((pose_factor * T_ee_pose_to_pre_grasp_pose[:, 0, 3].unsqueeze(-1),
                                                            pose_factor * T_ee_pose_to_pre_grasp_pose[:, 1, 3].unsqueeze(-1),
                                                            pose_factor * T_ee_pose_to_pre_grasp_pose[:, 2, 3].unsqueeze(-1),
                                                            ori_factor * action_orientation[:, 0].unsqueeze(-1),
                                                            ori_factor * action_orientation[:, 1].unsqueeze(-1),
                                                            ori_factor * action_orientation[:, 2].unsqueeze(-1),
                                                            zero_tensor,
                                                            zero_tensor,
                                                            zero_tensor,
                                                            zero_tensor,
                                                            zero_tensor), dim=1)[go_to_start_envs]
                # self.action_env = torch.tensor([[pose_factor*T_ee_pose_to_pre_grasp_pose[0][3], pose_factor*T_ee_pose_to_pre_grasp_pose[1][3],
                #                                 pose_factor *
                #                                 T_ee_pose_to_pre_grasp_pose[2][3], ori_factor *
                #                                 action_orientation[0],
                #                                 ori_factor*action_orientation[1], ori_factor*action_orientation[2], 0, 0, 0, 0, 0]], dtype=torch.float).to(self.device)
                # if ((torch.max(torch.abs(self.action_env[0][:3]))) <= 0.005 and (torch.max(torch.abs(self.action_env[0][3:6]))) <= 0.005):
                #     self.go_to_start[go_to_start_envs] = False
                # print("torch.max(torch.abs(self.actions[go_to_start_envs][:3]))", torch.max(torch.abs(self.actions[:, :3])))
                # print("torch.max(torch.abs(self.actions[go_to_start_envs][3:6]))", torch.max(torch.abs(self.actions[:, 3:6])))
                go_to_start_reached_envs = torch.where((torch.max(torch.abs(self.actions[:, :3]), dim=1)[0] <= 0.005) & (torch.max(torch.abs(self.actions[:, 3:6]), dim=1)[0] <= 0.005) & (self.go_to_start == 1) & (self.frame_count.type(torch.int64) >= self.cooldown_frames) & (self.free_envs_list.type(torch.int64) == 0) & (self.RL_flag.type(torch.int64) == 1))[0].type(torch.int64).to(self.device).to(self.device)
                if len(go_to_start_reached_envs) > 0:
                    self.go_to_start[go_to_start_reached_envs] = False
                    print("go to start reached", go_to_start_reached_envs)
            
            #####PRIM SWIPE##########################################################################################
            not_go_to_start_envs = torch.where((self.go_to_start == 0) & (self.frame_count.type(torch.int64) >= self.cooldown_frames) & (self.free_envs_list.type(torch.int64) == 0) & (self.RL_flag.type(torch.int64) == 1))[0].type(torch.int64).to(self.device).to(self.device)
            if len(not_go_to_start_envs) > 0:
                # lookup table for which direction to swipe
                act = torch.zeros(self.num_envs, 3).to(self.device)
                act[:, 0] = 0.0
                act[:, 1] = 1.0
                act[:, 2] = 3.0

                curr_prim = self.prim[not_go_to_start_envs]
                curr_prim_1_envs = torch.where(curr_prim == 1)[0].to(self.device)
                if len(curr_prim_1_envs) > 0:
                    self.true_target_dist[curr_prim_1_envs] = self.prim_target_dist_y[curr_prim_1_envs]
                curr_prim_not_1_envs = torch.where(curr_prim != 1)[0].to(self.device)
                if len(curr_prim_not_1_envs) > 0:
                    self.true_target_dist[curr_prim_not_1_envs] = self.prim_target_dist_x[curr_prim_not_1_envs]

                true_target_dist_neg_envs = torch.where((self.true_target_dist < 0) & (self.go_to_start == 0))[0].to(self.device)
                if len(true_target_dist_neg_envs) > 0:
                    self.true_target_dist[true_target_dist_neg_envs] = abs(self.true_target_dist[true_target_dist_neg_envs])
                    # import pdb; pdb.set_trace()
                    act[true_target_dist_neg_envs, self.prim[true_target_dist_neg_envs].type(torch.long)] += 3.0

                self.target_dist[not_go_to_start_envs, 0, 0] = self.true_target_dist[not_go_to_start_envs]
                self.target_dist[not_go_to_start_envs, 1, 1] = -self.true_target_dist[not_go_to_start_envs]
                self.target_dist[not_go_to_start_envs, 2, 2] = self.true_target_dist[not_go_to_start_envs]
                self.target_dist[not_go_to_start_envs, 3, 0] = -self.true_target_dist[not_go_to_start_envs]
                self.target_dist[not_go_to_start_envs, 4, 1] = self.true_target_dist[not_go_to_start_envs]
                self.target_dist[not_go_to_start_envs, 5, 2] = -self.true_target_dist[not_go_to_start_envs]
                action_str = np.empty((len(not_go_to_start_envs), 1), dtype="U10")
                temp_actions = self.action[not_go_to_start_envs.cpu(), act[not_go_to_start_envs, curr_prim.type(torch.long)].type(torch.long).cpu()]
                if type(temp_actions) == np.str_:
                    temp_actions = np.array([temp_actions], dtype="U10")
                    temp_actions = temp_actions.reshape(-1, 1)
                    action_str = temp_actions
                else:
                    action_str = np.expand_dims(temp_actions, axis=1)
                curr_eef_pos = self.states["eef_pos"][not_go_to_start_envs]
                curr_eef_quat = self.states["eef_quat"][not_go_to_start_envs]
                move_dist = self.target_dist[not_go_to_start_envs, act[not_go_to_start_envs, self.prim[not_go_to_start_envs].type(torch.long)].type(torch.long)]
                for i in range(len(not_go_to_start_envs)):
                    u_arm_temp[i, 0:6], res = self.primitives[not_go_to_start_envs[i]].move_w_ori(action_str[i].item(), curr_eef_pos[i].unsqueeze(0), curr_eef_quat[i].unsqueeze(0), move_dist[i])
                    action_str[i] = np.array([[res]], dtype="U10")
                done_single_action_envs = torch.from_numpy(np.where(action_str == "done")[0]).to(self.device)
                # make sure that done_single_action_envs must be in not_go_to_start_envs as well
                done_single_action_envs = not_go_to_start_envs[torch.where(torch.isin(not_go_to_start_envs, done_single_action_envs))[0].to(self.device)]
                if len(done_single_action_envs) > 0:
                    self.prim[done_single_action_envs] += 1
                    # if curr_prim is 2, reset to 0
                    curr_prim_2_envs = torch.where(self.prim == 2)[0].to(self.device)
                    curr_prim_2_envs = done_single_action_envs[torch.where(torch.isin(done_single_action_envs, curr_prim_2_envs))[0].to(self.device)]
                    if len(curr_prim_2_envs) > 0:
                        self.prim[curr_prim_2_envs] = 0
                        env_list_reset_default_arm_pose = torch.cat((env_list_reset_default_arm_pose, done_single_action_envs), axis=0)
                        self.frame_count[curr_prim_2_envs] = 0
                        self.progress_buf[curr_prim_2_envs] = 0
                        # print("reset arm pose")
                        # RESET ARM
                        self.primitive_count[curr_prim_2_envs] += 1
                        prim_count_greater_envs = torch.where(self.primitive_count >= self.num_primitive_actions + 1)[0].to(self.device)
                        prim_count_greater_envs = curr_prim_2_envs[torch.where(torch.isin(curr_prim_2_envs, prim_count_greater_envs))[0].to(self.device)]
                        if len(prim_count_greater_envs) > 0:
                            self.RL_flag[prim_count_greater_envs] = 0
                            self.free_envs_list[prim_count_greater_envs] = 1
                            # print("RL flag set to 0", prim_count_greater_envs)
                        prim_count_less_envs = torch.where(self.primitive_count < self.num_primitive_actions + 1)[0].to(self.device)
                        prim_count_less_envs = curr_prim_2_envs[torch.where(torch.isin(curr_prim_2_envs, prim_count_less_envs))[0].to(self.device)]
                        if len(prim_count_less_envs) > 0:
                            self.init_go_to_start[prim_count_less_envs] = True
                            self.go_to_start[prim_count_less_envs] = True
                
                self.actions[not_go_to_start_envs] = torch.cat((u_arm_temp[not_go_to_start_envs, 0:6], torch.zeros((len(not_go_to_start_envs), 5)).to(self.device)), dim=1)
                # force sensor update
                self.gym.refresh_force_sensor_tensor(self.sim)
                _fsdata = self.gym.acquire_force_sensor_tensor(self.sim)
                fsdata = gymtorch.wrap_tensor(_fsdata)
                self.force_pre_physics[not_go_to_start_envs] = - \
                    fsdata[not_go_to_start_envs][:, 2].detach()
                for i in not_go_to_start_envs:
                    retKey = self.force_list_save.get(i)
                    if (retKey == None):
                        self.force_list_save[i] = torch.tensor(
                            [self.force_pre_physics[i]])
                    else:
                        force_list_env = self.force_list_save[i]
                        force_list_env = torch.cat(
                            (force_list_env, torch.tensor([self.force_pre_physics[i]])))
                        self.force_list_save[i] = force_list_env

        grasping_envs = torch.where((self.frame_count.type(torch.int64) >= self.cooldown_frames) & (self.free_envs_list.type(torch.int64) == 0) & (self.RL_flag.type(torch.int64) == 0))[0].type(torch.int64).to(self.device).to(self.device)
        if len(grasping_envs) > 0:
            # print("IN GRASPING ENVS", grasping_envs)
            action_contrib_less_than_1_envs = torch.where((self.action_contrib <= 1.0) & (self.frame_count.type(torch.int64) >= self.cooldown_frames) & (self.free_envs_list.type(torch.int64) == 0) & (self.RL_flag.type(torch.int64) == 0))[0].type(torch.int64).to(self.device).to(self.device)
            if len(action_contrib_less_than_1_envs) > 0:
                # force sensor update
                self.gym.refresh_force_sensor_tensor(self.sim)
                _fsdata = self.gym.acquire_force_sensor_tensor(self.sim)
                fsdata = gymtorch.wrap_tensor(_fsdata)
                self.force_pre_physics[action_contrib_less_than_1_envs] = - \
                    fsdata[action_contrib_less_than_1_envs][:, 2].detach()
                for i in action_contrib_less_than_1_envs:
                    retKey = self.force_list_save.get(i)
                    if (retKey == None):
                        self.force_list_save[i] = torch.tensor(
                            [self.force_pre_physics[i]])
                    else:
                        force_list_env = self.force_list_save[i]
                        force_list_env = torch.cat(
                            (force_list_env, torch.tensor([self.force_pre_physics[i]])))
                        self.force_list_save[i] = force_list_env
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
                [0, 0, 0]).to(self.device), "XYZ", degrees=False).repeat(self.num_envs, 1, 1)
            T_camera_to_object = transformation_matrices(
                        rotation_matrix_camera_to_object, self.xyz_point)
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
                self.curr_poses[:, self.multi_body_idx['ee_link']][:, 3:7])
            translation_ee_pose = self.curr_poses[:, self.multi_body_idx['wrist_3_link']][:, :3]
            T_world_to_ee_pose = transformation_matrices(
                rotation_matrix_ee_pose, translation_ee_pose)
            T_ee_pose_to_pre_grasp_pose = torch.matmul(
                torch.inverse(T_world_to_ee_pose), T_world_to_pre_grasp_pose)
            # Orientation error
            action_orientation = matrix_to_euler_angles(
                T_ee_pose_to_pre_grasp_pose[:, :3, :3], "XYZ")
            
            action_contrib_greater_envs = torch.where((self.action_contrib >= torch.tensor(1)) & (self.frame_count.type(torch.int64) >= self.cooldown_frames) & (self.free_envs_list.type(torch.int64) == 0) & (self.RL_flag.type(torch.int64) == 0))[0].type(torch.int64).to(self.device).to(self.device)
            if len(action_contrib_greater_envs) > 0:
                pose_factor, ori_factor = 2.0, 1.0
                zero_tensor = torch.zeros((self.num_envs, 1)).to(self.device)
                # print("ACTION CONTRIBUTION GREATER THAN 1", action_contrib_greater_envs)
                self.actions[action_contrib_greater_envs] = torch.cat((pose_factor * T_ee_pose_to_pre_grasp_pose[:, 0, 3].unsqueeze(-1),
                                                                            pose_factor * T_ee_pose_to_pre_grasp_pose[:, 1, 3].unsqueeze(-1),
                                                                            pose_factor * T_ee_pose_to_pre_grasp_pose[:, 2, 3].unsqueeze(-1),
                                                                            ori_factor * action_orientation[:, 0].unsqueeze(-1),
                                                                            ori_factor * action_orientation[:, 1].unsqueeze(-1),
                                                                            ori_factor * action_orientation[:, 2].unsqueeze(-1),
                                                                            zero_tensor,
                                                                            zero_tensor,
                                                                            zero_tensor,
                                                                            zero_tensor,
                                                                            zero_tensor), dim=1)[action_contrib_greater_envs]
            action_contrib_less_envs = torch.where((self.action_contrib < torch.tensor(1)) & (self.frame_count.type(torch.int64) >= self.cooldown_frames) & (self.free_envs_list.type(torch.int64) == 0) & (self.RL_flag.type(torch.int64) == 0))[0].type(torch.int64).to(self.device).to(self.device)
            if len(action_contrib_less_envs) > 0:
                # print("action_contrib_less_envs", action_contrib_less_envs)
                # Transformation for grasp pose (wg --> wo*og)
                rotation_matrix_grasp_pose = euler_angles_to_matrix(torch.tensor(
                    [0, 0, 0]).to(self.device), "XYZ", degrees=False).repeat(self.num_envs, 1, 1)
                translation_grasp_pose = torch.tensor(
                    [self.speed[0], 0, 0]).to(self.device).type(torch.float)
                translation_grasp_pose = torch.matmul(
                    rotation_matrix_grasp_pose, translation_grasp_pose).repeat(self.num_envs, 1)

                start_point = T_world_to_pre_grasp_pose[:, :3, 3]
                end_point = T_world_to_object[:, :3, 3]
                current_point = T_world_to_ee_pose[:, :3, 3]
                v = end_point - start_point
                # Calculate the vector connecting p1 to p
                w = current_point - start_point
                # Calculate the projection of w onto v
                t = torch.sum(w*v, dim=1) / torch.sum(v*v, dim=1)
                # Calculate the closest point on the line to p
                # q = torch.tensor([start_point[0] + t*v[0], start_point[1] +
                #                 t*v[1], start_point[2] + t*v[2]]).to(self.device)
                q = start_point + t.unsqueeze(-1)*v
                # Find the distance between p and q
                self.distance = current_point - q
                zero_tensor = torch.zeros((len(action_contrib_less_envs), 1)).to(self.device)
                self.actions[action_contrib_less_envs] = torch.cat((self.speed[0] * torch.ones((len(action_contrib_less_envs), 1)).to(self.device),
                                                                    (translation_grasp_pose[action_contrib_less_envs, 1] - self.distance[action_contrib_less_envs, 1] * 100 * self.speed[0]).unsqueeze(-1),
                                                                    (translation_grasp_pose[action_contrib_less_envs, 2] - self.distance[action_contrib_less_envs, 2] * 100 * self.speed[0]).unsqueeze(-1),
                                                                    self.speed[0] * 100 * action_orientation[action_contrib_less_envs, 0].unsqueeze(-1),
                                                                    self.speed[0] * 100 * action_orientation[action_contrib_less_envs, 1].unsqueeze(-1),
                                                                    self.speed[0] * 100 * action_orientation[action_contrib_less_envs, 2].unsqueeze(-1),
                                                                    zero_tensor,
                                                                    zero_tensor,
                                                                    zero_tensor,
                                                                    zero_tensor,
                                                                    zero_tensor), dim=1)

            is_grasping_env = torch.where((self.frame_count.type(torch.int64) >= self.cooldown_frames) & (self.frame_count_contact_object == 0))[0].to(self.device)
            if len(is_grasping_env) > 0:
                set_lower_contrib_env = torch.where((torch.max(torch.abs(self.actions[:, :3]), dim=1)[0] <= 0.005) & 
                                                    (torch.max(torch.abs(self.actions[:, 3:6]), dim=1)[0] <= 0.005) & 
                                                    (self.frame_count.type(torch.int64) >= self.cooldown_frames) & 
                                                    (self.frame_count_contact_object == 0) & (self.env_reset_id_env == 0))[0].to(self.device)
                if len(set_lower_contrib_env) > 0:
                    # print("set_lower_contrib_env", set_lower_contrib_env)
                    self.action_contrib[set_lower_contrib_env] -= 1
                    # print("self.action_contrib", self.action_contrib)
                    eef_rgb_tensor = torch.stack(self.eef_rgb_camera_tensors)
                    eef_rgb_image = eef_rgb_tensor[set_lower_contrib_env].to(self.device)
                    eef_rgb_image_copy = torch.reshape(
                        eef_rgb_image, (eef_rgb_image.shape[0], eef_rgb_image.shape[1], -1, 4))[..., :3].clone().detach()
                    
                    eef_mask_tensors = torch.stack(self.eef_mask_camera_tensors)
                    eef_segmask = eef_mask_tensors[set_lower_contrib_env].to(self.device).clone().detach()
                    
                    eef_depth_tensor = torch.stack(self.eef_depth_camera_tensors)
                    eef_depth_image = eef_depth_tensor[set_lower_contrib_env].to(self.device).clone().detach()
                    eef_depth_image = -eef_depth_image

                    for i, true_env in enumerate(set_lower_contrib_env):
                        self.suction_deformation_score[true_env], temp_xyz_point, temp_grasp = self.suction_score_object_gripper.calculator(
                            eef_depth_image[i], eef_segmask[i], eef_rgb_image_copy[i], None, self.object_target_id[true_env])
                        self.suction_score_store_env[true_env] = self.suction_deformation_score[true_env]
                        if (self.suction_deformation_score[true_env] > self.force_threshold):
                            self.force_SI[true_env] = self.force_object.regression(
                                self.suction_deformation_score[true_env])
                        else:
                            self.force_SI[true_env] = torch.tensor(
                                1000).to(self.device)
                        if (self.action_contrib[true_env] == 1):
                            temp_grasp = torch.tensor([0, 0, 0])
                            self.xyz_point[true_env][0] += temp_xyz_point[0]
                            self.grasp_angle[true_env] = temp_grasp
                # print("self.force_pre_physics", self.force_pre_physics)
                # print("self.force_threshold", self.force_threshold)
                # print("self.force_SI", self.force_SI)
                # print("self.action_contrib", self.action_contrib)
                # print("self.force_encounter", self.force_encounter)
                # print("self.frame_count", self.frame_count)
                # print("self.frame_count_contact_object", self.frame_count_contact_object)
                check_collision_envs = torch.where((self.force_pre_physics > 10) & (self.action_contrib == 1) & (self.frame_count.type(torch.int64) >= self.cooldown_frames) & (self.frame_count_contact_object == 0))[0].to(self.device)
                if len(check_collision_envs) > 0:
                    # print("Force collision detected", check_collision_envs)
                    env_complete_reset = torch.cat((env_complete_reset, check_collision_envs), axis=0)
                    env_list_reset_objects = torch.cat((env_list_reset_objects, check_collision_envs), axis=0)
                check_suction_success_envs = torch.where((self.force_pre_physics > self.force_threshold) & (self.force_pre_physics > self.force_SI) &
                                                         (self.action_contrib == 0) & (self.force_encounter == 0) &
                                                         (self.frame_count.type(torch.int64) >= self.cooldown_frames) & (self.frame_count_contact_object == 0))[0].to(self.device)
                if len(check_suction_success_envs) > 0:
                    # print("check_suction_success_envs", check_suction_success_envs)
                    self.force_encounter[check_suction_success_envs] = 1
                    eef_rgb_tensor = torch.stack(self.eef_rgb_camera_tensors)
                    eef_rgb_image = eef_rgb_tensor[check_suction_success_envs].to(self.device)
                    eef_rgb_image_copy = torch.reshape(
                        eef_rgb_image, (eef_rgb_image.shape[0], eef_rgb_image.shape[1], -1, 4))[..., :3].clone().detach()
                    
                    eef_mask_tensors = torch.stack(self.eef_mask_camera_tensors)
                    eef_segmask = eef_mask_tensors[check_suction_success_envs].to(self.device).clone().detach()
                    
                    eef_depth_tensor = torch.stack(self.eef_depth_camera_tensors)
                    eef_depth_image = eef_depth_tensor[check_suction_success_envs].to(self.device).clone().detach()
                    eef_depth_image = -eef_depth_image
                    for i, true_env in enumerate(check_suction_success_envs):
                        object_pose_at_contact = self._root_state[true_env, self._object_model_id[int(self.object_target_id[true_env].item())-1], :][:7].type(
                                torch.float).clone().detach()
                        self.offset_object_pose_retract[true_env] = object_pose_at_contact[:3] - \
                            T_world_to_ee_pose[true_env, :3, 3] + \
                            torch.tensor([0.005, 0, 0]).to(self.device)
                        self.retract_start_pose[true_env] = T_world_to_ee_pose[true_env, :3, 3]
                        score_gripper, _, _ = self.suction_score_object_gripper.calculator(
                            eef_depth_image[i], eef_segmask[i], eef_rgb_image_copy[i], None, self.object_target_id[true_env])
                        # print(true_env, " force: ", self.force_pre_physics[true_env].item(), " score: ", score_gripper.item())

                        if not self.retract:
                            self.frame_count_contact_object[true_env] = 1

                        self.success[true_env] = False
                        if (score_gripper > torch.tensor(0.1).to(self.device)):
                            ## Patrick: only use 0/0/1 sparse reward for now instead of 0/1/2
                            self.success[true_env] = 1
                if self.retract:
                    retract_envs = torch.where((self.force_encounter == 1) & (self.frame_count.type(torch.int64) >= self.cooldown_frames) & (self.frame_count_contact_object == 0))[0].to(self.device)
                    # print("RETRACT ENVS", retract_envs)
                    if len(retract_envs) > 0:
                        # Transformation for grasp pose (wg --> wo*og)
                        rotation_matrix_grasp_pose = euler_angles_to_matrix(torch.tensor(
                            [0, 0, 0]).to(self.device), "XYZ", degrees=False).repeat(self.num_envs, 1, 1)
                        translation_grasp_pose = torch.tensor(
                        [self.speed[0], 0, 0]).to(self.device).type(torch.float)
                        translation_grasp_pose = torch.matmul(
                            rotation_matrix_grasp_pose, translation_grasp_pose).repeat(self.num_envs, 1)

                        retract_up_envs = torch.where((self.retract_up == 1) & (self.force_encounter == 1) & (self.frame_count.type(torch.int64) >= self.cooldown_frames) & (self.frame_count_contact_object == 0))[0].to(self.device)
                        if len(retract_up_envs) > 0:
                            end_point = T_world_to_pre_grasp_pose[:, :3, 3].clone(
                            ).detach()
                            end_point[:, 2] += 0.03
                            # end_point[0] -= 0.02
                        not_retract_up_envs = torch.where((self.retract_up == 0) & (self.force_encounter == 1) & (self.frame_count.type(torch.int64) >= self.cooldown_frames) & (self.frame_count_contact_object == 0))[0].to(self.device)
                        if len(not_retract_up_envs) > 0:
                            end_point = self.retract_start_pose.clone().detach()
                            end_point[:, 2] += 0.03

                        start_point = self.retract_start_pose
                        current_point = T_world_to_ee_pose[:, :3, 3]
                        v = end_point - start_point
                        # Calculate the vector connecting p1 to p
                        w = current_point - start_point
                        # Calculate the projection of w onto v
                        t = torch.sum(w*v, dim=1) / torch.sum(v*v, dim=1)
                        # Calculate the closest point on the line to p
                        # q = torch.tensor([start_point[0] + t*v[0], start_point[1] +
                        #                 t*v[1], start_point[2] + t*v[2]]).to(self.device)
                        q = start_point + t.unsqueeze(-1)*v
                        # Find the distance between p and q
                        self.distance = current_point - q
                        zero_tensor = torch.zeros((len(not_retract_up_envs), 1)).to(self.device)
                        if len(not_retract_up_envs) > 0:
                            # print("NOT RETRACT UP ENVS", not_retract_up_envs)

                            self.actions[not_retract_up_envs] = torch.cat((-0.001 * torch.ones((len(not_retract_up_envs), 1)).to(self.device),
                                                                                (translation_grasp_pose[not_retract_up_envs, 1] - self.distance[not_retract_up_envs, 1] * self.speed[0]).unsqueeze(-1),
                                                                                self.speed[0].repeat(len(not_retract_up_envs), 1),
                                                                                self.speed[0].unsqueeze(-1) * 50 * action_orientation[not_retract_up_envs, 0].unsqueeze(-1),
                                                                                self.speed[0].unsqueeze(-1) * 50 * action_orientation[not_retract_up_envs, 1].unsqueeze(-1),
                                                                                self.speed[0].unsqueeze(-1) * 50 * action_orientation[not_retract_up_envs, 2].unsqueeze(-1),
                                                                                zero_tensor,
                                                                                zero_tensor,
                                                                                zero_tensor,
                                                                                zero_tensor,
                                                                                zero_tensor), dim=1)
                        zero_tensor = torch.zeros((len(retract_up_envs), 1)).to(self.device)
                        if len(retract_up_envs) > 0:
                            # print("RETRACT UP ENVS", retract_up_envs)
                            self.actions[retract_up_envs] = torch.cat((-self.speed[0].repeat(len(retract_up_envs), 1),
                                                                            (translation_grasp_pose[retract_up_envs, 1] - self.distance[retract_up_envs, 1] * 50 * self.speed[0]).unsqueeze(-1),
                                                                            (translation_grasp_pose[retract_up_envs, 2] - self.distance[retract_up_envs, 2] * 50 * self.speed[0]).unsqueeze(-1),
                                                                            self.speed[0].unsqueeze(-1) * 50 * action_orientation[retract_up_envs, 0].unsqueeze(-1),
                                                                            self.speed[0].unsqueeze(-1) * 50 * action_orientation[retract_up_envs, 1].unsqueeze(-1),
                                                                            self.speed[0].unsqueeze(-1) * 50 * action_orientation[retract_up_envs, 2].unsqueeze(-1),
                                                                            zero_tensor,
                                                                            zero_tensor,
                                                                            zero_tensor,
                                                                            zero_tensor,
                                                                            zero_tensor), dim=1)

                        self.env_reset_retract = torch.cat(
                            (self.env_reset_retract, retract_envs), axis=0).type(torch.long)
                
                        # env_temp_id = torch.tensor([]).to(self.device)

                        # env_temp_id = torch.cat(
                        #     (env_temp_id, torch.tensor([env_count]).to(self.device)), axis=0).type(torch.long)
                        # self.retract_object_state[env_count] = torch.zeros(
                        #     len(env_temp_id), 13, device=self.device)
                        # self.retract_object_state[env_count][:,
                        #                                     3:7] = self._init_object_model_state[self.object_target_id[env_count]-1][env_count][3:7]

                        # self.retract_object_state[env_count][:, :3] = T_world_to_ee_pose[:3,
                        #                                                                 3] + self.offset_object_pose_retract[env_count]
                        # this_object_state_all = self._init_object_model_state[
                        #     self.object_target_id[env_count]-1]

                        # # Lastly, set these sampled values as the new init state
                        # this_object_state_all[env_count,
                        #                     :] = self.retract_object_state[env_count]
                        # self._object_model_state[self.object_target_id[env_count] -
                        #                         1][env_count] = self._init_object_model_state[self.object_target_id[env_count]-1][env_count]

                        # angle_error = quaternion_to_euler_angles(self._eef_state[env_count][3:7], "XYZ", degrees=False) - torch.tensor(
                        #     [0, -self.grasp_angle[env_count][1], self.grasp_angle[env_count][0]]).to(self.device)

                        # current_target_object_pose = self._root_state[env_count, self._object_model_id[int(self.object_target_id[env_count].item())-1], :][:3].type(
                        #     torch.float).clone().detach() - self.offset_object_pose_retract[env_count]

                        switch_retract_up_envs = torch.where((self.retract_up == 0) & (current_point[:, 2] >= end_point[:, 2]) & (self.force_encounter == 1) & (self.frame_count.type(torch.int64) >= self.cooldown_frames) & (self.frame_count_contact_object == 0))[0].to(self.device)
                        if len(switch_retract_up_envs) > 0:
                            self.retract_up[switch_retract_up_envs] = 1
                            self.retract_start_pose[switch_retract_up_envs] = T_world_to_ee_pose[switch_retract_up_envs, :3, 3]
                        # ### HOW MUCH TO RETRACT
                        stop_retract_envs = torch.where((self.retract_up == 1) & (current_point[:, 0] <= 0.1) & (self.force_encounter == 1) & (self.frame_count.type(torch.int64) >= self.cooldown_frames) & (self.frame_count_contact_object == 0))[0].to(self.device)
                        if len(stop_retract_envs) > 0:
                            self.frame_count_contact_object[stop_retract_envs] = 1
        reset_envs = torch.where((self.frame_count.type(torch.int64) >= self.cooldown_frames) & (self.frame_count_contact_object == 1))[0].to(self.device)
        if len(reset_envs) > 0:
            # print("RESET ARM POSE 3", reset_envs)
            env_complete_reset = torch.cat((env_complete_reset, reset_envs), axis=0)
            env_list_reset_default_arm_pose = torch.cat((env_list_reset_default_arm_pose, reset_envs), axis=0)



                    

############RESETS##########################################################################################
        if (len(env_complete_reset) != 0):
            # print("env_complete_reset: ", env_complete_reset)
            env_complete_reset = torch.unique(env_complete_reset)
            env_list_reset_objects = torch.cat(
                (env_list_reset_objects, env_complete_reset), axis=0)
            pos = self.reset_init_arm_pose(
                env_complete_reset.to(self.device).type(torch.long))
            env_ids = env_complete_reset.to(self.device).type(torch.long)
            self.deploy_actions(env_ids, pos)

        if len(env_list_reset_arm_pose) > 0:
            # print("env_list_reset_arm_pose", env_list_reset_arm_pose)
            env_list_reset_arm_pose = torch.unique(env_list_reset_arm_pose)
            pos = self.reset_pre_grasp_pose(env_list_reset_arm_pose.to(self.device).type(torch.long))
            env_ids = env_list_reset_arm_pose.to(self.device).type(torch.long)
            self.deploy_actions(env_ids, pos)
        if (len(env_list_reset_objects) != 0 or len(self.env_reset_retract) != 0):
            # print("env_list_reset_objects: ", env_list_reset_objects)
            if (len(self.env_reset_retract) != 0 and len(env_list_reset_objects) != 0):
                self.env_reset_retract = self.env_reset_retract.to(
                    self.device).type(torch.long)
                env_list_reset_objects = torch.unique(env_list_reset_objects)
                self.reset_object_pose(env_list_reset_objects.to(
                    self.device).type(torch.long), reset_retract_object=True)
            elif (len(self.env_reset_retract) == 0 and len(env_list_reset_objects) != 0):
                env_list_reset_objects = torch.unique(env_list_reset_objects)
                self.reset_object_pose(env_list_reset_objects.to(
                    self.device).type(torch.long), reset_retract_object=False)
            elif (len(self.env_reset_retract) != 0 and len(env_list_reset_objects) == 0):
                self.env_reset_retract = self.env_reset_retract.to(
                    self.device).type(torch.long)
                self.reset_object_pose(None, reset_retract_object=True)
        if (len(env_list_reset_default_arm_pose) != 0):
            # print("env_list_reset_default_arm_pose: ",  env_list_reset_default_arm_pose)
            env_list_reset_default_arm_pose = torch.unique(
                env_list_reset_default_arm_pose)
            pos = self.reset_default_arm_pose(env_list_reset_default_arm_pose.to(
                self.device).type(torch.long))
            env_ids = env_list_reset_default_arm_pose.to(
                self.device).type(torch.long)
            self.deploy_actions(env_ids, pos)
            

        self.frame_count += 1

        # get indicies where self.reset_env is true
        # reset_buf_indicies = torch.where(self.reset_buf == 1)[0].to(self.device)
        # if reset_buf_indicies.shape[0] > 0:
        #     print("Timeout reset env: ", reset_buf_indicies)
        # env_complete_reset = torch.cat((reset_buf_indicies, env_complete_reset), axis=0)

        
        # Split arm and gripper command
        u_arm, u_gripper = self.actions[:, :6], self.actions[:, 6]

        # Control arm (scale value first)
        u_arm = u_arm * self.cmd_limit / self.action_scale
        if self.control_type == "osc":
            u_temp = self._compute_osc_torques(dpose=u_arm)
            u_arm = torch.clip(u_temp, min=-10, max=10)
        self._arm_control[:, :6] = u_arm

        # Deploy actions
        self.gym.set_dof_position_target_tensor(
            self.sim, gymtorch.unwrap_tensor(self._pos_control))
        self.gym.set_dof_actuation_force_tensor(
            self.sim, gymtorch.unwrap_tensor(self._effort_control))

    def post_physics_step(self):
        # Update state
        self.progress_buf += 1
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)

        # check if there is atimeout
        # if len(env_ids) > 0:
        #     for env_id in env_ids:
        #         env_count = env_id.item()
        #         if (self.force_list_save[env_count] != None and len(self.force_list_save[env_count]) > 10):
        #             oscillation = self.detect_oscillation(
        #                 self.force_list_save[env_count])
        #             self.success[env_count] = False
        #             json_save = {
        #                 "force_array": self.force_list_save[env_count].tolist(),
        #                 "grasp point": self.grasp_point[env_count].tolist(),
        #                 "grasp_angle": self.grasp_angle[env_count].tolist(),
        #                 "suction_deformation_score": self.suction_deformation_score[env_count].item(),
        #                 "oscillation": oscillation,
        #                 "gripper_score": 0,
        #                 "success": self.success[env_count].item(),
        #                 "object_id": self.object_target_id[env_count].item(),
        #                 "penetration": False,
        #                 "unreachable": False
        #             }
        #             print("success: ", self.success[env_count].item())
        #             new_dir_name = str(
        #                 env_count)+"_"+str(self.track_save[env_count].type(torch.int).item())
        #             self.track_save[env_count] = self.track_save[env_count] + \
        #                 torch.tensor(1).to(self.device)
        #     print(f"timeout reset for environment {env_ids}")
           
        self.compute_observations()
        self.compute_reward()
        self.finished_prim = torch.zeros(self.num_envs).to(self.device)
     
        # Compute resets
        self.reset_buf = torch.where(
            (self.progress_buf >= self.max_episode_length - 1), torch.ones_like(self.reset_buf), self.reset_buf)

    def quaternion_conj(self, q):
        w, x, y, z = q
        return torch.tensor([w, -x, -y, -z])

    def quaternion_mult(self, q1, q2):
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
        z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2
        return torch.tensor([w, x, y, z])
