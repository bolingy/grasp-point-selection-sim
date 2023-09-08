import random
import numpy as np
import os
import torch
import yaml
import json
import sys

from isaacgym import gymtorch
from isaacgym import gymapi

from isaacgym.torch_utils import *
from isaacgymenvs.utils.torch_jit_utils import *
from isaacgymenvs.tasks.base.vec_task import VecTask

# For camera module
from PIL import Image as im
from PIL import Image

import matplotlib.pyplot as plt

from isaacgym import gymutil

import math
import cv2

from suction_cup_modelling.suction_score_calcualtor import calcualte_suction_score
from suction_cup_modelling.force_calculator import calcualte_force

from gqcnn_examples.policy_for_training import dexnet3
from autolab_core import (YamlConfig, Logger, BinaryImage,
                          CameraIntrinsics, ColorImage, DepthImage, RgbdImage)

# import assets.urdf_models.models_data as md
from homogeneous_trasnformation_and_conversion.rotation_conversions import *

import time
import pandas as pd

from pathlib import Path
import glob


class UR16eManipulation(VecTask):

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render, bin_id, data_path=None, google_scanned_objects_path=None):
        self.cfg = cfg

        self.max_episode_length = self.cfg["env"]["episodeLength"]

        self.action_scale = self.cfg["env"]["actionScale"]
        self.aggregate_mode = self.cfg["env"]["aggregateMode"]

        # Bin ID (bin size)
        self.bin_id = bin_id

        self.google_scanned_objects_path = google_scanned_objects_path

        # Controller type
        self.control_type = self.cfg["env"]["controlType"]
        assert self.control_type in {"osc", "joint_tor"},\
            "Invalid control type specified. Must be one of: {osc, joint_tor}"

        # dimensions
        # obs include: cubeA_pose (7) + cubeB_pos (3) + eef_pose (7) + q_gripper (2)
        self.cfg["env"]["numObservations"] = 9 if self.control_type == "osc" else 28
        # actions include: delta EEF if OSC (6) or joint torques (7) + bool gripper (1)
        self.cfg["env"]["numActions"] = 6 if self.control_type == "osc" else 7

        # Values to be filled in at runtime
        # will be dict filled with relevant states to use for reward calculation
        self.states = {}
        # will be dict mapping names to relevant sim handles
        self.handles = {}
        self.num_dofs = None                    # Total number of DOFs per env
        self.actions = None                     # Current actions to be deployed

        # self.models_lib = md.model_lib()

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

        self.debug_viz = self.cfg["env"]["enableDebugVis"]

        self.up_axis = "z"
        self.up_axis_idx = 2

        self.ur16e_hand = "ee_link"
        self.ur16e_wrist_3_link = "wrist_3_link"
        self.ur16e_base = "base_link"
        self.suction_gripper = "epick_end_effector"

        self.force_threshold = 0.1
        self.object_coordiante_camera = torch.tensor([0, 0, 0])

        # Parallelization
        self.init_camera_capture = 1

        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../configs")+'/collision_primitives_'+str(self.bin_id)+'.yml') as file:
            self.world_params = yaml.load(file, Loader=yaml.FullLoader)

        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id,
                         headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)

        self.ur16e_default_dof_pos = to_torch(
            [-1.57, 0, 0, 0, 0, 0, 0], device=self.device
        )

        self.cooldown_frames = 150

        # System IDentification data results
        self.data_path = data_path or os.path.expanduser(
            "/tmp/grasp_data/")
        # for env_number in range(self.num_envs):
        #     new_dir_path = os.path.join(
        #         self.data_path, f"{self.bin_id}/{env_number}/")
        #     os.makedirs(new_dir_path, exist_ok=True)

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
        self.action_contrib = torch.ones(self.num_envs)*2
        self.frame_count_contact_object = torch.zeros(self.num_envs)
        self.force_encounter = torch.zeros(self.num_envs)
        self.frame_count = torch.zeros(self.num_envs)
        self.free_envs_list = torch.ones(self.num_envs)
        self.object_pose_check_list = torch.ones(self.num_envs)
        self.object_target_id = torch.zeros(self.num_envs).type(torch.int)
        self.speed = torch.ones(self.num_envs)*0.1
        print("No. of environments: ", self.num_envs)

        # Parameter storage and Trackers for each environments
        self.suction_deformation_score_temp = torch.Tensor()
        self.xyz_point_temp = torch.Tensor([])
        self.grasp_angle_temp = torch.Tensor([])
        self.force_SI_temp = torch.Tensor()
        self.grasp_point_temp = torch.Tensor([])
        self.dexnet_score_temp = torch.Tensor([])

        self.suction_deformation_score_env = {}
        self.xyz_point_env = {}
        self.grasp_angle_env = {}
        self.force_SI_env = {}
        self.grasp_point_env = {}
        self.dexnet_score_env = {}
        self.retract_flag_env = {}
        self.force_contact_flag = torch.zeros(self.num_envs)

        self.suction_deformation_score = {}
        self.xyz_point = {}
        self.grasp_angle = {}
        self.force_SI = {}
        self.grasp_point = {}
        self.dexnet_score = {}
        self.last_object_pose = {}
        self.all_objects_last_pose = {}
        self.object_pose_store = {}
        self.target_object_disp_save = {}
        self.object_disp_save = {}
        self.offset_object_pose_retract = {}
        self.retract_start_pose = {}
        self.oscillation_store_env = {}
        self.suction_score_store_env = {}
        self.retract_up = torch.zeros(self.num_envs)

        self.selected_object_env = {}
        self.retract_object_state = {}

        self.track_save = torch.tensor(0)
        # self.config_env_count = torch.zeros(self.num_envs)
        self.force_list_save = {}
        self.depth_image_save = {}
        self.segmask_save = {}
        self.rgb_save = {}
        self.grasp_point_save = {}
        self.grasp_angle_save = {}
        self.dexnet_score_save = {}
        self.suction_deformation_score_save = {}
        self.force_require_SI = {}
        self.count_step_suction_score_calculator = torch.zeros(self.num_envs)
        self.num_grasps_per_sim = 0

        self.env_reset_id_env = torch.ones(self.num_envs)

        self.grasps_done_env = torch.zeros(self.num_envs)
        self.env_done_grasping = torch.zeros(self.num_envs)

        self.check_object_coord_bins = {
            "3F": [330, 549, 524, 752],
            "3E": [394, 510, 524, 752],
            "3H": [378, 527, 524, 752],
        }

        self.crop_coord_bins = {
            "3F": [210, 690, 315, 955],
            "3E": [210, 690, 315, 955],
            "3H": [210, 690, 315, 955],
        }

        self.object_bin_prob_spawn = {
            "3F": [0.01, 0.2, 0.6, 1],
            "3E": [0.05, 0.5, 0.8, 1],
            "3H": [0.05, 0.5, 0.8, 1],
        }

        self.object_height_spawn = {
            "3F": 1.3,
            "3E": 1.4,
            "3H": 1.4,
        }

        if (self.bin_id == "3E" or self.bin_id == "3H"):
            self.smaller_bin = True
        else:
            self.smaller_bin = False

        self.check_object_coord = self.check_object_coord_bins[self.bin_id]
        self.crop_coord = self.crop_coord_bins[self.bin_id]

        # Set control limits
        self.cmd_limit = to_torch([0.1, 0.1, 0.1, 0.5, 0.5, 0.5], device=self.device).unsqueeze(0) if \
            self.control_type == "osc" else self._ur16e_effort_limits[:6].unsqueeze(0)

        # Reset all environments
        self.reset_idx(torch.arange(self.num_envs, device=self.device))

        # Refresh tensors
        self._refresh()
        self.current_directory = os.getcwd()

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

        asset_root = "assets"
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
            [0, 0, 0, 0, 0, 0, 0], dtype=torch.float, device=self.device)
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
        # asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX
        asset_options.override_com = True
        asset_options.override_inertia = True
        asset_options.vhacd_enabled = True
        # asset_options.vhacd_params = gymapi.VhacdParams()
        # asset_options.vhacd_params.resolution = 500000

        self.object_models = []
        items = os.listdir(self.google_scanned_objects_path)
        directories = [d for d in items if os.path.isdir(
            os.path.join(self.google_scanned_objects_path, d))]

        self.object_count_unique = 0
        for object_name in directories:
            self.object_count_unique += 1
            self.object_models.append(object_name)

        object_model_asset_file = []
        object_model_asset = []
        for counter, model in enumerate(self.object_models):
            object_model_asset_file.append(f"{model}/model.urdf")

            object_model_asset.append(self.gym.load_asset(
                self.sim, self.google_scanned_objects_path, object_model_asset_file[counter], asset_options))

        self.num_ur16e_bodies = self.gym.get_asset_rigid_body_count(
            ur16e_asset)
        self.num_ur16e_dofs = self.gym.get_asset_dof_count(ur16e_asset)

        print("num ur16e bodies: ", self.num_ur16e_bodies)
        print("num ur16e dofs: ", self.num_ur16e_dofs)

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
        ur16e_start_pose.p = gymapi.Vec3(0, 0, 2.020)

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

        self.table_count = 0
        # Count cubes for building the pod
        if ('cube' in self.world_params['world_model']['coll_objs']):
            cube = self.world_params['world_model']['coll_objs']['cube']
            for obj in cube.keys():
                self.table_count += 1

        # compute aggregate size
        num_ur16e_bodies = self.gym.get_asset_rigid_body_count(ur16e_asset)
        num_ur16e_shapes = self.gym.get_asset_rigid_shape_count(ur16e_asset)
        max_agg_bodies = num_ur16e_bodies + \
            self.table_count + len(self.object_models)
        max_agg_shapes = num_ur16e_shapes + \
            self.table_count + len(self.object_models)

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
                    # For flap
                    if (int(obj[4:]) >= 100):
                        dims = cube[obj]['dims']
                        pose = cube[obj]['pose']
                        self.add_table(dims, pose, ur16e_start_pose,
                                       env_ptr, i, color=[1.0, 0.96, 0.18], mesh_visual_only=False)
                    else:
                        dims = cube[obj]['dims']
                        pose = cube[obj]['pose']
                        self.add_table(dims, pose, ur16e_start_pose,
                                       env_ptr, i, color=[0.6, 0.6, 0.6], mesh_visual_only=False)

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
            self.camera_properties_back_cam.horizontal_fov = 80.0
            self.camera_properties_back_cam.width = 1280
            self.camera_properties_back_cam.height = 786
            camera_handle = self.gym.create_camera_sensor(
                env_ptr, self.camera_properties_back_cam)
            # for camera at center of the bin, coordinates are [-0.48, 0.05, 0.6]
            self.camera_base_link_translation = torch.tensor(
                [-0.18, 0.175, 0.6]).to(self.device)
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

            l_color = gymapi.Vec3(1, 1, 1)
            l_ambient = gymapi.Vec3(0.05, 0.05, 0.05)

            l_direction = gymapi.Vec3(-1, -1, 1)
            self.gym.set_light_parameters(
                self.sim, 0, l_color, l_ambient, l_direction)

            l_direction = gymapi.Vec3(-1, 1, 1)
            self.gym.set_light_parameters(
                self.sim, 1, l_color, l_ambient, l_direction)

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
    def add_table(self, table_dims, table_pose, robot_pose, env_ptr, env_id, color=[1.0, 0.0, 0.0], mesh_visual_only=False):

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
        if (not mesh_visual_only):
            self.gym.set_rigid_body_color(
                env_ptr, table_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, obj_color)
        else:
            self.gym.set_rigid_body_color(
                env_ptr, table_handle, 0, gymapi.MESH_VISUAL, obj_color)
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
        self._effort_control = torch.zeros_like(self._pos_control)
        # Initialize control
        self._arm_control = self._effort_control[:, :]

        # Initialize indices    ------ > self.num_envs * num of actors
        self._global_indices = torch.arange(self.num_envs * (self.table_count+1 + len(self.object_models)), dtype=torch.int32,
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
            self.camera_intrinsics_back_cam)
        self.dexnet_object = dexnet3(self.camera_intrinsics_back_cam)
        self.dexnet_object.load_dexnet_model()

        print("focal length in x axis: ", self.fx_back_cam)
        print("focal length in y axis: ", self.fy_back_cam)
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
            self.camera_intrinsics_gripper)
        self.force_object = calcualte_force()

    def _update_states(self):
        self.states.update({
            # ur16e
            "base_link": self._base_link[:, :7],
            "wrist_3_link": self._wrist_3_link[:, :7],
            "wrist_3_link_vel": self._wrist_3_link[:, 7:],
            "q": self._q[:, :],
            "q_gripper": self._q[:, -2:],
            "eef_pos": self._eef_state[:, :3],
            "eef_quat": self._eef_state[:, 3:7],
            "eef_vel": self._eef_state[:, 7:],
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
        self._q[env_ids, :] = pos
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
        # How many objects should we spawn 2 or 3
        probabilities = self.object_bin_prob_spawn[self.bin_id]
        random_number = self.random_number_with_probabilities(
            probabilities)

        random_number += 1
        object_list_env = {}
        object_set = range(1, self.object_count_unique+1)

        selected_object = np.random.choice(
            object_set, p=None, size=random_number, replace=False)

        list_objects_domain_randomizer = torch.tensor([])
        for object_count in selected_object:
            offset_object = np.array([np.random.uniform(0.67, 0.7, 1).reshape(
                1,)[0], np.random.uniform(-0.22, -0.12, 1).reshape(1,)[0], self.object_height_spawn[self.bin_id], np.random.uniform(0.0, 6.28, 1).reshape(1,)[0],
                np.random.uniform(0.0, 6.28, 1).reshape(1,)[0], np.random.uniform(0.0, 6.28, 1).reshape(1,)[0]])

            quat = euler_angles_to_quaternion(
                torch.tensor(offset_object[3:6]), "XYZ", degrees=False)
            offset_object = np.concatenate(
                [offset_object[:3], quat.cpu().numpy()])
            item_config = object_count
            object_list_env[item_config] = torch.tensor(offset_object)
            list_objects_domain_randomizer = torch.cat(
                (list_objects_domain_randomizer, torch.tensor([item_config])))

        for env_count in env_ids:
            env_count = env_count.item()

            self.selected_object_env[env_count] = list_objects_domain_randomizer
            self.object_pose_store[env_count] = object_list_env
            if(self.grasps_done_env[0] == 0):
                print("objects spawned in each environemnt",
                    self.selected_object_env[0], object_list_env)
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
            self.object_pose_check_list[env_id] = torch.tensor(3)
            self.speed[env_id] = torch.tensor(0.15)
            self.retract_flag_env[env_id] = 0
            self.force_encounter[env_id] = 0
            self.oscillation_store_env[env_id] = 0
            self.suction_score_store_env[env_id] = torch.tensor(0.0)
            self.retract_start_pose[env_id] = None
            self.offset_object_pose_retract[env_id] = None
            self.retract_up[env_id] = 0
            self.count_step_suction_score_calculator[env_id] = 0

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
                    env_config = torch.tensor(0).to(self.device)
                    # resetting force list
                    # randomly spawning objects
                    retKey = self.object_pose_store[env_config.item()].get(
                        counter+1)
                    if retKey is None:
                        object_pose_env = torch.tensor(
                            [[counter/4, 1, 0.5, 0.0, 0.0, 0.0, 1.0]]).to(self.device)
                    else:
                        object_pose_env = self.object_pose_store[env_config.item(
                        )][counter+1].clone().detach().to(self.device)
                        # quat = self.object_pose_store[env_count.item(
                        # )][counter+1][3:7]
                        # object_pose_env = torch.cat([object_pose_env[:3], quat])
                        object_pose_env = object_pose_env.unsqueeze(0)
                    self.object_poses = torch.cat(
                        [self.object_poses, object_pose_env])

                self._reset_init_object_state(
                    env_ids=env_ids, object=self.object_models[counter], offset=self.object_poses)

            for env_count in env_ids:
                self.force_list_save[env_count.item()] = None
                self.target_object_disp_save[env_count.item()] = None
                self.object_disp_save[env_count.item()] = {}
                self.all_objects_last_pose[env_count.item()] = {}

            # setting the objects with randomly generated poses
            for counter in range(len(self.object_models)):
                self._object_model_state[counter][env_ids] = self._init_object_model_state[counter][env_ids]

            self.progress_buf[env_ids] = 0
            self.reset_buf[env_ids] = 0

            # reinitializing the variables
            for env_id in env_ids:
                self.action_contrib[env_id] = 2
                self.force_encounter[env_id] = 0
                self.frame_count_contact_object[env_id] = 0
                self.frame_count[env_id] = 0
                self.env_reset_id_env[env_id] = 1
                self.speed[env_id] = 0.15
                self.force_contact_flag[env_id.item()] = torch.tensor(
                    0).type(torch.bool)

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
        self._refresh()
        obs = ["eef_pos", "eef_quat"]
        obs += ["q_gripper"] if self.control_type == "osc" else ["q"]
        self.obs_buf = torch.cat([self.states[ob] for ob in obs], dim=-1)

        return self.obs_buf

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
            path = str(Path(__file__).parent.absolute())
            joint_poses_list = torch.load(f"{path}/../../misc/joint_poses.pt")
            temp_pos = joint_poses_list[torch.randint(
                0, len(joint_poses_list), (1,))[0]].to(self.device)
            temp_pos = torch.reshape(temp_pos, (1, len(temp_pos)))
            temp_pos = torch.cat(
                (temp_pos, torch.tensor([[0]]).to(self.device)), dim=1)
            # temp_pos = tensor_clamp(temp_pos.unsqueeze(0), self.ur16e_dof_lower_limits.unsqueeze(0), self.ur16e_dof_upper_limits)
            pos = torch.cat([pos, temp_pos])
        return pos

    def pre_physics_step(self, actions):
        '''
        Camera access in the pre physics step to compute the force using suction cup deformation score
        '''
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)
        # communicate physics to graphics system
        self.gym.step_graphics(self.sim)
        # render the camera sensors
        self.gym.render_all_camera_sensors(self.sim)
        self.gym.start_access_image_tensors(self.sim)

        '''
        Commands to the arm for eef control
        '''
        self.actions = torch.zeros(0, 7)
        # Before each loop this will track all the environments where a condition for reset has been called
        env_list_reset_objects = torch.tensor([])
        env_list_reset_arm_pose = torch.tensor([])
        env_complete_reset = torch.tensor([])
        object_mask_area = {}
        for env_count in range(self.num_envs):
            mask_camera_tensor = self.gym.get_camera_image_gpu_tensor(
                self.sim, self.envs[env_count], self.camera_handles[env_count][0], gymapi.IMAGE_SEGMENTATION)
            torch_mask_tensor = gymtorch.wrap_tensor(mask_camera_tensor)
            segmask = torch_mask_tensor.to(self.device)
            segmask_check = segmask
            self.cmd_limit = to_torch(
                [0.1, 0.1, 0.1, 0.5, 0.5, 0.5], device=self.device).unsqueeze(0)

            if ((self.frame_count[env_count] == self.cooldown_frames) and (self.object_pose_check_list[env_count] >= torch.tensor(1))):
                # setting the pose of the object after cool down period
                bin_objects_current_pose = {}
                for object_id in self.selected_object_env[env_count]:
                    bin_objects_current_pose[int(object_id.item())] = self._root_state[env_count, self._object_model_id[int(object_id.item())-1], :][:7].type(
                        torch.float).clone().detach()
                    # Adding a 3mm offset for stablizing the pose after reset
                    # bin_objects_current_pose[int(object_id.item())][2] += 0.003
                self.object_pose_store[env_count] = bin_objects_current_pose
                env_list_reset_objects = torch.cat(
                    (env_list_reset_objects, torch.tensor([env_count])), axis=0)
                self.object_pose_check_list[env_count] -= torch.tensor(1)

            if (self.num_grasps_per_sim and (torch.sum(self.grasps_done_env) >= self.num_grasps_per_sim) and all(self.free_envs_list == 1)):
                print("Done with all environments now restart the kernel")
                sys.exit()

            ''' 
            Spawning objects until they acquire stable pose and also doesn't falls down
            '''
            if ((self.frame_count[env_count] == self.cooldown_frames) and (self.object_pose_check_list[env_count] == torch.tensor(0))):
                mask_camera_tensor = self.gym.get_camera_image_gpu_tensor(
                    self.sim, self.envs[env_count], self.camera_handles[env_count][0], gymapi.IMAGE_SEGMENTATION)
                torch_mask_tensor = gymtorch.wrap_tensor(mask_camera_tensor)
                segmask = torch_mask_tensor.to(self.device)

                segmask_object_count = segmask[self.check_object_coord[0]:self.check_object_coord[1],
                                               self.check_object_coord[2]:self.check_object_coord[3]]

                objects_spawned = len(torch.unique(segmask_object_count))-1
                total_objects = len(self.selected_object_env[env_count])

                segmask_object_coords = segmask[self.check_object_coord[0]:self.check_object_coord[1],
                                                self.check_object_coord[2]:self.check_object_coord[3]]

                object_coords_match = cv2.countNonZero(segmask.cpu().numpy(
                )) == cv2.countNonZero(segmask_object_coords.cpu().numpy())

                _all_objects_current_pose = {}
                _all_object_position_error = torch.tensor(0.0).to(self.device)

                # collecting pose of all objects
                for object_id in self.selected_object_env[env_count]:
                    _all_objects_current_pose[int(object_id.item())] = self._root_state[env_count, self._object_model_id[int(object_id.item())-1], :][:3].type(
                        torch.float).detach().clone()
                    _all_object_position_error += torch.sum(self.object_pose_store[env_count][int(object_id.item(
                    ))][:3] - self._root_state[env_count, self._object_model_id[int(object_id.item())-1], :][:3])

                    if (_all_objects_current_pose[int(object_id.item())][2] < torch.tensor(0.5)):
                        if (env_count == 0):
                            print(
                                f"Object falled down in environment {env_count}, where total objects are {total_objects} and only {objects_spawned} were spawned inside the bin")
                        env_complete_reset = torch.cat(
                            (env_complete_reset, torch.tensor([env_count])), axis=0)
                        break
                _all_object_position_error = torch.abs(
                    _all_object_position_error)
                if (_all_object_position_error > torch.tensor(0.0055)):
                    if (env_count == 0):
                        print(
                            env_count, f" object moved inside bin error with {_all_object_position_error} meters")
                    env_complete_reset = torch.cat(
                        (env_complete_reset, torch.tensor([env_count])), axis=0)
                    total_objects = 1000

                object_mask_area[env_count] = 1000
                for object_id in torch.unique(segmask_object_count):
                    segmask_area = np.zeros_like(
                        segmask_check.cpu().numpy().astype(np.uint8))
                    segmask_area[segmask_check.cpu().numpy().astype(
                        np.uint8) == object_id.item()] = 1
                    area = np.count_nonzero(segmask_area)
                    if (object_mask_area[env_count] > area):
                        object_mask_area[env_count] = area

                # check if the environment returned from reset and the frame for that enviornment is 30 or not
                # 30 frames is for cooldown period at the start for the simualtor to settle down
                if (env_count == 0 and (self.free_envs_list[env_count] == torch.tensor(1)) and total_objects == objects_spawned and object_coords_match and object_mask_area[env_count] >= 1000 and self.env_done_grasping[env_count] == 0):
                    '''
                    Running DexNet 3.0 after investigating the pose error after spawning
                    '''
                    random_object_select = random.sample(
                        self.selected_object_env[env_count].tolist(), 1)
                    for env_count_free in range(self.num_envs):
                        self.object_target_id[env_count_free] = torch.tensor(
                            random_object_select).to(self.device).type(torch.int)
                    rgb_camera_tensor = self.gym.get_camera_image_gpu_tensor(
                        self.sim, self.envs[env_count], self.camera_handles[env_count][0], gymapi.IMAGE_COLOR)
                    torch_rgb_tensor = gymtorch.wrap_tensor(rgb_camera_tensor)
                    rgb_image = torch_rgb_tensor.to(self.device)
                    rgb_image_copy = torch.reshape(
                        rgb_image, (rgb_image.shape[0], -1, 4))[..., :3]

                    self.rgb_save[env_count] = rgb_image_copy[self.crop_coord[0]:self.crop_coord[1],
                                                              self.crop_coord[2]:self.crop_coord[3]].clone().detach().cpu().numpy()

                    depth_camera_tensor = self.gym.get_camera_image_gpu_tensor(
                        self.sim, self.envs[env_count], self.camera_handles[env_count][0], gymapi.IMAGE_DEPTH)
                    torch_depth_tensor = gymtorch.wrap_tensor(
                        depth_camera_tensor)
                    depth_image = torch_depth_tensor.to(self.device)
                    depth_image = -depth_image

                    segmask_dexnet = segmask.clone().detach()
                    self.segmask_save[env_count] = segmask[self.crop_coord[0]:self.crop_coord[1],
                                                           self.crop_coord[2]:self.crop_coord[3]].clone(
                    ).detach().cpu().numpy().astype(np.uint8)

                    segmask_numpy = np.zeros_like(
                        segmask_dexnet.cpu().numpy().astype(np.uint8))
                    segmask_numpy_temp = np.zeros_like(
                        segmask_dexnet.cpu().numpy().astype(np.uint8))
                    segmask_numpy_temp[segmask_dexnet.cpu().numpy().astype(
                        np.uint8) == self.object_target_id[env_count].cpu().numpy()] = 1
                    segmask_numpy[segmask_dexnet.cpu().numpy().astype(
                        np.uint8) == self.object_target_id[env_count].cpu().numpy()] = 255
                    segmask_dexnet = BinaryImage(
                        segmask_numpy[self.crop_coord[0]:self.crop_coord[1],
                                      self.crop_coord[2]:self.crop_coord[3]], frame=self.camera_intrinsics_back_cam.frame)

                    depth_image_dexnet = depth_image.clone().detach()
                    noise_image = torch.normal(
                        0, 0.0005, size=depth_image_dexnet.size()).to(self.device)
                    depth_image_dexnet = depth_image_dexnet + noise_image

                    depth_image_save_temp = depth_image_dexnet.clone().detach().cpu().numpy()
                    self.depth_image_save[env_count] = depth_image_save_temp[self.crop_coord[0]:self.crop_coord[1],
                                                                             self.crop_coord[2]:self.crop_coord[3]]

                    '''
                    Saving data for independent environments method
                    '''
                    # # saving depth image, rgb image and segmentation mask
                    # for env_count_free in range(self.num_envs):
                    #     self.config_env_count[env_count_free] += torch.tensor(
                    #         1).type(torch.int)
                    # env_number = env_count
                    # new_dir_path = os.path.join(
                    #     self.data_path, f"{self.bin_id}/{env_number}/")
                    # env_config = self.config_env_count[env_count].type(
                    #     torch.int).item()
                    # save_dir_depth_npy = os.path.join(
                    #     new_dir_path, f'depth_image_{env_number}_{env_config}.npy')
                    # save_dir_segmask_npy = os.path.join(
                    #     new_dir_path, f'segmask_{env_number}_{env_config}.npy')
                    # save_dir_rgb_npy = os.path.join(
                    #     new_dir_path, f'rgb_{env_number}_{env_config}.npy')
                    # save_dir_rgb_png = os.path.join(
                    #     new_dir_path, f'rgb_{env_number}_{env_config}.png')

                    new_dir_path = self.data_path

                    save_dir_depth_npy = os.path.join(
                        new_dir_path, f'depth_image.npy')
                    save_dir_segmask_npy = os.path.join(
                        new_dir_path, f'segmask.npy')
                    save_dir_rgb_npy = os.path.join(
                        new_dir_path, f'rgb.npy')
                    save_dir_rgb_png = os.path.join(
                        new_dir_path, f'rgb.png')

                    Image.fromarray(self.rgb_save[env_count]).save(
                        save_dir_rgb_png)

                    with open(save_dir_depth_npy, 'wb') as f:
                        np.save(f, self.depth_image_save[env_count])

                    with open(save_dir_segmask_npy, 'wb') as f:
                        np.save(f, self.segmask_save[env_count])

                    with open(save_dir_rgb_npy, 'wb') as f:
                        np.save(f, self.rgb_save[env_count])

                    # cropping the image and modifying depth to match the DexNet 3.0 input configuration
                    depth_image_dexnet -= 0.2
                    depth_numpy = depth_image_dexnet.cpu().numpy()
                    depth_numpy_temp = depth_numpy*segmask_numpy_temp
                    depth_numpy_temp[depth_numpy_temp == 0] = 0.75

                    depth_img_dexnet = DepthImage(
                        depth_numpy_temp[self.crop_coord[0]:self.crop_coord[1],
                                         self.crop_coord[2]:self.crop_coord[3]], frame=self.camera_intrinsics_back_cam.frame)
                    max_num_grasps = 0

                    # Storing all the sampled grasp point and its properties
                    try:
                        action, self.grasps_and_predictions, self.unsorted_grasps_and_predictions = self.dexnet_object.inference(
                            depth_img_dexnet, segmask_dexnet, None)

                        self.suction_deformation_score_temp = torch.Tensor()
                        self.xyz_point_temp = torch.empty((0, 3))
                        self.grasp_angle_temp = torch.empty((0, 3))
                        self.grasp_point_temp = torch.empty((0, 2))
                        self.force_SI_temp = torch.Tensor()
                        self.dexnet_score_temp = torch.Tensor()
                        max_num_grasps = len(self.grasps_and_predictions)
                        top_grasps = max_num_grasps if max_num_grasps <= 10 else 7
                        # max_num_grasps = 3
                        self.num_grasps_per_sim = max_num_grasps
                        print(f"{self.num_grasps_per_sim} grasp points were sampled")
                        for i in range(max_num_grasps):
                            grasp_point = torch.tensor(
                                [self.grasps_and_predictions[i][0].center.x, self.grasps_and_predictions[i][0].center.y])

                            depth_image_suction = depth_image
                            offset = torch.tensor(
                                [self.crop_coord[2], self.crop_coord[0]])
                            suction_deformation_score, xyz_point, grasp_angle = self.suction_score_object.calculator(
                                depth_image_suction, segmask, rgb_image_copy, self.grasps_and_predictions[i][0], self.object_target_id[env_count], offset)
                            grasp_angle = torch.tensor([0, 0, 0])
                            self.suction_deformation_score_temp = torch.cat(
                                (self.suction_deformation_score_temp, torch.tensor([suction_deformation_score]))).type(torch.float)
                            self.xyz_point_temp = torch.cat(
                                [self.xyz_point_temp, xyz_point.unsqueeze(0)], dim=0)
                            self.grasp_angle_temp = torch.cat(
                                [self.grasp_angle_temp, grasp_angle.unsqueeze(0)], dim=0)
                            self.grasp_point_temp = torch.cat(
                                [self.grasp_point_temp, grasp_point.clone().detach().unsqueeze(0)], dim=0)
                            # cv2.circle(self.rgb_save[env_count], (grasp_point.clone().detach().cpu().numpy()[0], grasp_point.clone().detach().cpu().numpy()[1]), 2, (0, 0, 0), 1)
                            self.object_coordiante_camera = xyz_point.clone().detach()
                            if (suction_deformation_score > 0):
                                force_SI = self.force_object.regression(
                                    suction_deformation_score)
                            else:
                                force_SI = torch.tensor(0).to(self.device)

                            self.force_SI_temp = torch.cat(
                                (self.force_SI_temp, torch.tensor([force_SI])))
                            self.dexnet_score_temp = torch.cat(
                                (self.dexnet_score_temp, torch.tensor([self.grasps_and_predictions[i][1]])))

                        if (top_grasps > 0):
                            for env_count_free in range(self.num_envs):
                                # Do not need to save the json file here because the simulation has not yet started
                                env_list_reset_arm_pose = torch.cat(
                                    (env_list_reset_arm_pose, torch.tensor([env_count_free])), axis=0)
                                env_list_reset_objects = torch.cat(
                                    (env_list_reset_objects, torch.tensor([env_count_free])), axis=0)
                        else:
                            print("No sample points")
                            for env_count_free in range(self.num_envs):
                                env_complete_reset = torch.cat(
                                    (env_complete_reset, torch.tensor([env_count_free])), axis=0)
                    except Exception as e:
                        print("dexnet error: ", e)
                        for env_count_free in range(self.num_envs):
                            env_complete_reset = torch.cat(
                                (env_complete_reset, torch.tensor([env_count_free])), axis=0)

                    self.suction_deformation_score_env[env_count] = self.suction_deformation_score_temp
                    self.grasp_angle_env[env_count] = self.grasp_angle_temp
                    self.force_SI_env[env_count] = self.force_SI_temp
                    self.xyz_point_env[env_count] = self.xyz_point_temp
                    self.grasp_point_env[env_count] = self.grasp_point_temp
                    self.dexnet_score_env[env_count] = self.dexnet_score_temp
                    for env_count_free in range(self.num_envs):
                        self.free_envs_list[env_count_free] = torch.tensor(0)

                elif (total_objects != objects_spawned and (self.free_envs_list[env_count] == torch.tensor(1))):
                    for env_count_free in range(self.num_envs):
                        env_complete_reset = torch.cat(
                            (env_complete_reset, torch.tensor([env_count_free])), axis=0)
                elif (all(self.env_done_grasping == 1) and all(self.free_envs_list == 1)):
                    print("Done with all environments now restart the kernel")
                    sys.exit()

            # After every reset popping out each prperty to be used for the pick
            self.action_env = torch.tensor(
                [[0, 0, 0, 0, 0, 0, 1]], dtype=torch.float)
            if ((self.env_reset_id_env[env_count] == 1) and (self.frame_count[env_count] > self.cooldown_frames)):
                self.env_reset_id_env[env_count] = torch.tensor(0)
                self.action_env = torch.tensor(
                    [[0, 0, 0, 0, 0, 0, 1]], dtype=torch.float)
                env_config = 0
                if ((env_config in self.grasp_angle_env) and (len(self.grasp_angle_env[env_config]) != 0)):
                    self.suction_deformation_score[env_count] = self.suction_deformation_score_env[env_config][0]
                    self.suction_deformation_score_env[env_config] = self.suction_deformation_score_env[env_config][1:]
                    self.grasp_angle[env_count] = self.grasp_angle_env[env_config][0]
                    self.grasp_angle_env[env_config] = self.grasp_angle_env[env_config][1:]
                    self.xyz_point[env_count] = self.xyz_point_env[env_config][0]
                    self.xyz_point_env[env_config] = self.xyz_point_env[env_config][1:]
                    self.grasp_point[env_count] = self.grasp_point_env[env_config][0]
                    self.grasp_point_env[env_config] = self.grasp_point_env[env_config][1:]
                    self.dexnet_score[env_count] = self.dexnet_score_env[env_config][0]
                    self.dexnet_score_env[env_config] = self.dexnet_score_env[env_config][1:]
                    self.force_SI[env_count] = self.force_SI_env[env_config][0]
                    self.force_SI_env[env_config] = self.force_SI_env[env_config][1:]
                    self.grasps_done_env[env_count] += 1
                else:
                    '''
                    Uncomment this if you want to only collect data from each environment only once
                    '''
                    if (self.grasps_done_env[env_count] >= 1):
                        self.env_done_grasping[env_count] = 1
                    env_complete_reset = torch.cat(
                        (env_complete_reset, torch.tensor([env_count])), axis=0)
                try:
                    if ((env_config in self.grasp_angle_env) and (torch.count_nonzero(self.xyz_point[env_count]) < 1)):
                        # error due to illegal 3d coordinate
                        print("xyz point error", self.xyz_point[env_count])
                        env_list_reset_arm_pose = torch.cat(
                            (env_list_reset_arm_pose, torch.tensor([env_count])), axis=0)
                        env_list_reset_objects = torch.cat(
                            (env_list_reset_objects, torch.tensor([env_count])), axis=0)
                        oscillation = False
                        success = False
                        # saving all the properties of a single pick
                        json_save = {
                            "force_array": [],
                            "object_disp": {},
                            "grasp point": self.grasp_point[env_count].tolist(),
                            "grasp_angle": self.grasp_angle[env_count].tolist(),
                            "dexnet_score": self.dexnet_score[env_count].item(),
                            "suction_deformation_score": self.suction_deformation_score[env_count].item(),
                            "oscillation": oscillation,
                            "gripper_score": 0,
                            "success": success,
                            "object_id": self.object_target_id[env_count].item(),
                            "penetration": False,
                            "unreachable": True,
                            "retract": False
                        }
                        new_dir_name = str(
                            self.track_save.type(torch.int).item())
                        save_dir_json = self.data_path + "/json_data_"+new_dir_name+".json"
                        with open(save_dir_json, 'w') as json_file:
                            json.dump(json_save, json_file)
                        self.track_save = self.track_save + torch.tensor(1)
                except Exception as error:
                    env_complete_reset = torch.cat(
                        (env_complete_reset, torch.tensor([env_count])), axis=0)
                    print("xyz error in env ", env_count,
                          " and the error is ", error)

            elif (self.env_reset_id_env[env_count] == 0 and self.frame_count[env_count] > torch.tensor(self.cooldown_frames) and self.free_envs_list[env_count] == torch.tensor(0)):

                # force sensor update
                self.gym.refresh_force_sensor_tensor(self.sim)

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
                T_world_to_object = torch.matmul(
                    self.T_world_to_camera_link, T_camera_to_object)
                # Transformation for pre grasp pose (wp --> wo*op)
                rotation_matrix_pre_grasp_pose = euler_angles_to_matrix(
                    torch.tensor([0, 0, 0]).to(self.device), "XYZ", degrees=True)
                translation_pre_grasp_pose = torch.tensor(
                    [-0.25, 0, 0]).to(self.device)
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
                T_ee_pose_to_pre_grasp_pose = torch.matmul(
                    torch.inverse(T_world_to_ee_pose), T_world_to_pre_grasp_pose)
                # Orientation error
                action_orientation = matrix_to_euler_angles(
                    T_ee_pose_to_pre_grasp_pose[:3, :3], "XYZ")

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

                    if (_all_objects_current_pose[int(object_id.item())][2] < torch.tensor(0.5) and self.force_encounter[env_count] == 0):
                        print(
                            f"object falled down in environment {env_count} while reaching the grasping point")
                        env_list_reset_arm_pose = torch.cat(
                            (env_list_reset_arm_pose, torch.tensor([env_count])), axis=0)
                        env_list_reset_objects = torch.cat(
                            (env_list_reset_objects, torch.tensor([env_count])), axis=0)
                        oscillation = False
                        success = False
                        json_save = {
                            "force_array": [],
                            "object_disp": {},
                            "grasp point": self.grasp_point[env_count].tolist(),
                            "grasp_angle": self.grasp_angle[env_count].tolist(),
                            "dexnet_score": self.dexnet_score[env_count].item(),
                            "suction_deformation_score": self.suction_deformation_score[env_count].item(),
                            "oscillation": oscillation,
                            "gripper_score": 0,
                            "success": success,
                            "object_id": self.object_target_id[env_count].item(),
                            "penetration": False,
                            "unreachable": True,
                            "retract": False
                        }

                        new_dir_name = str(
                            self.track_save.type(torch.int).item())
                        save_dir_json = self.data_path + "/json_data_"+new_dir_name+".json"
                        with open(save_dir_json, 'w') as json_file:
                            json.dump(json_save, json_file)
                        self.track_save = self.track_save + torch.tensor(1)

                _all_object_position_error = torch.abs(
                    _all_object_position_error)
                _all_object_rotation_error = torch.abs(
                    _all_object_rotation_error)
                if ((_all_object_position_error > torch.tensor(0.0055)) and (self.action_contrib[env_count] == 2) and self.force_encounter[env_count] == 0):
                    env_list_reset_arm_pose = torch.cat(
                        (env_list_reset_arm_pose, torch.tensor([env_count])), axis=0)
                    env_list_reset_objects = torch.cat(
                        (env_list_reset_objects, torch.tensor([env_count])), axis=0)
                    oscillation = False
                    success = False
                    json_save = {
                        "force_array": [],
                        "object_disp": {},
                        "grasp point": self.grasp_point[env_count].tolist(),
                        "grasp_angle": self.grasp_angle[env_count].tolist(),
                        "dexnet_score": self.dexnet_score[env_count].item(),
                        "suction_deformation_score": self.suction_deformation_score[env_count].item(),
                        "oscillation": oscillation,
                        "gripper_score": 0,
                        "success": success,
                        "object_id": self.object_target_id[env_count].item(),
                        "penetration": False,
                        "unreachable": True,
                        "retract": False
                    }
                    new_dir_name = str(self.track_save.type(torch.int).item())
                    save_dir_json = self.data_path + "/json_data_"+new_dir_name+".json"
                    with open(save_dir_json, 'w') as json_file:
                        json.dump(json_save, json_file)
                    self.track_save = self.track_save + torch.tensor(1)

                self.distance = torch.tensor([1, 1, 1])
                # Gving the error between pre grasp pose and the current end effector pose
                if (self.action_contrib[env_count] >= torch.tensor(1)):
                    pose_factor, ori_factor = 1., 0.3
                    self.action_env = torch.tensor([[pose_factor*T_ee_pose_to_pre_grasp_pose[0][3], pose_factor*T_ee_pose_to_pre_grasp_pose[1][3],
                                                    pose_factor *
                                                     T_ee_pose_to_pre_grasp_pose[2][3], ori_factor *
                                                     action_orientation[0],
                                                    ori_factor*action_orientation[1], ori_factor*action_orientation[2], 1]], dtype=torch.float)

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
                              "reset because of object re placement check")
                        oscillation = False
                        success = False
                        json_save = {
                            "force_array": [],
                            "object_disp": {},
                            "grasp point": self.grasp_point[env_count].tolist(),
                            "grasp_angle": self.grasp_angle[env_count].tolist(),
                            "dexnet_score": self.dexnet_score[env_count].item(),
                            "suction_deformation_score": self.suction_deformation_score[env_count].item(),
                            "oscillation": oscillation,
                            "gripper_score": 0,
                            "success": success,
                            "object_id": self.object_target_id[env_count].item(),
                            "penetration": False,
                            "unreachable": True,
                            "retract": False
                        }
                        new_dir_name = str(
                            self.track_save.type(torch.int).item())
                        save_dir_json = self.data_path + "/json_data_"+new_dir_name+".json"
                        with open(save_dir_json, 'w') as json_file:
                            json.dump(json_save, json_file)
                        self.track_save = self.track_save + torch.tensor(1)
                    for object_id in self.selected_object_env[env_count]:
                        self.all_objects_last_pose[env_count][int(
                            object_id.item())] = _all_objects_current_pose[int(object_id.item())]
                else:
                    # Transformation for grasp pose (wg --> wo*og)
                    rotation_matrix_grasp_pose = euler_angles_to_matrix(torch.tensor(
                        [0, -self.grasp_angle[env_count][1], self.grasp_angle[env_count][0]]).to(self.device), "XYZ", degrees=False).type(torch.float)
                    translation_grasp_pose = torch.tensor(
                        [self.speed[env_count], 0, 0]).to(self.device).type(torch.float)
                    translation_grasp_pose = torch.matmul(
                        rotation_matrix_grasp_pose, translation_grasp_pose)

                    start_point = T_world_to_pre_grasp_pose[:3, 3].clone(
                    ).detach()
                    end_point = T_world_to_object[:3, 3].clone().detach()
                    current_point = T_world_to_ee_pose[:3, 3].clone().detach()
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
                    self.distance = current_point - q
                    self.action_env = torch.tensor([[self.speed[env_count],
                                                     translation_grasp_pose[1] -
                                                     self.distance[1] *
                                                     100*self.speed[env_count],
                                                     translation_grasp_pose[2] -
                                                     self.distance[2] *
                                                     100*self.speed[env_count],
                                                     self.speed[env_count]*100 *
                                                     action_orientation[0],
                                                     self.speed[env_count]*100 *
                                                     action_orientation[1],
                                                     self.speed[env_count]*100*action_orientation[2], 1]], dtype=torch.float)

                    if (not self.count_step_suction_score_calculator[env_count] % 10 and self.suction_deformation_score[env_count] > self.force_threshold and self.force_encounter[env_count] == 0):
                        rgb_camera_tensor = self.gym.get_camera_image_gpu_tensor(
                            self.sim, self.envs[env_count], self.camera_handles[env_count][1], gymapi.IMAGE_COLOR)
                        torch_rgb_tensor = gymtorch.wrap_tensor(
                            rgb_camera_tensor)
                        rgb_image = torch_rgb_tensor.to(self.device)
                        rgb_image_copy_gripper = torch.reshape(
                            rgb_image, (rgb_image.shape[0], -1, 4))[..., :3]
                        rgb_image_copy_gripper = rgb_image_copy_gripper.clone().detach()

                        mask_camera_tensor = self.gym.get_camera_image_gpu_tensor(
                            self.sim, self.envs[env_count], self.camera_handles[env_count][1], gymapi.IMAGE_SEGMENTATION)
                        torch_mask_tensor = gymtorch.wrap_tensor(
                            mask_camera_tensor)
                        segmask_gripper = torch_mask_tensor.to(self.device)
                        segmask_gripper = segmask_gripper.clone().detach()

                        depth_camera_tensor = self.gym.get_camera_image_gpu_tensor(
                            self.sim, self.envs[env_count], self.camera_handles[env_count][1], gymapi.IMAGE_DEPTH)
                        torch_depth_tensor = gymtorch.wrap_tensor(
                            depth_camera_tensor)
                        depth_image = torch_depth_tensor.to(self.device)
                        depth_image = -depth_image
                        depth_numpy_gripper = depth_image.clone().detach()
                        offset = torch.tensor(
                            [self.crop_coord[2], self.crop_coord[0]])
                        self.suction_deformation_score[env_count], _, _ = self.suction_score_object_gripper.calculator(
                            depth_numpy_gripper, segmask_gripper, rgb_image_copy_gripper, None, self.object_target_id[env_count], offset)

                        if (self.suction_deformation_score[env_count] > self.force_threshold):
                            self.force_SI[env_count] = self.force_object.regression(
                                self.suction_deformation_score[env_count])
                    self.count_step_suction_score_calculator[env_count] += 1

                    current_object_pose = self._root_state[env_count, self._object_model_id[self.object_target_id[env_count]-1], :][:3].type(
                        torch.float).detach().clone()
                    try:
                        object_pose_error = torch.abs(torch.norm(
                            current_object_pose - self.last_object_pose[env_count]))
                    except:
                        object_pose_error = torch.tensor(0)

                    # Compute the segmask and depth map of the camera at the gripper
                    mask_camera_tensor = self.gym.get_camera_image_gpu_tensor(
                        self.sim, self.envs[env_count], self.camera_handles[env_count][1], gymapi.IMAGE_SEGMENTATION)
                    torch_mask_tensor = gymtorch.wrap_tensor(
                        mask_camera_tensor)
                    segmask_gripper = torch_mask_tensor.to(self.device)
                    segmask_gripper = segmask_gripper.clone().detach()

                    depth_camera_tensor = self.gym.get_camera_image_gpu_tensor(
                        self.sim, self.envs[env_count], self.camera_handles[env_count][1], gymapi.IMAGE_DEPTH)
                    torch_depth_tensor = gymtorch.wrap_tensor(
                        depth_camera_tensor)
                    depth_image = torch_depth_tensor.to(self.device)
                    depth_image = -depth_image
                    depth_numpy_gripper = depth_image.clone().detach()

                    # center pixel of the griper camera
                    mask_point_cam = segmask_gripper[int(
                        self.height_gripper/2), int(self.width_gripper/2)]
                    if (mask_point_cam == self.object_target_id[env_count]):
                        depth_point_cam = depth_numpy_gripper[int(
                            self.height_gripper/2), int(self.width_gripper/2)]
                    else:
                        depth_point_cam = torch.tensor(10.)

                    if (object_pose_error >= 0.0003):
                        self.object_movement_enabled = 1
                    # Calculate the contact existance of the suction gripper and the target object
                    try:
                        contact_exist = self.suction_score_object_gripper.calculate_contact(
                            depth_numpy_gripper, segmask_gripper, None, None, self.object_target_id[env_count])
                    except Exception as e:
                        print("suction contact error", e)
                        contact_exist = torch.tensor(0)
                    if (self.action_contrib[env_count] == 0):
                        angle_error = quaternion_to_euler_angles(self._eef_state[env_count][3:7], "XYZ", degrees=False) - torch.tensor(
                            [0, -self.grasp_angle[env_count][1], self.grasp_angle[env_count][0]]).to(self.device)
                        if (torch.max(torch.abs(angle_error)) > torch.deg2rad(torch.tensor(7.5)) and self.force_encounter[env_count] == 0):
                            # encountered the arm insertion constraint
                            env_list_reset_arm_pose = torch.cat(
                                (env_list_reset_arm_pose, torch.tensor([env_count])), axis=0)
                            env_list_reset_objects = torch.cat(
                                (env_list_reset_objects, torch.tensor([env_count])), axis=0)
                            print(env_count, "reset because of arm angle error")
                            oscillation = False
                            success = False
                            json_save = {
                                "force_array": [],
                                "object_disp": {},
                                "grasp point": self.grasp_point[env_count].tolist(),
                                "grasp_angle": self.grasp_angle[env_count].tolist(),
                                "dexnet_score": self.dexnet_score[env_count].item(),
                                "suction_deformation_score": self.suction_deformation_score[env_count].item(),
                                "oscillation": oscillation,
                                "gripper_score": 0,
                                "success": success,
                                "object_id": self.object_target_id[env_count].item(),
                                "penetration": False,
                                "unreachable": True,
                                "retract": False
                            }
                            new_dir_name = str(
                                self.track_save.type(torch.int).item())
                            save_dir_json = self.data_path + "/json_data_"+new_dir_name+".json"
                            with open(save_dir_json, 'w') as json_file:
                                json.dump(json_save, json_file)
                            self.track_save = self.track_save + torch.tensor(1)

                        self.force_contact_flag[env_count] = torch.tensor(
                            1).type(torch.bool)

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
                    if ((_all_object_pose_error > torch.tensor(0.0075)) and contact_exist == torch.tensor(0) and self.force_encounter[env_count] == 0):
                        env_list_reset_arm_pose = torch.cat(
                            (env_list_reset_arm_pose, torch.tensor([env_count])), axis=0)
                        env_list_reset_objects = torch.cat(
                            (env_list_reset_objects, torch.tensor([env_count])), axis=0)
                        print(env_count, _all_object_pose_error,
                              "reset because of object re placement check")
                        oscillation = False
                        success = False
                        json_save = {
                            "force_array": [],
                            "object_disp": {},
                            "grasp point": self.grasp_point[env_count].tolist(),
                            "grasp_angle": self.grasp_angle[env_count].tolist(),
                            "dexnet_score": self.dexnet_score[env_count].item(),
                            "suction_deformation_score": self.suction_deformation_score[env_count].item(),
                            "oscillation": oscillation,
                            "gripper_score": 0,
                            "success": success,
                            "object_id": self.object_target_id[env_count].item(),
                            "penetration": False,
                            "unreachable": True,
                            "retract": False
                        }
                        new_dir_name = str(
                            self.track_save.type(torch.int).item())
                        save_dir_json = self.data_path + "/json_data_"+new_dir_name+".json"
                        with open(save_dir_json, 'w') as json_file:
                            json.dump(json_save, json_file)
                        self.track_save = self.track_save + torch.tensor(1)

                    # If the object is moving then increase the speed else go to the default value of 0.1
                    if ((depth_point_cam < torch.tensor(0.03)) and (self.action_contrib[env_count] == torch.tensor(0)) and (object_pose_error <= torch.tensor(0.001)) and (contact_exist == torch.tensor(1)) and self.force_encounter[env_count] == 0):
                        self.speed[env_count] += torch.tensor(0.025)
                        self.speed[env_count] = torch.min(
                            torch.tensor(1.), self.speed[env_count])
                        self.cmd_limit = to_torch(
                            [0.25, 0.25, 0.25, 0.75, 0.75, 0.75], device=self.device).unsqueeze(0)
                    else:
                        self.speed[env_count] = torch.tensor(0.15)
                        self.cmd_limit = to_torch(
                            [0.1, 0.1, 0.1, 0.5, 0.5, 0.5], device=self.device).unsqueeze(0)

                    self.last_object_pose[env_count] = current_object_pose

                    for object_id in self.selected_object_env[env_count]:
                        self.all_objects_last_pose[env_count][int(
                            object_id.item())] = _all_objects_current_pose[int(object_id.item())]

                # If arm cross the force required to grasp the object
                if (self.frame_count[env_count] > torch.tensor(self.cooldown_frames) and self.frame_count_contact_object[env_count] == torch.tensor(0)):
                    if ((torch.max(torch.abs(self.action_env[0][:3]))) <= 0.001 and (torch.max(torch.abs(self.action_env[0][3:6]))) <= 0.001):
                        self.action_contrib[env_count] -= 1
                        rgb_camera_tensor = self.gym.get_camera_image_gpu_tensor(
                            self.sim, self.envs[env_count], self.camera_handles[env_count][1], gymapi.IMAGE_COLOR)
                        torch_rgb_tensor = gymtorch.wrap_tensor(
                            rgb_camera_tensor)
                        rgb_image = torch_rgb_tensor.to(self.device)
                        rgb_image_copy_gripper = torch.reshape(
                            rgb_image, (rgb_image.shape[0], -1, 4))[..., :3]
                        rgb_image_copy_gripper = rgb_image_copy_gripper.clone().detach()

                        mask_camera_tensor = self.gym.get_camera_image_gpu_tensor(
                            self.sim, self.envs[env_count], self.camera_handles[env_count][1], gymapi.IMAGE_SEGMENTATION)
                        torch_mask_tensor = gymtorch.wrap_tensor(
                            mask_camera_tensor)
                        segmask_gripper = torch_mask_tensor.to(self.device)
                        segmask_gripper = segmask_gripper.clone().detach()

                        depth_camera_tensor = self.gym.get_camera_image_gpu_tensor(
                            self.sim, self.envs[env_count], self.camera_handles[env_count][1], gymapi.IMAGE_DEPTH)
                        torch_depth_tensor = gymtorch.wrap_tensor(
                            depth_camera_tensor)
                        depth_image = torch_depth_tensor.to(self.device)
                        depth_image = -depth_image
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

                    if (self.force_pre_physics > torch.max(torch.tensor([self.force_threshold, self.force_SI[env_count]])) and self.action_contrib[env_count] == 0 and self.force_encounter[env_count] == 0):
                        self.force_encounter[env_count] = 1
                        self.retract_flag_env[env_count] = 1
                        object_pose_at_contact = self._root_state[env_count, self._object_model_id[int(self.object_target_id[env_count].item())-1], :][:7].type(
                            torch.float).clone().detach()

                        self.offset_object_pose_retract[env_count] = object_pose_at_contact[:3] - \
                            T_world_to_ee_pose[:3, 3].clone().detach() + \
                            torch.tensor([0.005, 0, 0]).to(self.device)
                        self.retract_start_pose[env_count] = T_world_to_ee_pose[:3, 3].clone(
                        ).detach()

                        '''
                        Gripper camera
                        '''
                        rgb_camera_tensor = self.gym.get_camera_image_gpu_tensor(
                            self.sim, self.envs[env_count], self.camera_handles[env_count][1], gymapi.IMAGE_COLOR)
                        torch_rgb_tensor = gymtorch.wrap_tensor(
                            rgb_camera_tensor)
                        rgb_image = torch_rgb_tensor.to(self.device)
                        rgb_image_copy_gripper = torch.reshape(
                            rgb_image, (rgb_image.shape[0], -1, 4))[..., :3]
                        rgb_image_copy_gripper = rgb_image_copy_gripper.clone().detach()

                        mask_camera_tensor = self.gym.get_camera_image_gpu_tensor(
                            self.sim, self.envs[env_count], self.camera_handles[env_count][1], gymapi.IMAGE_SEGMENTATION)
                        torch_mask_tensor = gymtorch.wrap_tensor(
                            mask_camera_tensor)
                        segmask_gripper = torch_mask_tensor.to(self.device)
                        segmask_gripper = segmask_gripper.clone().detach()

                        depth_camera_tensor = self.gym.get_camera_image_gpu_tensor(
                            self.sim, self.envs[env_count], self.camera_handles[env_count][1], gymapi.IMAGE_DEPTH)
                        torch_depth_tensor = gymtorch.wrap_tensor(
                            depth_camera_tensor)
                        depth_image = torch_depth_tensor.to(self.device)
                        depth_image = -depth_image
                        depth_numpy_gripper = depth_image.clone().detach()

                        offset = torch.tensor(
                            [self.crop_coord[2], self.crop_coord[0]])
                        self.suction_score_store_env[env_count], _, _ = self.suction_score_object_gripper.calculator(
                            depth_numpy_gripper, segmask_gripper, rgb_image_copy_gripper, None, self.object_target_id[env_count], offset)
                        print(env_count, " force: ", self.force_pre_physics)
                        print(env_count, " suction gripper ",
                              self.suction_score_store_env[env_count])

                        self.oscillation_store_env[env_count] = self.detect_oscillation(
                            self.force_list_save[env_count])

                    # If the arm collided with the environment
                    elif (self.force_pre_physics > torch.tensor(10) and self.action_contrib[env_count] == 1):
                        print(env_count, " force due to collision: ",
                              self.force_pre_physics)
                        env_list_reset_arm_pose = torch.cat(
                            (env_list_reset_arm_pose, torch.tensor([env_count])), axis=0)
                        env_list_reset_objects = torch.cat(
                            (env_list_reset_objects, torch.tensor([env_count])), axis=0)
                        oscillation = False
                        success = False
                        json_save = {
                            "force_array": [],
                            "object_disp": {},
                            "grasp point": self.grasp_point[env_count].tolist(),
                            "grasp_angle": self.grasp_angle[env_count].tolist(),
                            "dexnet_score": self.dexnet_score[env_count].item(),
                            "suction_deformation_score": self.suction_deformation_score[env_count].item(),
                            "oscillation": oscillation,
                            "gripper_score": 0,
                            "success": success,
                            "object_id": self.object_target_id[env_count].item(),
                            "penetration": False,
                            "unreachable": True,
                            "retract": False
                        }
                        new_dir_name = str(
                            self.track_save.type(torch.int).item())
                        save_dir_json = self.data_path + "/json_data_"+new_dir_name+".json"
                        with open(save_dir_json, 'w') as json_file:
                            json.dump(json_save, json_file)
                        self.track_save = self.track_save + torch.tensor(1)

                    if (self.force_encounter[env_count] == 1):
                        # Transformation for grasp pose (wg --> wo*og)
                        rotation_matrix_grasp_pose = euler_angles_to_matrix(torch.tensor(
                            [0, -self.grasp_angle[env_count][1], self.grasp_angle[env_count][0]]).to(self.device), "XYZ", degrees=False).type(torch.float)
                        translation_grasp_pose = torch.tensor(
                            [self.speed[env_count], 0, 0]).to(self.device).type(torch.float)
                        translation_grasp_pose = torch.matmul(
                            rotation_matrix_grasp_pose, translation_grasp_pose)

                        if (self.retract_up[env_count] == 1):
                            end_point = T_world_to_pre_grasp_pose[:3, 3].clone(
                            ).detach()
                            end_point[2] += 0.03
                            # end_point[0] -= 0.02
                        else:
                            end_point = self.retract_start_pose[env_count].clone(
                            ).detach()
                            end_point[2] += 0.03

                        start_point = self.retract_start_pose[env_count].clone(
                        ).detach()
                        current_point = T_world_to_ee_pose[:3, 3].clone(
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
                        self.distance = current_point - q

                        if (self.retract_up[env_count] == 0):
                            self.action_env = torch.tensor([[-0.001,
                                                            translation_grasp_pose[1] -
                                                            self.distance[1] *
                                                            50 *
                                                            self.speed[env_count],
                                                            self.speed[env_count],
                                                            self.speed[env_count]*50 *
                                                            action_orientation[0],
                                                            self.speed[env_count]*50 *
                                                            action_orientation[1],
                                                            self.speed[env_count]*50*action_orientation[2], 1]], dtype=torch.float)

                        if (self.retract_up[env_count] == 1):
                            self.action_env = torch.tensor([[-self.speed[env_count],
                                                            translation_grasp_pose[1] -
                                                            self.distance[1] *
                                                            50 *
                                                            self.speed[env_count],
                                                            translation_grasp_pose[2] -
                                                            self.distance[2] *
                                                            50 *
                                                            self.speed[env_count],
                                                            self.speed[env_count]*50 *
                                                            action_orientation[0],
                                                            self.speed[env_count]*50 *
                                                            action_orientation[1],
                                                            self.speed[env_count]*50*action_orientation[2], 1]], dtype=torch.float)

                        self.env_reset_retract = torch.cat(
                            (self.env_reset_retract, torch.tensor([env_count]).to(self.device)), axis=0).type(torch.long)

                        env_temp_id = torch.tensor([]).to(self.device)

                        env_temp_id = torch.cat(
                            (env_temp_id, torch.tensor([env_count]).to(self.device)), axis=0).type(torch.long)
                        self.retract_object_state[env_count] = torch.zeros(
                            len(env_temp_id), 13, device=self.device)
                        self.retract_object_state[env_count][:,
                                                             3:7] = self._init_object_model_state[self.object_target_id[env_count]-1][env_count][3:7]

                        self.retract_object_state[env_count][:, :3] = T_world_to_ee_pose[:3,
                                                                                         3].clone().detach() + self.offset_object_pose_retract[env_count].clone().detach()
                        this_object_state_all = self._init_object_model_state[
                            self.object_target_id[env_count]-1]

                        # Lastly, set these sampled values as the new init state
                        this_object_state_all[env_count,
                                              :] = self.retract_object_state[env_count]
                        self._object_model_state[self.object_target_id[env_count] -
                                                 1][env_count] = self._init_object_model_state[self.object_target_id[env_count]-1][env_count]

                        angle_error = quaternion_to_euler_angles(self._eef_state[env_count][3:7], "XYZ", degrees=False) - torch.tensor(
                            [0, -self.grasp_angle[env_count][1], self.grasp_angle[env_count][0]]).to(self.device)

                        current_target_object_pose = self._root_state[env_count, self._object_model_id[int(self.object_target_id[env_count].item())-1], :][:3].type(
                            torch.float).clone().detach() - self.offset_object_pose_retract[env_count]

                        if (torch.max(torch.abs(angle_error)) > torch.deg2rad(torch.tensor(25)) or (torch.max(torch.abs(current_target_object_pose - T_world_to_ee_pose[:3, 3])) >= 0.0055)):

                            # encountered the arm insertion constraint
                            env_list_reset_arm_pose = torch.cat(
                                (env_list_reset_arm_pose, torch.tensor([env_count])), axis=0)
                            env_list_reset_objects = torch.cat(
                                (env_list_reset_objects, torch.tensor([env_count])), axis=0)
                            print(
                                env_count, "reset because of arm angle error while retract")

                            success = False
                            if (self.suction_score_store_env[env_count] > torch.tensor(0.1) and self.oscillation_store_env[env_count] == False):
                                success = True
                            penetration = False
                            if (self.suction_score_store_env[env_count] == torch.tensor(0)):
                                penetration = True

                            object_disp_json_save = self.object_disp_save[env_count].copy(
                            )
                            for object_id in self.selected_object_env[env_count]:
                                object_disp_json_save[int(object_id.item(
                                ))] = object_disp_json_save[int(object_id.item())].tolist()

                            json_save = {
                                "force_array": self.force_list_save[env_count].tolist(),
                                "object_disp": object_disp_json_save,
                                "grasp point": self.grasp_point[env_count].tolist(),
                                "grasp_angle": self.grasp_angle[env_count].tolist(),
                                "dexnet_score": self.dexnet_score[env_count].item(),
                                "suction_deformation_score": self.suction_deformation_score[env_count].item(),
                                "oscillation": self.oscillation_store_env[env_count],
                                "gripper_score": self.suction_score_store_env[env_count].item(),
                                "success": success,
                                "object_id": self.object_target_id[env_count].item(),
                                "penetration": penetration,
                                "unreachable": False,
                                "retract": False
                            }
                            new_dir_name = str(
                                self.track_save.type(torch.int).item())
                            save_dir_json = self.data_path + "/json_data_"+new_dir_name+".json"
                            with open(save_dir_json, 'w') as json_file:
                                json.dump(json_save, json_file)
                            self.track_save = self.track_save + torch.tensor(1)

                        if (self.retract_up[env_count] == 0 and current_point[2] >= end_point[2]):
                            self.retract_up[env_count] = 1
                            self.retract_start_pose[env_count] = T_world_to_ee_pose[:3, 3]

                        if (self.retract_up[env_count] == 1 and current_point[0] <= 0.25):
                            self.frame_count_contact_object[env_count] = 1

                            success = False
                            if (self.suction_score_store_env[env_count] > torch.tensor(0.1) and self.oscillation_store_env[env_count] == False):
                                success = True
                            penetration = False
                            if (self.suction_score_store_env[env_count] == torch.tensor(0)):
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
                                "oscillation": self.oscillation_store_env[env_count],
                                "gripper_score": self.suction_score_store_env[env_count].item(),
                                "success": success,
                                "object_id": self.object_target_id[env_count].item(),
                                "penetration": penetration,
                                "unreachable": False,
                                "retract": True
                            }
                            new_dir_name = str(
                                self.track_save.type(torch.int).item())
                            save_dir_json = self.data_path + "/json_data_"+new_dir_name+".json"
                            with open(save_dir_json, 'w') as json_file:
                                json.dump(json_save, json_file)
                            self.track_save = self.track_save + torch.tensor(1)

                            print("final_pose after retract for env ",
                                  T_world_to_ee_pose[:3, 3], env_count)

                elif (self.frame_count_contact_object[env_count] == torch.tensor(1) and self.frame_count[env_count] > torch.tensor(self.cooldown_frames)):
                    # Dont need to save json here because this condition is satisfied when the pick was successfull
                    env_list_reset_arm_pose = torch.cat(
                        (env_list_reset_arm_pose, torch.tensor([env_count])), axis=0)
                    env_list_reset_objects = torch.cat(
                        (env_list_reset_objects, torch.tensor([env_count])), axis=0)

            if (self.frame_count[env_count] <= torch.tensor(self.cooldown_frames)):
                self.action_env = torch.tensor(
                    [[0, 0, 0, 0, 0, 0, 1]], dtype=torch.float)

            self.actions = torch.cat([self.actions, self.action_env])
            self.frame_count[env_count] += torch.tensor(1)

        # Parallelizing multiple environments for resetting the arm for pre grasp pose or to reset the particular environment
        if (len(env_complete_reset) != 0 and len(env_list_reset_arm_pose) != 0):
            env_complete_reset = torch.unique(env_complete_reset)

            env_list_reset_objects = torch.cat(
                (env_list_reset_objects, env_complete_reset), axis=0)

            env_list_reset_arm_pose = torch.unique(env_list_reset_arm_pose)
            env_list_reset_arm_pose = torch.tensor(
                [x for x in env_list_reset_arm_pose if x not in env_complete_reset])

            env_ids = torch.cat(
                (env_list_reset_arm_pose, env_complete_reset), axis=0)
            env_ids = env_ids.to(self.device).type(torch.long)
            pos1 = self.reset_pre_grasp_pose(
                env_list_reset_arm_pose.to(self.device).type(torch.long))
            pos2 = self.reset_init_arm_pose(
                env_complete_reset.to(self.device).type(torch.long))

            pos = torch.cat([pos1, pos2])
            self.deploy_actions(env_ids, pos)

        elif (len(env_list_reset_arm_pose) != 0):
            env_list_reset_arm_pose = torch.unique(env_list_reset_arm_pose)
            pos = self.reset_pre_grasp_pose(
                env_list_reset_arm_pose.to(self.device).type(torch.long))

            env_ids = env_list_reset_arm_pose.to(self.device).type(torch.long)
            self.deploy_actions(env_ids, pos)

        elif (len(env_complete_reset) != 0):
            env_complete_reset = torch.unique(env_complete_reset)
            env_list_reset_objects = torch.cat(
                (env_list_reset_objects, env_complete_reset), axis=0)
            pos = self.reset_init_arm_pose(
                env_complete_reset.to(self.device).type(torch.long))
            env_ids = env_complete_reset.to(self.device).type(torch.long)
            self.deploy_actions(env_ids, pos)

        if (len(env_list_reset_objects) != 0 or len(self.env_reset_retract) != 0):
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

        self.actions = self.actions.clone().detach().to(self.device)

        # Split arm and gripper command
        u_arm, u_gripper = self.actions[:, :-1], self.actions[:, -1]

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
        self.progress_buf += 1
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)

        # check if there is atimeout
        if len(env_ids) > 0:
            for env_id in env_ids:
                env_count = env_id.item()
                if (self.force_list_save[env_count] != None and len(self.force_list_save[env_count]) > 10):
                    if (self.force_encounter[env_count] == 1):
                        print("post physics reset because retraction failed")
                        # encountered the arm insertion constraint
                        success = False
                        if (self.suction_score_store_env[env_count] > torch.tensor(0.1) and self.oscillation_store_env[env_count] == False):
                            success = True
                        penetration = False
                        if (self.suction_score_store_env[env_count] == torch.tensor(0)):
                            penetration = True

                        object_disp_json_save = self.object_disp_save[env_count].copy(
                        )
                        for object_id in self.selected_object_env[env_count]:
                            object_disp_json_save[int(object_id.item(
                            ))] = object_disp_json_save[int(object_id.item())].tolist()

                        json_save = {
                            "force_array": self.force_list_save[env_count].tolist(),
                            "object_disp": object_disp_json_save,
                            "grasp point": self.grasp_point[env_count].tolist(),
                            "grasp_angle": self.grasp_angle[env_count].tolist(),
                            "dexnet_score": self.dexnet_score[env_count].item(),
                            "suction_deformation_score": self.suction_deformation_score[env_count].item(),
                            "oscillation": self.oscillation_store_env[env_count],
                            "gripper_score": self.suction_score_store_env[env_count].item(),
                            "success": success,
                            "object_id": self.object_target_id[env_count].item(),
                            "penetration": penetration,
                            "unreachable": False,
                            "retract": False
                        }
                        new_dir_name = str(
                            self.track_save.type(torch.int).item())
                        save_dir_json = self.data_path + "/json_data_"+new_dir_name+".json"
                        with open(save_dir_json, 'w') as json_file:
                            json.dump(json_save, json_file)
                        self.track_save = self.track_save + torch.tensor(1)
                    else:
                        oscillation = self.detect_oscillation(
                            self.force_list_save[env_count])
                        success = False

                        object_disp_json_save = self.object_disp_save[env_count].copy(
                        )
                        for object_id in self.selected_object_env[env_count]:
                            object_disp_json_save[int(object_id.item(
                            ))] = object_disp_json_save[int(object_id.item())].tolist()

                        json_save = {
                            "force_array": self.force_list_save[env_count].tolist(),
                            "object_disp": object_disp_json_save,
                            "grasp point": self.grasp_point[env_count].tolist(),
                            "grasp_angle": self.grasp_angle[env_count].tolist(),
                            "dexnet_score": self.dexnet_score[env_count].item(),
                            "suction_deformation_score": self.suction_deformation_score[env_count].item(),
                            "oscillation": oscillation,
                            "gripper_score": 0,
                            "success": success,
                            "object_id": self.object_target_id[env_count].item(),
                            "penetration": False,
                            "unreachable": False,
                            "retract": False
                        }
                        new_dir_name = str(
                            self.track_save.type(torch.int).item())
                        save_dir_json = self.data_path + "/json_data_"+new_dir_name+".json"
                        with open(save_dir_json, 'w') as json_file:
                            json.dump(json_save, json_file)
                        self.track_save = self.track_save + torch.tensor(1)
                else:
                    oscillation = False
                    success = False
                    unreachable = True
                    json_save = {
                        "force_array": [],
                        "object_disp": {},
                        "grasp point": self.grasp_point[env_count].tolist(),
                        "grasp_angle": self.grasp_angle[env_count].tolist(),
                        "dexnet_score": self.dexnet_score[env_count].item(),
                        "suction_deformation_score": self.suction_deformation_score[env_count].item(),
                        "oscillation": oscillation,
                        "gripper_score": 0,
                        "success": success,
                        "object_id": self.object_target_id[env_count].item(),
                        "penetration": False,
                        "unreachable": unreachable,
                        "retract": False
                    }
                    new_dir_name = str(self.track_save.type(torch.int).item())
                    save_dir_json = self.data_path + "/json_data_"+new_dir_name+".json"
                    with open(save_dir_json, 'w') as json_file:
                        json.dump(json_save, json_file)
                    self.track_save = self.track_save + torch.tensor(1)

            print(f"timeout reset for environment {env_ids}")
            pos = self.reset_pre_grasp_pose(env_ids)
            self.deploy_actions(env_ids, pos)
            self.reset_object_pose(env_ids)

        self.compute_observations()
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
