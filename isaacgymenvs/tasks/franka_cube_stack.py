import numpy as np
import os
import torch
import yaml

from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym.torch_utils import *

from isaacgymenvs.utils.torch_jit_utils import *
from isaacgymenvs.tasks.base.vec_task import VecTask

# For camera module
from PIL import Image as im
import matplotlib.pyplot as plt

from isaacgym import gymutil
import math
import cv2

from suction_cup_modelling.suction_score_calcualtor import calcualte_suction_score
from suction_cup_modelling.force_calculator import calcualte_force

from gqcnn.examples.policy_for_training import dexnet3

from autolab_core import (YamlConfig, Logger, BinaryImage, CameraIntrinsics, ColorImage, DepthImage, RgbdImage)
import assets.urdf_models.models_data as md

from homogeneous_trasnformation_and_conversion.rotation_conversions import *

import time
from pathlib import Path
import pandas as pd

from pathlib import Path
cur_path = str(Path(__file__).parent.absolute())

class FrankaCubeStack(VecTask):

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        self.cfg = cfg

        self.max_episode_length = self.cfg["env"]["episodeLength"]

        self.action_scale = self.cfg["env"]["actionScale"]
        self.start_position_noise = self.cfg["env"]["startPositionNoise"]
        self.start_rotation_noise = self.cfg["env"]["startRotationNoise"]
        self.franka_position_noise = self.cfg["env"]["frankaPositionNoise"]
        self.franka_rotation_noise = self.cfg["env"]["frankaRotationNoise"]
        self.franka_dof_noise = self.cfg["env"]["frankaDofNoise"]
        self.aggregate_mode = self.cfg["env"]["aggregateMode"]

        # Controller type
        self.control_type = self.cfg["env"]["controlType"]
        assert self.control_type in {"osc", "joint_tor"},\
            "Invalid control type specified. Must be one of: {osc, joint_tor}"

        # dimensions
        # obs include: cubeA_pose (7) + cubeB_pos (3) + eef_pose (7) + q_gripper (2)
        self.cfg["env"]["numObservations"] = 9 if self.control_type == "osc" else 27
        # actions include: delta EEF if OSC (6) or joint torques (7) + bool gripper (1)
        self.cfg["env"]["numActions"] = 6 if self.control_type == "osc" else 7

        # Values to be filled in at runtime
        self.states = {}                        # will be dict filled with relevant states to use for reward calculation
        self.handles = {}                       # will be dict mapping names to relevant sim handles
        self.num_dofs = None                    # Total number of DOFs per env
        self.actions = None                     # Current actions to be deployed

        self.models_lib = md.model_lib()

        # Tensor placeholders
        self._root_state = None             # State of root body        (n_envs, 13)
        self._dof_state = None  # State of all joints       (n_envs, n_dof)
        self._q = None  # Joint positions           (n_envs, n_dof)
        self._qd = None                     # Joint velocities          (n_envs, n_dof)
        self._rigid_body_state = None  # State of all rigid bodies             (n_envs, n_bodies, 13)
        self._contact_forces = None     # Contact forces in sim
        self._eef_state = None  # end effector state (at grasping point)
        self._j_eef = None  # Jacobian for end effector
        self._mm = None  # Mass matrix
        self._arm_control = None  # Tensor buffer for controlling arm
        self._gripper_control = None  # Tensor buffer for controlling gripper
        self._pos_control = None            # Position actions
        self._effort_control = None         # Torque actions
        self._franka_effort_limits = None        # Actuator effort limits for franka
        self._global_indices = None         # Unique indices corresponding to all envs in flattened array

        # camera handles
        self.camera_handles = [[]]
        self.camera_info = None

        self.debug_viz = self.cfg["env"]["enableDebugVis"]

        self.up_axis = "z"
        self.up_axis_idx = 2

        self.franka_hand = "ee_link"
        self.franka_wrist_3_link = "wrist_3_link"
        self.franka_base = "base_link"
        self.suction_gripper = "epick_end_effector"

        self.object_coordiante_camera = torch.tensor([0, 0, 0])
        

        # Parallelization
        self.init_camera_capture = 1

        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../configs")+'/collision_primitives_3d.yml') as file:
            self.world_params = yaml.load(file, Loader=yaml.FullLoader)

        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)

        # Franka defaults
        # self.franka_default_dof_pos = to_torch(
        #     [0.06, -2.5, 2.53, 0.58, 1.67, 1.74], device=self.device
        # )
        self.franka_default_dof_pos = to_torch(
            [-1.57, 0, 0, 0, 0, 0, 0], device=self.device
        )

        self.cooldown_frames = 100

        # System IDentification data results
        self.cur_path = str(Path(__file__).parent.absolute())
        self.force_list = np.array([])
        self.rgb_camera_visualization = None
        self.dexnet_coordinates = np.array([])
        self.grasp_angle = torch.tensor([[0, 0, 0]])
        self.grasps_and_predictions = None

        # #dexnet results
        # self.suction_deformation_score = torch.tensor(0.0)
        # self.score_gripper = 0.0
        # self.force_SI = torch.tensor(0)
        # self.last_pos = None

        # OSC Gains
        self.kp = to_torch([150., 150., 150., 100., 100., 100.], device=self.device)
        self.kd = 2 * torch.sqrt(self.kp)
        self.kp_null = to_torch([10.]*6, device=self.device)
        self.kd_null = 2 * torch.sqrt(self.kp_null)
        #self.cmd_limit = None                   # filled in later

        self.action_contrib = torch.ones(self.num_envs)
        self.frame_count_contact_object = torch.zeros(self.num_envs)
        self.force_encounter = torch.zeros(self.num_envs)
        self.frame_count = torch.zeros(self.num_envs)
        # self.stack_grasp_points = [[]]*self.num_envs
        self.free_envs_list = torch.ones(self.num_envs)
        print("No. of environments: ", self.num_envs)

        self.suction_deformation_score_temp = torch.Tensor()
        self.xyz_point_temp = torch.Tensor([])
        self.grasp_angle_temp = torch.Tensor([])
        self.force_SI_temp = torch.Tensor()

        self.suction_deformation_score_env = {}
        self.xyz_point_env = {}
        self.grasp_angle_env = {}
        self.force_SI_env = {}

        self.suction_deformation_score = {}
        self.xyz_point = {}
        self.grasp_angle = {}
        self.force_SI = {}

        self.env_reset_id_env = torch.ones(self.num_envs)
        self.bootup_reset = torch.ones(self.num_envs)

        # Set control limits
        self.cmd_limit = to_torch([0.1, 0.1, 0.1, 0.5, 0.5, 0.5], device=self.device).unsqueeze(0) if \
        self.control_type == "osc" else self._franka_effort_limits[:6].unsqueeze(0)

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
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../assets")
        franka_asset_file = "urdf/Aurmar_description/robots/robot_storm.urdf"
        
        # load franka asset
        asset_options = gymapi.AssetOptions()
        asset_options.flip_visual_attachments = True
        asset_options.fix_base_link = True
        asset_options.collapse_fixed_joints = False
        asset_options.disable_gravity = True
        asset_options.thickness = 0.001
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_EFFORT
        asset_options.use_mesh_materials = True
        franka_asset = self.gym.load_asset(self.sim, asset_root, franka_asset_file, asset_options)

        franka_dof_stiffness = to_torch([0, 0, 0, 0, 0, 0, 0], dtype=torch.float, device=self.device)
        franka_dof_damping = to_torch([0., 0., 0., 0., 0., 0., 0.], dtype=torch.float, device=self.device)

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

        # self.object_models = ["centrum_box", "pizza_cutter", "calcium_bottle"]
        self.object_models = ["centrum_box"]
        object_model_asset_file = []
        object_model_asset = []
        for counter, model in enumerate(self.object_models):
            object_model_asset_file.append("urdf_models/models/"+model+"/model.urdf")
            object_model_asset.append(self.gym.load_asset(self.sim, asset_root, object_model_asset_file[counter], asset_options))
            
        self.num_franka_bodies = self.gym.get_asset_rigid_body_count(franka_asset)
        self.num_franka_dofs = self.gym.get_asset_dof_count(franka_asset)

        print("num franka bodies: ", self.num_franka_bodies)
        print("num franka dofs: ", self.num_franka_dofs)

        # set franka dof properties
        franka_dof_props = self.gym.get_asset_dof_properties(franka_asset)
        self.franka_dof_lower_limits = []
        self.franka_dof_upper_limits = []
        self._franka_effort_limits = []
        for i in range(self.num_franka_dofs):
            franka_dof_props['driveMode'][i] = gymapi.DOF_MODE_POS if i > 6 else gymapi.DOF_MODE_EFFORT
            if self.physics_engine == gymapi.SIM_PHYSX:
                franka_dof_props['stiffness'][i] = franka_dof_stiffness[i]
                franka_dof_props['damping'][i] = franka_dof_damping[i]
            else:
                franka_dof_props['stiffness'][i] = 7000.0
                franka_dof_props['damping'][i] = 50.0

            self.franka_dof_lower_limits.append(franka_dof_props['lower'][i])
            self.franka_dof_upper_limits.append(franka_dof_props['upper'][i])
            self._franka_effort_limits.append(franka_dof_props['effort'][i])

        self.franka_dof_lower_limits = to_torch(self.franka_dof_lower_limits, device=self.device)
        self.franka_dof_upper_limits = to_torch(self.franka_dof_upper_limits, device=self.device)
        self._franka_effort_limits = to_torch(self._franka_effort_limits, device=self.device)
        self.franka_dof_speed_scales = torch.ones_like(self.franka_dof_lower_limits)

        # Define start pose for franka
        franka_start_pose = gymapi.Transform()
        franka_start_pose.p = gymapi.Vec3(0, 0, 2.020) #gymapi.Vec3(-0.45, 0.0, 1.0 + table_thickness / 2 + table_stand_height)
        
        quat = euler_angles_to_quaternion(torch.tensor([180, 0, 0]).to(self.device), "XYZ", degrees=True)
        franka_start_pose.r = gymapi.Quat(quat[0], quat[1], quat[2], quat[3])

        # Define start pose for table
        table_start_pose = gymapi.Transform()
        table_start_pose.p = gymapi.Vec3(*table_pos)
        table_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
        self._table_surface_pos = np.array(table_pos) + np.array([0, 0, table_thickness / 2])

        object_model_start_pose = []
        for counter in range(len(self.object_models)):
            object_model_start_pose.append(gymapi.Transform())
            object_model_start_pose[counter].p = gymapi.Vec3(0.0, -0.1, -10.0)
            object_model_start_pose[counter].r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        # compute aggregate size
        num_franka_bodies = self.gym.get_asset_rigid_body_count(franka_asset)
        num_franka_shapes = self.gym.get_asset_rigid_shape_count(franka_asset)
        max_agg_bodies = num_franka_bodies + 27 + len(self.object_models)   # 1 for table, table stand, cubeA, cubeB
        max_agg_shapes = num_franka_shapes + 27 + len(self.object_models)    # 1 for table, table stand, cubeA, cubeB

        self.frankas = []
        self.envs = []

        self.multi_body_idx = {
            "base_link": self.gym.find_asset_rigid_body_index(franka_asset, "base_link"),
            "wrist_3_link": self.gym.find_asset_rigid_body_index(franka_asset, "wrist_3_link"),
            "ee_link": self.gym.find_asset_rigid_body_index(franka_asset, "ee_link"),
            "epick_end_effector": self.gym.find_asset_rigid_body_index(franka_asset, "epick_end_effector"),
        }
        
        # force sensor
        sensor_pose = gymapi.Transform()
        self.gym.create_asset_force_sensor(franka_asset, self.multi_body_idx["wrist_3_link"], sensor_pose)

        # Create environments
        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)

            # Create actors and define aggregate group appropriately depending on setting
            # NOTE: franka should ALWAYS be loaded first in sim!
            if self.aggregate_mode >= 3:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            # Create franka
            self.franka_actor = self.gym.create_actor(env_ptr, franka_asset, franka_start_pose, "franka", i, 0, 0)
            self.gym.set_actor_dof_properties(env_ptr, self.franka_actor, franka_dof_props)
            self.gym.enable_actor_dof_force_sensors(env_ptr, self.franka_actor)

            if self.aggregate_mode == 2:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            count = 0
            # Create pod
            if ('cube' in self.world_params['world_model']['coll_objs']):
                cube = self.world_params['world_model']['coll_objs']['cube']
                for obj in cube.keys():
                    count += 1
                    dims = cube[obj]['dims']
                    pose = cube[obj]['pose']
                    self.add_table(dims, pose, franka_start_pose, env_ptr, i, color=[0.6, 0.6, 0.6])

            if self.aggregate_mode == 1:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            # Set urdf objects
            self._object_model_id = []
            for counter in range(len(self.object_models)):
                self._object_model_id.append(self.gym.create_actor(env_ptr, object_model_asset[counter], object_model_start_pose[counter], self.object_models[counter], i, 0, counter+1))

            if self.aggregate_mode > 0:
                self.gym.end_aggregate(env_ptr)

            # Store the created env pointers
            self.envs.append(env_ptr)
            self.frankas.append(self.franka_actor)

            # Addign friction to the suction cup
            franka_handle = 0
            suction_gripper_handle = self.gym.find_actor_rigid_body_handle(env_ptr, franka_handle, self.suction_gripper)
            suction_gripper_shape_props = self.gym.get_actor_rigid_shape_properties(env_ptr, suction_gripper_handle)
            suction_gripper_shape_props[0].friction = 1.0
            self.gym.set_actor_rigid_shape_properties(env_ptr, suction_gripper_handle, suction_gripper_shape_props)

            '''
            Camera Setup
            '''
            # Camera environment setup (Back cam)
            self.camera_handles.append([])
            self.body_states = []
            self.camera_properties_back_cam = gymapi.CameraProperties()
            self.camera_properties_back_cam.enable_tensors = True
            self.camera_properties_back_cam.horizontal_fov = 54.0
            self.camera_properties_back_cam.width = 640
            self.camera_properties_back_cam.height = 480
            camera_handle = self.gym.create_camera_sensor(env_ptr, self.camera_properties_back_cam)
            self.camera_base_link_translation = torch.tensor([-0.48, 0.05, 0.6]).to(self.device)
            local_transform = gymapi.Transform()
            local_transform.p = gymapi.Vec3(self.camera_base_link_translation[0], self.camera_base_link_translation[1], self.camera_base_link_translation[2])
            camera_rotation_x = gymapi.Quat.from_axis_angle(gymapi.Vec3(1, 0, 0), np.deg2rad(180))
            local_transform.r = camera_rotation_x

            franka_base_link_handle = self.gym.find_actor_rigid_body_handle(env_ptr, 0, self.franka_base)
            self.gym.attach_camera_to_body(camera_handle, env_ptr, franka_base_link_handle, local_transform, gymapi.FOLLOW_TRANSFORM)
            self.camera_handles[i].append(camera_handle)
            
            franka_hand_link_handle = self.gym.find_actor_rigid_body_handle(env_ptr, 0, "ee_link")
            self.camera_gripper_link_translation = []
            self.camera_properties_gripper = gymapi.CameraProperties()
            self.camera_properties_gripper.enable_tensors = True
            self.camera_properties_gripper.horizontal_fov = 150.0
            self.camera_properties_gripper.width = 1920
            self.camera_properties_gripper.height = 1080
            camera_handle_gripper = self.gym.create_camera_sensor(env_ptr, self.camera_properties_gripper)
            self.camera_gripper_link_translation.append(torch.tensor([0.0, 0, 0]).to(self.device))
            local_transform = gymapi.Transform()
            local_transform.p = gymapi.Vec3(self.camera_gripper_link_translation[0][0], 
                                            self.camera_gripper_link_translation[0][1], 
                                            self.camera_gripper_link_translation[0][2])
            self.gym.attach_camera_to_body(camera_handle_gripper, env_ptr, franka_hand_link_handle, local_transform, gymapi.FOLLOW_TRANSFORM)
            self.camera_handles[i].append(camera_handle_gripper)

            l_color = gymapi.Vec3(1, 1, 1)
            l_ambient = gymapi.Vec3(0.2, 0.2, 0.2)
            l_direction = gymapi.Vec3(-1, -1, 1)
            self.gym.set_light_parameters(self.sim, 0, l_color, l_ambient, l_direction)
            l_direction = gymapi.Vec3(-1, 1, 1)
            self.gym.set_light_parameters(self.sim, 1, l_color, l_ambient, l_direction)

        # Setup data
        self.init_data()
            
        self._init_object_model_state = []
        for counter in range(len(self.object_models)):
            self._init_object_model_state.append(torch.zeros(self.num_envs, 13, device=self.device))


    def add_table(self, table_dims, table_pose, robot_pose, env_ptr, env_id, color=[1.0,0.0,0.0]):

        table_dims = gymapi.Vec3(table_dims[0], table_dims[1], table_dims[2])

        asset_options = gymapi.AssetOptions()
        asset_options.armature = 0.001
        asset_options.fix_base_link = True
        asset_options.thickness = 0.002
        obj_color = gymapi.Vec3(color[0], color[1], color[2])
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(table_pose[0], table_pose[1], table_pose[2])
        pose.r = gymapi.Quat(table_pose[3], table_pose[4], table_pose[5], table_pose[6])
        table_asset = self.gym.create_box(self.sim, table_dims.x, table_dims.y, table_dims.z,
                                          asset_options)

        table_pose = robot_pose * pose
        table_handle = self.gym.create_actor(env_ptr, table_asset, table_pose, 'table', env_id, 0)
        self.gym.set_rigid_body_color(env_ptr, table_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, obj_color)
        table_shape_props = self.gym.get_actor_rigid_shape_properties(env_ptr, table_handle)
        table_shape_props[0].friction = 0.4
        self.gym.set_actor_rigid_shape_properties(env_ptr, table_handle, table_shape_props)

    def init_data(self):
        # Setup sim handles
        env_ptr = self.envs[0]
        franka_handle = 0
        self.handles = {
            # Franka
            "base_link": self.gym.find_actor_rigid_body_handle(env_ptr, franka_handle, self.franka_base),
            "wrist_3_link": self.gym.find_actor_rigid_body_handle(env_ptr, franka_handle, self.franka_wrist_3_link),
            "hand": self.gym.find_actor_rigid_body_handle(env_ptr, franka_handle, self.franka_hand),
        }

        for counter in range(len(self.object_models)):
            self.handles[self.object_models[counter]+"_body_handle"] = self.gym.find_actor_rigid_body_handle(self.envs[0], self._object_model_id[counter], self.object_models[counter])
            object_shape_props = self.gym.get_actor_rigid_shape_properties(env_ptr, self._object_model_id[counter])
            object_shape_props[0].friction = 0.2
            self.gym.set_actor_rigid_shape_properties(env_ptr, self._object_model_id[counter], object_shape_props)
        
        # Get total DOFs
        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs

        # Setup tensor buffers
        _actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        _dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        _rigid_body_state_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self._root_state = gymtorch.wrap_tensor(_actor_root_state_tensor).view(self.num_envs, -1, 13)
        self._dof_state = gymtorch.wrap_tensor(_dof_state_tensor).view(self.num_envs, -1, 2)
        self._rigid_body_state = gymtorch.wrap_tensor(_rigid_body_state_tensor).view(self.num_envs, -1, 13)
        self._q = self._dof_state[..., 0]
        self._qd = self._dof_state[..., 1]
        self._eef_state = self._rigid_body_state[:, self.handles["hand"], :]
        self._base_link = self._rigid_body_state[:, self.handles["base_link"], :]
        self._wrist_3_link = self._rigid_body_state[:, self.handles["wrist_3_link"], :]
        _jacobian = self.gym.acquire_jacobian_tensor(self.sim, "franka")
        jacobian = gymtorch.wrap_tensor(_jacobian)
        hand_joint_index = self.gym.get_actor_joint_dict(env_ptr, franka_handle)['ee_fixed_joint']
        self._j_eef = jacobian[:, hand_joint_index, :, :6]
        _massmatrix = self.gym.acquire_mass_matrix_tensor(self.sim, "franka")
        mm = gymtorch.wrap_tensor(_massmatrix)
        self._mm = mm[:, :6, :6]

        self._object_model_state = []
        for counter in range(len(self.object_models)):
            self._object_model_state.append(self._root_state[:, self._object_model_id[counter], :])
        
        # Initialize actions
        self._pos_control = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
        self._effort_control = torch.zeros_like(self._pos_control)
        # Initialize control
        self._arm_control = self._effort_control[:, :]

        # Initialize indices    ------ > self.num_envs * num of actors
        self._global_indices = torch.arange(self.num_envs * (28 + len(self.object_models)), dtype=torch.int32,
                                           device=self.device).view(self.num_envs, -1)
        
        '''
        camera intrinsics for back cam and gripper cam
        '''
        # cam_vinv_back_cam = torch.inverse((torch.tensor(self.gym.get_camera_view_matrix(self.sim, self.envs[i], self.camera_handles[i][0])))).to(self.device)
        cam_proj_back_cam = torch.tensor(self.gym.get_camera_proj_matrix(self.sim, self.envs[0], self.camera_handles[0][0]), device=self.device)
        self.width_back_cam = self.camera_properties_back_cam.width
        self.height_back_cam = self.camera_properties_back_cam.height
        self.fx_back_cam = self.width_back_cam/(2/cam_proj_back_cam[0, 0])
        self.fy_back_cam = self.height_back_cam/(2/cam_proj_back_cam[1, 1])
        self.cx_back_cam = self.width_back_cam/2
        self.cy_back_cam = self.height_back_cam/2
        self.camera_intrinsics_back_cam = CameraIntrinsics(frame="camera_back", fx=self.fx_back_cam, fy=self.fy_back_cam, 
                                                           cx=self.cx_back_cam, cy=self.cy_back_cam, skew=0.0, 
                                                           height=self.height_back_cam, width=self.width_back_cam)
        self.suction_score_object = calcualte_suction_score(self.camera_intrinsics_back_cam)
        self.dexnet_object = dexnet3(self.camera_intrinsics_back_cam)
        self.dexnet_object.load_dexnet_model()

        # cam_vinv_gripper = torch.inverse((torch.tensor(self.gym.get_camera_view_matrix(self.sim, self.envs[i], self.camera_handles[i][0])))).to(self.device)
        cam_proj_gripper = torch.tensor(self.gym.get_camera_proj_matrix(self.sim, self.envs[0], self.camera_handles[0][1]), device=self.device)
        self.width_gripper = self.camera_properties_gripper.width
        self.height_gripper = self.camera_properties_gripper.height
        self.fx_gripper = self.width_gripper/(2/cam_proj_gripper[0, 0])
        self.fy_gripper = self.height_gripper/(2/cam_proj_gripper[1, 1])
        self.cx_gripper = self.width_gripper/2
        self.cy_gripper = self.height_gripper/2
        self.camera_intrinsics_gripper = CameraIntrinsics(frame="camera_gripper", fx=self.fx_gripper, fy=self.fy_gripper, 
                                                           cx=self.cx_gripper, cy=self.cy_gripper, skew=0.0, 
                                                           height=self.height_gripper, width=self.width_gripper)
        self.suction_score_object_gripper = calcualte_suction_score(self.camera_intrinsics_gripper)
        self.force_object = calcualte_force()

        '''
        Transformation for static links
        '''
        poses_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.curr_poses = gymtorch.wrap_tensor(poses_tensor).view(self.num_envs, -1, 13)
        # Transformation of base_link from world coordiante frame (wb)
        rotation_matrix_base_link = euler_angles_to_matrix(torch.tensor([180, 0, 0]).to(self.device), "XYZ", degrees=True)
        translation_base_link = torch.tensor([0, 0, 2.020]).to(self.device)
        self.T_base_link = transformation_matrix(rotation_matrix_base_link, translation_base_link)
        # Transformation for camera (wc --> wb*bc)
        rotation_matrix_camera_offset = euler_angles_to_matrix(torch.tensor([180, 0, 0]).to(self.device), "XYZ", degrees=True)
        T_base_link_to_camera = transformation_matrix(rotation_matrix_camera_offset, self.camera_base_link_translation)
        self.T_world_to_camera_link = torch.matmul(self.T_base_link, T_base_link_to_camera)


    def _update_states(self):
        self.states.update({
            # Franka
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
    
    def reset_init_arm_pose(self, env_ids):
        # pos = torch.tensor(np.random.uniform(low=-6.2832, high=6.2832, size=(6,))).to(self.device).type(torch.float)
        # pos = tensor_clamp(pos.unsqueeze(0), self.franka_dof_lower_limits.unsqueeze(0), self.franka_dof_upper_limits)
        pos = tensor_clamp(self.franka_default_dof_pos.unsqueeze(0), self.franka_dof_lower_limits.unsqueeze(0), self.franka_dof_upper_limits)
        pos = pos.repeat(len(env_ids), 1)

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
                                                        gymtorch.unwrap_tensor(self._pos_control),
                                                        gymtorch.unwrap_tensor(multi_env_ids_int32),
                                                        len(multi_env_ids_int32))
        self.gym.set_dof_actuation_force_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(self._effort_control),
                                                        gymtorch.unwrap_tensor(multi_env_ids_int32),
                                                        len(multi_env_ids_int32))
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self._dof_state),
                                              gymtorch.unwrap_tensor(multi_env_ids_int32),
                                              len(multi_env_ids_int32))
        
        # reinitializing the variables
        for env_id in env_ids:
            self.action_contrib[env_id] = 1
            self.force_encounter[env_id] = 0
            self.frame_count_contact_object[env_id] = 0
            self.frame_count[env_id] = 0
            self.free_envs_list[env_id] = torch.tensor(1)
        
        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0

    def reset_object_pose(self, env_ids):
        for counter in range(len(self.object_models)):
            self._reset_init_object_state(object=self.object_models[counter], env_ids=env_ids, offset=[0.08, counter/6-0.16], check_valid=True)
            self._object_model_state[counter][env_ids] = self._init_object_model_state[counter][env_ids]
        
        multi_env_ids_cubes_int32 = self._global_indices[env_ids, -len(self.object_models):].flatten()
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self._root_state),
            gymtorch.unwrap_tensor(multi_env_ids_cubes_int32), len(multi_env_ids_cubes_int32))
        
        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        
        # reinitializing the variables
        for env_id in env_ids:
            self.action_contrib[env_id] = 1
            self.force_encounter[env_id] = 0
            self.frame_count_contact_object[env_id] = 0
            self.env_reset_id_env[env_id] = 1

    def reset_idx(self, env_ids):
        self.reset_init_arm_pose(env_ids)
        # Update cube states
        self.reset_object_pose(env_ids)

    def compute_observations(self):
        self._refresh()
        obs = ["eef_pos", "eef_quat"]
        obs += ["q_gripper"] if self.control_type == "osc" else ["q"]
        self.obs_buf = torch.cat([self.states[ob] for ob in obs], dim=-1)

        return self.obs_buf
    
    def _reset_init_object_state(self, object, env_ids, offset, check_valid=True):
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
        # If env_ids is None, we reset all the envs
        if env_ids is None:
            env_ids = torch.arange(start=0, end=self.num_envs, device=self.device, dtype=torch.long)

        # Initialize buffer to hold sampled values
        num_resets = len(env_ids)
        sampled_object_state = torch.zeros(num_resets, 13, device=self.device)

        # Get correct references depending on which one was selected
        for i in range(len(self.object_models)):
            if object == self.object_models[i]:
                this_object_state_all = self._init_object_model_state[i]

        # Sampling is "centered" around middle of table
        centered_object_xy_state = torch.tensor(np.array([0.52, 0.0]), device=self.device, dtype=torch.float32)
        
        # For variable height
        # firs row H 1.7, second row G 1.5, third row F 1.35, fourth row E 1.2
        # sampling height at which the cube will be dropped
        sampled_object_state[:, 2] = torch.tensor(np.array([1.35]), device=self.device, dtype=torch.float32)

        # Initialize rotation, which is no rotation (quat w = 1)
        sampled_object_state[:, 6] = 1.0

        # If we're verifying valid sampling, we need to check and re-sample if any are not collision-free
        # We use a simple heuristic of checking based on cubes' radius to determine if a collision would occur
        if check_valid:
            success = False
            # Indexes corresponding to envs we're still actively sampling for
            active_idx = torch.arange(num_resets, device=self.device)
            num_active_idx = len(active_idx)

            # Sample x y values
            sampled_object_state[active_idx, :2] = centered_object_xy_state
            # Setting the X axis value for the object
            # Set the offset for cube A on both the environment
            offset_xy = torch.zeros(1, 2).to(self.device)
            # offset_x_axis = torch.tensor(offset_xy).to(device=self.device)
            offset_x_axis = offset_xy +  torch.tensor([offset]).to(self.device)
            # Sometimes it only tries to sample for one environment, this might be due to frequency mismatch of the env (but not sure)
            sampled_object_state[active_idx, :2] += offset_x_axis
                    
        else:
            # We just directly sample if check_valid variable is False
            sampled_object_state[:, :2] = centered_object_xy_state.unsqueeze(0) + \
                                              2.0 * self.start_position_noise * (
                                                      torch.rand(num_resets, 2, device=self.device) - 0.5)

        # Lastly, set these sampled values as the new init state
        # print('env id', env_ids, 'pose', sampled_cube_state)
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
                (self.franka_default_dof_pos[:6] - q + np.pi) % (2 * np.pi) - np.pi)
        u_null[:, self.num_franka_dofs:] *= 0
        u_null = self._mm @ u_null.unsqueeze(-1)
        u += (torch.eye((6), device=self.device).unsqueeze(0) - torch.transpose(self._j_eef, 1, 2) @ j_eef_inv) @ u_null

        # Clip the values to be within valid effort range
        u = tensor_clamp(u.squeeze(-1),
                         -self._franka_effort_limits[:6].unsqueeze(0), self._franka_effort_limits[:6].unsqueeze(0))

        return u
    
    def to_pose_ik(self, dpose):
        '''print('state pose shape', self.states["wrist_3_link"].shape)
        pos_err = goal_pose - self.states["wrist_3_link"][:, :3]
        orn_err = self.orientation_error(goal_orientation)'''
        dpose = dpose.unsqueeze(-1)
        u = self._compute_ik(dpose)
        self._q += u
        self._qd = torch.zeros_like(self._qd)

    def _compute_ik(self, dpose):
        damping = 0.05
        j_eef_T = torch.transpose(self._j_eef, 1, 2)
        lmbda = torch.eye(6, device=self.device) * (damping ** 2)
        u = (j_eef_T @ torch.inverse(self._j_eef @ j_eef_T + lmbda) @ dpose).view(self.num_envs, 6)
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
        print(env_ids)
        pos = torch.zeros(0, self.num_franka_dofs).to(self.device)
        for _ in env_ids:
            joint_poses_list = torch.load(f"{cur_path}/../../misc/joint_poses.pt")
            temp_pos = joint_poses_list[torch.randint(0, len(joint_poses_list), (1,))[0]].to(self.device)
            temp_pos = torch.reshape(temp_pos, (1, len(temp_pos)))
            temp_pos = torch.cat((temp_pos, torch.tensor([[0]]).to(self.device)), dim=1)
            pos = torch.cat([pos, temp_pos])
        # pos = tensor_clamp(pos.unsqueeze(0), self.franka_dof_lower_limits.unsqueeze(0), self.franka_dof_upper_limits)
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
                                                        gymtorch.unwrap_tensor(self._pos_control),
                                                        gymtorch.unwrap_tensor(multi_env_ids_int32),
                                                        len(multi_env_ids_int32))
        self.gym.set_dof_actuation_force_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(self._effort_control),
                                                        gymtorch.unwrap_tensor(multi_env_ids_int32),
                                                        len(multi_env_ids_int32))
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self._dof_state),
                                              gymtorch.unwrap_tensor(multi_env_ids_int32),
                                              len(multi_env_ids_int32))

    def pre_physics_step(self, actions):
        '''
        Camera access in the pre physics step to compute the force using suction cup deformation score
        '''
        # communicate physics to graphics system
        self.gym.step_graphics(self.sim)
        # render the camera sensors
        self.gym.render_all_camera_sensors(self.sim)

        object_id = torch.tensor(1).to(self.device)
        total_objects = len(self.object_models)
        '''
        Commands to the arm for eef control
        '''
        
        self.actions = torch.zeros(0, 7)
        # Before each loop this will track all the environments where a condition for reset has been called
        env_list_reset = torch.tensor([])
        env_list_reset_dexnet = torch.tensor([])
        for env_count in range(self.num_envs):
            # if(self.frame_count[env_count] == 10 and self.bootup_reset[env_count] == 1):
            #     env_list_reset = torch.cat((env_list_reset, torch.tensor([env_count])), axis=0)
            #     self.bootup_reset[env_count] = 0

            # check if the environment returned from reset and the frame for that enviornment is 30 or not
            # 30 frames is for cooldown period at the start for the simualtor to settle down
            if((self.free_envs_list[env_count] == 1) and self.frame_count[env_count] == self.cooldown_frames):
                '''
                DexNet 3.0
                '''
                camera_env = env_count
                rgb_camera_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, self.envs[camera_env], self.camera_handles[camera_env][0], gymapi.IMAGE_COLOR)
                torch_rgb_tensor = gymtorch.wrap_tensor(rgb_camera_tensor)
                rgb_image = torch_rgb_tensor.to(self.device)
                rgb_image_copy = torch.reshape(rgb_image, (rgb_image.shape[0], -1, 4))[..., :3]

                depth_camera_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, self.envs[camera_env], self.camera_handles[camera_env][0], gymapi.IMAGE_DEPTH)
                torch_depth_tensor = gymtorch.wrap_tensor(depth_camera_tensor)
                depth_image = torch_depth_tensor.to(self.device)
                depth_image = -depth_image
                
                mask_camera_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, self.envs[camera_env], self.camera_handles[camera_env][0], gymapi.IMAGE_SEGMENTATION)
                torch_mask_tensor = gymtorch.wrap_tensor(mask_camera_tensor)
                segmask = torch_mask_tensor.to(self.device)
                segmask_dexnet = segmask.clone().detach()
                segmask_numpy = np.zeros_like(segmask_dexnet.cpu().numpy().astype(np.uint8))
                segmask_numpy_temp = np.zeros_like(segmask_dexnet.cpu().numpy().astype(np.uint8))
                segmask_numpy_temp[segmask_dexnet.cpu().numpy().astype(np.uint8) == object_id.cpu().numpy()] = 1
                segmask_numpy[segmask_dexnet.cpu().numpy().astype(np.uint8) == object_id.cpu().numpy()] = 255
                segmask_dexnet = BinaryImage(segmask_numpy, frame=self.camera_intrinsics_back_cam.frame)
                
                depth_image_dexnet = depth_image.clone().detach()
                depth_image_dexnet -= 0.5
                noise_image = torch.normal(0, 0.0009, size=depth_image_dexnet.size()).to(self.device)
                depth_image_dexnet = depth_image_dexnet + noise_image
                depth_numpy = depth_image_dexnet.cpu().numpy()
                depth_numpy_temp = depth_numpy*segmask_numpy_temp
                depth_numpy_temp[depth_numpy_temp == 0] = 0.75
                depth_img_dexnet = DepthImage(depth_numpy_temp, frame=self.camera_intrinsics_back_cam.frame)

                action , self.grasps_and_predictions = self.dexnet_object.inference(depth_img_dexnet, segmask_dexnet, None)
                print(f"Quality is {action.q_value} grasp location is {action.grasp.center.x}, {action.grasp.center.y} for environment {env_count}")
                
                self.suction_deformation_score_temp = torch.Tensor()
                self.xyz_point_temp = torch.empty((0,3))
                self.grasp_angle_temp = torch.empty((0,3))
                self.force_SI_temp = torch.Tensor()
                for i in range(1):
                    stack_grasp_points = self.grasps_and_predictions[i][0]
                    # dexnet score
                    # print(self.grasps_and_predictions[i][1])
                    
                    depth_image_suction = depth_image.clone().detach()
                    suction_deformation_score, xyz_point, grasp_angle = self.suction_score_object.calculator(depth_image_suction, segmask, rgb_image_copy, stack_grasp_points, object_id)
                    self.suction_deformation_score_temp = torch.cat((self.suction_deformation_score_temp, torch.tensor([suction_deformation_score]))).type(torch.float)
                    self.xyz_point_temp = torch.cat([self.xyz_point_temp, xyz_point.unsqueeze(0)], dim=0)
                    self.grasp_angle_temp = torch.cat([self.grasp_angle_temp, grasp_angle.unsqueeze(0)], dim=0)
                    self.object_coordiante_camera = xyz_point.clone().detach()
                    if(suction_deformation_score > 0.2):
                        force_SI = self.force_object.regression(suction_deformation_score)
                    else:
                        force_SI = torch.tensor(0).to(self.device)
                    
                    self.force_SI_temp = torch.cat((self.force_SI_temp, torch.tensor([force_SI])))
                    # pixel coordinates
                    # dexnet_coordinates = np.array([grasp_data.grasp.center.x, grasp_data.grasp.center.y])
                    # print(f"Quality is {grasp_data.q_value} grasp location is {grasp_data.grasp.center.x}, {grasp_data.grasp.center.y}")
                    # print(f"suction deforamtion score --> {suction_deformation_score}, Force along z axis --> {force_SI}")
                
                self.suction_deformation_score_env[env_count] = self.suction_deformation_score_temp
                self.grasp_angle_env[env_count] = self.grasp_angle_temp
                self.force_SI_env[env_count] = self.force_SI_temp
                self.xyz_point_env[env_count] = self.xyz_point_temp
                self.free_envs_list[env_count] = torch.tensor(0)

            '''
            How to pop out form tensor stack
            def pop():
                item = self.stack[-1]  # Get the last element in the tensor
                self.stack = self.stack[:-1]  # Remove the last element from the tensor
                return item
            '''
            
            self.action_env = torch.tensor([[0, 0, 0, 0, 0, 0, 1]], dtype=torch.float)
            if((self.env_reset_id_env[env_count] == 1) and (self.frame_count[env_count] >= self.cooldown_frames)):
                self.env_reset_id_env[env_count] = torch.tensor(0)
                self.action_env = torch.tensor([[0, 0, 0, 0, 0, 0, 1]], dtype=torch.float)
                # print(self.frame_count[env_count], self.grasp_angle_env[env_count], len(self.grasp_angle_env[env_count]), env_count)
                if(len(self.grasp_angle_env[env_count]) != 0):
                    self.suction_deformation_score[env_count] = self.suction_deformation_score_env[env_count][0]
                    self.suction_deformation_score_env[env_count] = self.suction_deformation_score_env[env_count][1:]
                    self.grasp_angle[env_count] = self.grasp_angle_env[env_count][0]
                    self.grasp_angle_env[env_count] = self.grasp_angle_env[env_count][1:]
                    self.xyz_point[env_count] = self.xyz_point_env[env_count][0]
                    self.xyz_point_env[env_count] = self.xyz_point_env[env_count][1:]
                    self.force_SI[env_count] = self.force_SI_env[env_count][0]
                    self.force_SI_env[env_count] = self.force_SI_env[env_count][1:]
                    # env_list_reset = torch.cat((env_list_reset, torch.tensor([env_count])), axis=0)
                    env_list_reset_dexnet = torch.cat((env_list_reset_dexnet, torch.tensor([env_count])), axis=0)
                else:
                    self.reset_idx(torch.tensor([env_count]).to(self.device).type(torch.long))  
                if(torch.all(self.xyz_point[env_count]) == torch.tensor(0.)):
                    env_list_reset = torch.cat((env_list_reset, torch.tensor([env_count])), axis=0)
            elif(self.env_reset_id_env[env_count] == 0 and self.frame_count[env_count] > self.cooldown_frames):
                # Transformation for object from camera (wo --> wc*co)
                rotation_matrix_camera_to_object = euler_angles_to_matrix(torch.tensor([0, -self.grasp_angle[env_count][1], self.grasp_angle[env_count][0]]).to(self.device), "XYZ", degrees=False)
                T_camera_to_object = transformation_matrix(rotation_matrix_camera_to_object, self.xyz_point[env_count])
                # Transformation from base link to object
                T_world_to_object = torch.matmul(self.T_world_to_camera_link, T_camera_to_object)
                # Transformation for pre grasp pose (wp --> wo*op)
                rotation_matrix_pre_grasp_pose = euler_angles_to_matrix(torch.tensor([0, 0, 0]).to(self.device), "XYZ", degrees=True)
                translation_pre_grasp_pose = torch.tensor([-0.25, 0, 0]).to(self.device)
                T_pre_grasp_pose = transformation_matrix(rotation_matrix_pre_grasp_pose, translation_pre_grasp_pose)
                # Transformation of object with base link to pre grasp pose
                T_world_to_pre_grasp_pose = torch.matmul(T_world_to_object, T_pre_grasp_pose)
                # force sensor update
                self.gym.refresh_force_sensor_tensor(self.sim)
                try:
                    _fsdata = self.gym.acquire_force_sensor_tensor(self.sim)
                    fsdata = gymtorch.wrap_tensor(_fsdata)
                    self.force_pre_physics = -fsdata[env_count][2].detach().cpu().numpy()
                except:
                    print("error in force sensor")
                # Transformation for pre grasp pose (pe --> inv(we)*wp --> ew*wp)
                rotation_matrix_ee_pose = quaternion_to_matrix(self.curr_poses[env_count][self.multi_body_idx['ee_link']][3:7])
                translation_ee_pose = self.curr_poses[env_count][self.multi_body_idx['wrist_3_link']][:3]
                T_world_to_ee_pose = transformation_matrix(rotation_matrix_ee_pose, translation_ee_pose)
                T_ee_pose_to_pre_grasp_pose = torch.matmul(torch.inverse(T_world_to_ee_pose), T_world_to_pre_grasp_pose)
                # Orientation error
                action_orientation = matrix_to_euler_angles(T_ee_pose_to_pre_grasp_pose[:3, :3], "XYZ")
            
                self.distance = torch.tensor([1, 1, 1])
                if(self.action_contrib[env_count] == torch.tensor(1)):
                    pose_factor, ori_factor = 1., 0.3
                    self.action_env = torch.tensor([[pose_factor*T_ee_pose_to_pre_grasp_pose[0][3], pose_factor*T_ee_pose_to_pre_grasp_pose[1][3],
                                                    pose_factor*T_ee_pose_to_pre_grasp_pose[2][3], ori_factor*action_orientation[0],
                                                    ori_factor*action_orientation[1], ori_factor*action_orientation[2], 1]], dtype=torch.float)
                else:
                    # Transformation for grasp pose (wg --> wo*og)
                    rotation_matrix_grasp_pose = euler_angles_to_matrix(torch.tensor([0, -self.grasp_angle[env_count][1], self.grasp_angle[env_count][0]]).to(self.device), "XYZ", degrees=False).type(torch.float)
                    translation_grasp_pose = torch.tensor([0.1, 0, 0]).to(self.device).type(torch.float)
                    translation_grasp_pose = torch.matmul(rotation_matrix_grasp_pose, translation_grasp_pose)
                    
                    start_point = T_world_to_pre_grasp_pose[:3, 3]
                    end_point = T_world_to_object[:3, 3]
                    current_point = T_world_to_ee_pose[:3, 3]
                    v = torch.tensor([end_point[0] - start_point[0], end_point[1] - start_point[1], end_point[2] - start_point[2]])
                    # Calculate the vector connecting p1 to p
                    w = torch.tensor([current_point[0] - start_point[0], current_point[1] - start_point[1], current_point[2] - start_point[2]])
                    # Calculate the projection of w onto v
                    t = (w[0]*v[0] + w[1]*v[1] + w[2]*v[2])/(v[0]**2 + v[1]**2 + v[2]**2)
                    # Calculate the closest point on the line to p
                    q = torch.tensor([start_point[0] + t*v[0], start_point[1] + t*v[1], start_point[2] + t*v[2]]).to(self.device)
                    # Find the distance between p and q
                    self.distance = current_point - q
                    self.speed = 0.1
                    self.action_env = torch.tensor([[0.15, 
                                                     translation_grasp_pose[1]-self.distance[1]*100*self.speed, 
                                                     translation_grasp_pose[2]-self.distance[2]*100*self.speed, 
                                                     self.speed*100*action_orientation[0], 
                                                     self.speed*100*action_orientation[1], 
                                                     self.speed*100*action_orientation[2], 1]], dtype=torch.float)
                    
                if(self.frame_count[env_count] > torch.tensor(self.cooldown_frames) and self.frame_count_contact_object[env_count] == torch.tensor(0)):
                    if((torch.max(torch.abs(self.action_env[0][:6]))) <= 0.001):
                        self.action_contrib[env_count] = 0
                    if(self.force_pre_physics > torch.max(torch.tensor([2, self.force_SI[env_count]])) and self.action_contrib[env_count] == 0):
                        self.force_encounter[env_count] = 1
                        '''
                        Gripper camera
                        '''
                        rgb_camera_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, self.envs[env_count], self.camera_handles[env_count][1], gymapi.IMAGE_COLOR)
                        torch_rgb_tensor = gymtorch.wrap_tensor(rgb_camera_tensor)
                        rgb_image = torch_rgb_tensor.to(self.device)
                        rgb_image_copy_gripper = torch.reshape(rgb_image, (rgb_image.shape[0], -1, 4))[..., :3]
                        rgb_image_copy_gripper = rgb_image_copy_gripper.clone().detach()
                        
                        mask_camera_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, self.envs[env_count], self.camera_handles[env_count][1], gymapi.IMAGE_SEGMENTATION)
                        torch_mask_tensor = gymtorch.wrap_tensor(mask_camera_tensor)
                        segmask_gripper = torch_mask_tensor.to(self.device)
                        segmask_gripper = segmask_gripper.clone().detach()

                        depth_camera_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, self.envs[env_count], self.camera_handles[env_count][1], gymapi.IMAGE_DEPTH)
                        torch_depth_tensor = gymtorch.wrap_tensor(depth_camera_tensor)
                        depth_image = torch_depth_tensor.to(self.device)
                        depth_image = -depth_image
                        depth_numpy_gripper = depth_image.clone().detach()

                        score_gripper, _, _ = self.suction_score_object_gripper.calculator(depth_numpy_gripper, segmask_gripper, rgb_image_copy_gripper, None, object_id)
                        print(env_count, " force: ", self.force_pre_physics)
                        print(env_count, " suction gripper ", score_gripper)
                        self.frame_count_contact_object[env_count] = 1
                    elif(self.force_pre_physics > torch.tensor(10) and self.action_contrib[env_count] == 1):
                        print(env_count, " force due to collision: ", self.force_pre_physics)
                        env_list_reset = torch.cat((env_list_reset, torch.tensor([env_count])), axis=0)
                elif(self.frame_count_contact_object[env_count] == torch.tensor(1)):
                    env_list_reset = torch.cat((env_list_reset, torch.tensor([env_count])), axis=0)

            if(self.frame_count[env_count] <= torch.tensor(self.cooldown_frames)):
                self.action_env = torch.tensor([[0, 0, 0, 0, 0, 0, 1]], dtype=torch.float)

            self.actions = torch.cat([self.actions, self.action_env])
            self.frame_count[env_count] += torch.tensor(1)
        if(len(env_list_reset_dexnet) != 0):
            self.reset_pre_grasp_pose(env_list_reset_dexnet.to(self.device).type(torch.long))
        if(len(env_list_reset) != 0):
            self.reset_pre_grasp_pose(env_list_reset.to(self.device).type(torch.long))
            # self.reset_init_arm_pose(env_list_reset.to(self.device).type(torch.long))
            self.reset_object_pose(env_list_reset.to(self.device).type(torch.long))
        # if(self.force_encounter == 1):
        #     actions = torch.tensor(self.num_envs * [[-0.1, T_ee_pose_to_pre_grasp_pose[1][3], T_ee_pose_to_pre_grasp_pose[2][3], 0.1*action_orientation[0], 0.1*action_orientation[1], 0.1*action_orientation[2], 1]], dtype=torch.float)
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
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self._pos_control))
        self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self._effort_control))

    def post_physics_step(self):
        self.progress_buf += 1
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_pre_grasp_pose(env_ids)
            self.reset_object_pose(env_ids)

        self.compute_observations()
        # Compute resets
        self.reset_buf = torch.where((self.progress_buf >= self.max_episode_length - 1), torch.ones_like(self.reset_buf), self.reset_buf)