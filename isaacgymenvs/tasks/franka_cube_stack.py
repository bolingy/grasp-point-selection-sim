# Copyright (c) 2021-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

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
import imageio.v3 as iio
import matplotlib.cm as cm
import open3d as o3d
import cv2

from suction_cup_modelling.suction_score_calcualtor import calcualte_suction_score
from suction_cup_modelling.force_calculator import calcualte_force
from suction_cup_modelling.suction_score_calcualtor import CameraInfo

# import sys
# sys.path.insert(0, '/home/soofiyan_ws/Documents/Issac_gym_ws/grasp-point-selection-sim/')

from gqcnn.examples.policy_for_training import dexnet3

from autolab_core import (YamlConfig, Logger, BinaryImage, CameraIntrinsics, ColorImage, DepthImage, RgbdImage)
import assets.urdf_models.models_data as md

from scipy.spatial.transform import Rotation as R

from homogeneous_trasnformation_and_conversion.rotation_conversions import *
# This function is used to convert axis angle to quartenions
@torch.jit.script
def axisangle2quat(vec, eps=1e-6):
    """
    Converts scaled axis-angle to quat.
    Args:
        vec (tensor): (..., 3) tensor where final dim is (ax,ay,az) axis-angle exponential coordinates
        eps (float): Stability value below which small values will be mapped to 0

    Returns:
        tensor: (..., 4) tensor where final dim is (x,y,z,w) vec4 float quaternion
    """
    # type: (Tensor, float) -> Tensor
    # store input shape and reshape
    input_shape = vec.shape[:-1]
    vec = vec.reshape(-1, 3)

    # Grab angle
    angle = torch.norm(vec, dim=-1, keepdim=True)

    # Create return array
    quat = torch.zeros(torch.prod(torch.tensor(input_shape)), 4, device=vec.device)
    quat[:, 3] = 1.0

    # Grab indexes where angle is not zero an convert the input to its quaternion form
    idx = angle.reshape(-1) > eps
    quat[idx, :] = torch.cat([
        vec[idx, :] * torch.sin(angle[idx, :] / 2.0) / angle[idx, :],
        torch.cos(angle[idx, :] / 2.0)
    ], dim=-1)

    # Reshape and return output
    quat = quat.reshape(list(input_shape) + [4, ])
    return quat

def euler_from_quaternion(x, y, z, w):
    """
    Convert a quaternion into euler angles (roll, pitch, yaw)
    roll is rotation around x in radians (counterclockwise)
    pitch is rotation around y in radians (counterclockwise)
    yaw is rotation around z in radians (counterclockwise)
    """
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = math.atan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = math.asin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = math.atan2(t3, t4)

    return roll_x, pitch_y, yaw_z # in radians


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

        # Create dicts to pass to reward function
        self.reward_settings = {
            "r_dist_scale": self.cfg["env"]["distRewardScale"],
            "r_lift_scale": self.cfg["env"]["liftRewardScale"],
            "r_align_scale": self.cfg["env"]["alignRewardScale"],
            "r_stack_scale": self.cfg["env"]["stackRewardScale"],
        }

        # Controller type
        self.control_type = self.cfg["env"]["controlType"]
        assert self.control_type in {"osc", "joint_tor"},\
            "Invalid control type specified. Must be one of: {osc, joint_tor}"

        # dimensions
        # obs include: cubeA_pose (7) + cubeB_pos (3) + eef_pose (7) + q_gripper (2)
        self.cfg["env"]["numObservations"] = 19 if self.control_type == "osc" else 27
        # actions include: delta EEF if OSC (6) or joint torques (7) + bool gripper (1)
        self.cfg["env"]["numActions"] = 7 if self.control_type == "osc" else 8

        # Values to be filled in at runtime
        self.states = {}                        # will be dict filled with relevant states to use for reward calculation
        self.handles = {}                       # will be dict mapping names to relevant sim handles
        self.num_dofs = None                    # Total number of DOFs per env
        self.actions = None                     # Current actions to be deployed
        self._init_cubeA_state = None           # Initial state of cubeA for the current env
        self._init_cubeB_state = None           # Initial state of cubeB for the current env
        self._cubeA_state = None                # Current state of cubeA for the current env
        self._cubeB_state = None                # Current state of cubeB for the current env
        self._cubeA_id = None                   # Actor ID corresponding to cubeA for a given env
        self._cubeB_id = None                   # Actor ID corresponding to cubeB for a given env

        self._init_cylinder_state = None
        self._cylinder_state = None
        self._cylinder_id = None

        self.models_lib = md.model_lib()

        # Tensor placeholders
        self._root_state = None             # State of root body        (n_envs, 13)
        self._dof_state = None  # State of all joints       (n_envs, n_dof)
        self._q = None  # Joint positions           (n_envs, n_dof)
        self._qd = None                     # Joint velocities          (n_envs, n_dof)
        self._rigid_body_state = None  # State of all rigid bodies             (n_envs, n_bodies, 13)
        self._contact_forces = None     # Contact forces in sim
        self._eef_state = None  # end effector state (at grasping point)
        #self._eef_lf_state = None  # end effector state (at left fingertip)
        #self._eef_rf_state = None  # end effector state (at left fingertip)
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

        self.action_contrib = 1

        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../configs")+'/collision_primitives_3d.yml') as file:
            self.world_params = yaml.load(file, Loader=yaml.FullLoader)

        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)

        # Franka defaults
        # self.franka_default_dof_pos = to_torch(
        #     [0.06, -2.5, 2.53, 0.58, 1.67, 1.74], device=self.device
        # )
        self.franka_default_dof_pos = to_torch(
            [0.06, -2.5, 2.53, 0, 1.57, 0], device=self.device
        )

        # OSC Gains
        self.kp = to_torch([150.] * 6, device=self.device)
        self.kd = 2 * torch.sqrt(self.kp)
        self.kp_null = to_torch([10.] * self.num_franka_dofs, device=self.device)
        self.kd_null = 2 * torch.sqrt(self.kp_null)
        #self.cmd_limit = None                   # filled in later

        # Set control limits
        self.cmd_limit = to_torch([0.1, 0.1, 0.1, 0.5, 0.5, 0.5], device=self.device).unsqueeze(0) if \
        self.control_type == "osc" else self._franka_effort_limits[:7].unsqueeze(0)

        # Reset all environments
        self.reset_idx(torch.arange(self.num_envs, device=self.device))

        # Refresh tensors
        self._refresh()

        self.frame_count = 0
        self.fig = plt.figure()

        self.current_directory = os.getcwd()

        self.object_coordiante_camera = torch.tensor([0, 0, 0])

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
        franka_asset_file = "urdf/Aurmar_description/robots/robot_storm.urdf"#"urdf/franka_description/robots/franka_panda_gripper.urdf"
        '''
        if "asset" in self.cfg["env"]:
            asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.cfg["env"]["asset"].get("assetRoot", asset_root))
            franka_asset_file = self.cfg["env"]["asset"].get("assetFileNameFranka", franka_asset_file)
        '''
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

        franka_dof_stiffness = to_torch([0, 0, 0, 0, 0, 0], dtype=torch.float, device=self.device)
        franka_dof_damping = to_torch([0., 0., 0., 0., 0., 0.], dtype=torch.float, device=self.device)

        # Create table asset
        table_pos = [0.0, 0.0, 1.0]
        table_thickness = 0.05
        table_opts = gymapi.AssetOptions()
        table_opts.fix_base_link = True

        # Create table stand asset
        '''table_stand_height = 0.1
        table_stand_pos = [-0.5, 0.0, 1.0 + table_thickness / 2 + table_stand_height / 2]
        table_stand_opts = gymapi.AssetOptions()
        table_stand_opts.fix_base_link = True
        table_stand_asset = self.gym.create_box(self.sim, *[0.2, 0.2, table_stand_height], table_opts)'''

        self.cubeA_size = 0.050
        self.cubeB_size = 0.070
        self.cylinder_size = 0.011

        # Create cubeA asset
        cubeA_opts = gymapi.AssetOptions()
        cubeA_asset = self.gym.create_box(self.sim, *([self.cubeA_size] * 3), cubeA_opts)
        cubeA_color = gymapi.Vec3(0.6, 0.1, 0.0)

        # Create cubeB asset
        cubeB_opts = gymapi.AssetOptions()
        cubeB_asset = self.gym.create_box(self.sim, *([self.cubeB_size] * 3), cubeB_opts)
        cubeB_color = gymapi.Vec3(0.0, 0.4, 0.1)


        asset_options = gymapi.AssetOptions()
        asset_options.flip_visual_attachments = True
        asset_options.fix_base_link = False
        asset_options.collapse_fixed_joints = True
        asset_options.disable_gravity = False
        asset_options.thickness = 0.001
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_EFFORT
        asset_options.use_mesh_materials = True
        cylinder_asset_file = "urdf/Aurmar_description/objects/cylinder.urdf"
        # load cylinder asset
        cylinder_asset = self.gym.load_asset(self.sim, asset_root, cylinder_asset_file, asset_options)
        cylinder_color = gymapi.Vec3(0.5, 0.1, 0.5)

        self.object_models = ["sugar_box"]
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
        #self.franka_dof_speed_scales[[7, 8]] = 0.1
        #franka_dof_props['effort'][7] = 200
        #franka_dof_props['effort'][8] = 200

        # Define start pose for franka
        franka_start_pose = gymapi.Transform()
        franka_start_pose.p = gymapi.Vec3(0, 0, 2.020) #gymapi.Vec3(-0.45, 0.0, 1.0 + table_thickness / 2 + table_stand_height)
        from scipy.spatial.transform import Rotation as R
        r = R.from_euler('xyz', [180, 0, 0], degrees=True).as_quat()
        franka_start_pose.r = gymapi.Quat(r[0], r[1], r[2], r[3])#gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        # Define start pose for table
        table_start_pose = gymapi.Transform()
        table_start_pose.p = gymapi.Vec3(*table_pos)
        table_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
        self._table_surface_pos = np.array(table_pos) + np.array([0, 0, table_thickness / 2])
        self.reward_settings["table_height"] = self._table_surface_pos[2]

        # Define start pose for cubes (doesn't really matter since they're get overridden during reset() anyways)
        
        cubeA_start_pose = gymapi.Transform()
        cubeA_start_pose.p = gymapi.Vec3(0.0, 0.1, -10.0)
        cubeA_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
        cubeB_start_pose = gymapi.Transform()
        cubeB_start_pose.p = gymapi.Vec3(0.0, 0.2, -10.0)
        cubeB_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
        cylinder_start_pose = gymapi.Transform()
        cylinder_start_pose.p = gymapi.Vec3(0.0, -0.1, -10.0)
        cylinder_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        object_model_start_pose = []
        for counter in range(len(self.object_models)):
            object_model_start_pose.append(gymapi.Transform())
            object_model_start_pose[counter].p = gymapi.Vec3(0.0, -0.1, -10.0)
            object_model_start_pose[counter].r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        # compute aggregate size
        num_franka_bodies = self.gym.get_asset_rigid_body_count(franka_asset)
        num_franka_shapes = self.gym.get_asset_rigid_shape_count(franka_asset)
        max_agg_bodies = num_franka_bodies + 27 + 3 + len(self.object_models)   # 1 for table, table stand, cubeA, cubeB
        max_agg_shapes = num_franka_shapes + 27 + 3 + len(self.object_models)    # 1 for table, table stand, cubeA, cubeB

        self.frankas = []
        self.envs = []

        self.multi_body_idx = {
            "base_link": self.gym.find_asset_rigid_body_index(franka_asset, "base_link"),
            "wrist_3_link": self.gym.find_asset_rigid_body_index(franka_asset, "wrist_3_link"),
            "ee_link": self.gym.find_asset_rigid_body_index(franka_asset, "ee_link"),
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
            # Potentially randomize start pose
            '''if self.franka_position_noise > 0:
                rand_xy = self.franka_position_noise * (-1. + np.random.rand(2) * 2.0)
                franka_start_pose.p = gymapi.Vec3(-0.45 + rand_xy[0], 0.0 + rand_xy[1],
                                                 1.0 + table_thickness / 2 + table_stand_height)
            if self.franka_rotation_noise > 0:
                rand_rot = torch.zeros(1, 3)
                rand_rot[:, -1] = self.franka_rotation_noise * (-1. + np.random.rand() * 2.0)
                new_quat = axisangle2quat(rand_rot).squeeze().numpy().tolist()
                franka_start_pose.r = gymapi.Quat(*new_quat)'''

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

            # Create cubes
            self._cubeA_id = self.gym.create_actor(env_ptr, cubeA_asset, cubeA_start_pose, "cubeA", i, 1, 0)
            self._cubeB_id = self.gym.create_actor(env_ptr, cubeB_asset, cubeB_start_pose, "cubeB", i, 2, 0)
            # Set colors
            self.gym.set_rigid_body_color(env_ptr, self._cubeA_id, 0, gymapi.MESH_VISUAL, cubeA_color)
            self.gym.set_rigid_body_color(env_ptr, self._cubeB_id, 0, gymapi.MESH_VISUAL, cubeB_color)
            self.gym.set_rigid_body_segmentation_id(env_ptr, self._cubeA_id, 0, 1)
            self.gym.set_rigid_body_segmentation_id(env_ptr, self._cubeB_id, 0, 2)

            # Create cylinder
            self._cylinder_id = self.gym.create_actor(env_ptr, cylinder_asset, cylinder_start_pose, "cylinder", i, 3, 0)
            # Set colors
            self.gym.set_rigid_body_color(env_ptr, self._cylinder_id, 0, gymapi.MESH_VISUAL, cylinder_color)
            self.gym.set_rigid_body_segmentation_id(env_ptr, self._cylinder_id, 0, 3)

            self._object_model_id = []
            for counter in range(len(self.object_models)):
                self._object_model_id.append(self.gym.create_actor(env_ptr, object_model_asset[counter], object_model_start_pose[counter], self.object_models[counter], i, counter+4, 0))
                self.gym.set_rigid_body_segmentation_id(env_ptr, self._object_model_id[counter], 0, counter+4)

            if self.aggregate_mode > 0:
                self.gym.end_aggregate(env_ptr)

            # Store the created env pointers
            self.envs.append(env_ptr)
            self.frankas.append(self.franka_actor)

            '''
            Camera Setup
            '''
            # Camera environment setup
            self.camera_handles.append([])
            self.body_states = []
            self.camera_properties = gymapi.CameraProperties()
            self.camera_properties.enable_tensors = True
            self.camera_properties.horizontal_fov = 54.0
            self.camera_properties.width = 640
            self.camera_properties.height = 480
            camera_handle = self.gym.create_camera_sensor(env_ptr, self.camera_properties)
            self.camera_base_link_translation = torch.tensor([-0.54, 0.05, 0.6]).to(self.device)
            local_transform = gymapi.Transform()
            local_transform.p = gymapi.Vec3(self.camera_base_link_translation[0], self.camera_base_link_translation[1], self.camera_base_link_translation[2])
            camera_rotation_x = gymapi.Quat.from_axis_angle(gymapi.Vec3(1, 0, 0), np.deg2rad(180))
            local_transform.r = camera_rotation_x

            franka_base_link_handle = self.gym.find_actor_rigid_body_handle(env_ptr, 0, self.franka_base)
            self.gym.attach_camera_to_body(camera_handle, env_ptr, franka_base_link_handle, local_transform, gymapi.FOLLOW_TRANSFORM)
            self.camera_handles[i].append(camera_handle)

            l_color = gymapi.Vec3(1, 1, 1)
            l_ambient = gymapi.Vec3(0.2, 0.2, 0.2)
            l_direction = gymapi.Vec3(-1, -1, 1)
            self.gym.set_light_parameters(self.sim, 0, l_color, l_ambient, l_direction)
            l_direction = gymapi.Vec3(-1, 1, 1)
            self.gym.set_light_parameters(self.sim, 1, l_color, l_ambient, l_direction)
        # Setup init state buffer
        self._init_cubeA_state = torch.zeros(self.num_envs, 13, device=self.device)
        self._init_cubeB_state = torch.zeros(self.num_envs, 13, device=self.device)

        # Cylinder init state buffer
        self._init_cylinder_state = torch.zeros(self.num_envs, 13, device=self.device)
        self._init_object_model_state = []
        for counter in range(len(self.object_models)):
            self._init_object_model_state.append(torch.zeros(self.num_envs, 13, device=self.device))

        # Setup data
        self.init_data()

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

        # props = self.gym.get_actor_dof_properties(self.env_ptr, table_handle)
        # props["driveMode"].fill(gymapi.DOF_MODE_NONE)
        # props["stiffness"].fill(0.0)
        # props["damping"].fill(0.0)
        # props["friction"].fill(1.0)
        # self.gym.set_actor_dof_properties(self.env_ptr, table_handle, props)

        #self.table_handles.append(table_handle)

    def init_data(self):
        # Setup sim handles
        env_ptr = self.envs[0]
        franka_handle = 0
        self.handles = {
            # Franka
            "base_link": self.gym.find_actor_rigid_body_handle(env_ptr, franka_handle, self.franka_base),
            "wrist_3_link": self.gym.find_actor_rigid_body_handle(env_ptr, franka_handle, self.franka_wrist_3_link),
            "hand": self.gym.find_actor_rigid_body_handle(env_ptr, franka_handle, self.franka_hand),
            # Cubes
            "cubeA_body_handle": self.gym.find_actor_rigid_body_handle(self.envs[0], self._cubeA_id, "box"),
            "cubeB_body_handle": self.gym.find_actor_rigid_body_handle(self.envs[0], self._cubeB_id, "box"),
            "cylinder_body_handle": self.gym.find_actor_rigid_body_handle(self.envs[0], self._cylinder_id, "cylinder"),
        }

        for counter in range(len(self.object_models)):
            self.handles[self.object_models[counter]+"_body_handle"] = self.gym.find_actor_rigid_body_handle(self.envs[0], self._object_model_id[counter], self.object_models[counter])
        
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
        self._j_eef = jacobian[:, hand_joint_index, :, :7]
        _massmatrix = self.gym.acquire_mass_matrix_tensor(self.sim, "franka")
        mm = gymtorch.wrap_tensor(_massmatrix)
        self._mm = mm[:, :7, :7]
        self._cubeA_state = self._root_state[:, self._cubeA_id, :]
        self._cubeB_state = self._root_state[:, self._cubeB_id, :]
        self._cylinder_state = self._root_state[:, self._cylinder_id, :]

        self._object_model_state = []
        for counter in range(len(self.object_models)):
            self._object_model_state.append(self._root_state[:, self._object_model_id[counter], :])

        # Initialize states
        self.states.update({
            "cubeA_size": torch.ones_like(self._eef_state[:, 0]) * self.cubeA_size,
            "cubeB_size": torch.ones_like(self._eef_state[:, 0]) * self.cubeB_size,
        })
        
        # Initialize actions
        self._pos_control = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
        self._effort_control = torch.zeros_like(self._pos_control)
        # Initialize control
        self._arm_control = self._effort_control[:, :]

        # Initialize indices    ------ > self.num_envs * num of actors
        self._global_indices = torch.arange(self.num_envs * (3 + 28 + len(self.object_models)), dtype=torch.int32,
                                           device=self.device).view(self.num_envs, -1)

    def _update_states(self):
        self.states.update({
            # Franka
            "base_link": self._base_link[:, :7],
            "wrist_3_link": self._wrist_3_link[:, :7],
            "q": self._q[:, :],
            "q_gripper": self._q[:, -2:],
            "eef_pos": self._eef_state[:, :3],
            "eef_quat": self._eef_state[:, 3:7],
            "eef_vel": self._eef_state[:, 7:],
            # Cubes
            "cubeA_quat": self._cubeA_state[:, 3:7],
            "cubeA_pos": self._cubeA_state[:, :3],
            "cubeA_pos_relative": self._cubeA_state[:, :3] - self._eef_state[:, :3],
            "cubeB_quat": self._cubeB_state[:, 3:7],
            "cubeB_pos": self._cubeB_state[:, :3],
            "cubeA_to_cubeB_pos": self._cubeB_state[:, :3] - self._cubeA_state[:, :3],
        })

    def _refresh(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)
        self.gym.refresh_mass_matrix_tensors(self.sim)

        # Refresh states
        self._update_states()

    def compute_reward(self, actions):
        self.rew_buf[:], self.reset_buf[:] = compute_franka_reward(
            self.reset_buf, self.progress_buf, self.actions, self.states, self.reward_settings, self.max_episode_length
        )

    def compute_observations(self):
        self._refresh()
        obs = ["cubeA_quat", "cubeA_pos", "cubeA_to_cubeB_pos", "eef_pos", "eef_quat"]
        obs += ["q_gripper"] if self.control_type == "osc" else ["q"]
        self.obs_buf = torch.cat([self.states[ob] for ob in obs], dim=-1)

        maxs = {ob: torch.max(self.states[ob]).item() for ob in obs}

        return self.obs_buf

    def reset_idx(self, env_ids):
        # resetting the count for saving the iamges
        self.frame_count = 0

        env_ids_int32 = env_ids.to(dtype=torch.int32)

        # Reset cubes, sampling cube B first, then A
        # Used offset to set the cubes in the particular location for each environment
        self._reset_init_object_state(object='cubeA', env_ids=env_ids, offset=[0, -0.1], check_valid=True)
        self._reset_init_object_state(object='cubeB', env_ids=env_ids, offset=[0, -0.175], check_valid=True)
        self._reset_init_object_state(object='cylinder', env_ids=env_ids, offset=[0, -0.3], check_valid=True)

        for counter in range(len(self.object_models)):
            self._reset_init_object_state(object=self.object_models[counter], env_ids=env_ids, offset=[0, 0.1], check_valid=True)

        # Write these new init states to the sim states
        self._cubeA_state[env_ids] = self._init_cubeA_state[env_ids]
        self._cubeB_state[env_ids] = self._init_cubeB_state[env_ids]
        self._cylinder_state[env_ids] = self._init_cylinder_state[env_ids]

        for counter in range(len(self.object_models)):
            self._object_model_state[counter][env_ids] = self._init_object_model_state[counter][env_ids]

        # Reset agent
        reset_noise = torch.rand((len(env_ids), self.num_franka_dofs), device=self.device)
        pos = tensor_clamp(
            self.franka_default_dof_pos.unsqueeze(0) +
            self.franka_dof_noise * 2.0 * (reset_noise - 0.5),
            self.franka_dof_lower_limits.unsqueeze(0), self.franka_dof_upper_limits)
            # self.franka_dof_noise * 2.0 * (reset_noise - 0.5),

        # Overwrite gripper init pos (no noise since these are always position controlled)
        pos[:, -2:] = self.franka_default_dof_pos[-2:]

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

        # Update cube states
        multi_env_ids_cubes_int32 = self._global_indices[env_ids, -3-len(self.object_models):].flatten()
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self._root_state),
            gymtorch.unwrap_tensor(multi_env_ids_cubes_int32), len(multi_env_ids_cubes_int32))

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0

        self.action_contrib = 1

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
        if object.lower()[:4] == 'cube':
            if(object.lower()[4] == 'a'):
                this_object_state_all = self._init_cubeA_state
            elif(object.lower()[4] == 'b'):
                this_object_state_all = self._init_cubeB_state
        elif object.lower() == 'cylinder':
            this_object_state_all = self._init_cylinder_state
        elif object == self.object_models[0]:
            this_object_state_all = self._init_object_model_state[0]
        else:
            raise ValueError(f"Invalid cube specified, options are 'A' and 'B'; got: {object}")

        # Sampling is "centered" around middle of table
        centered_object_xy_state = torch.tensor(np.array([0.6, 0.0]), device=self.device, dtype=torch.float32)
        
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


            # retrying for 100 times
            for i in range(10000):
                # Sample x y values
                sampled_object_state[active_idx, :2] = centered_object_xy_state
                # Setting the X axis value for the object
                # Set the offset for cube A on both the environment
                offset_xy = torch.zeros(1, 2).to(self.device)
                # offset_x_axis = torch.tensor(offset_xy).to(device=self.device)
                offset_x_axis = offset_xy +  torch.tensor([offset]).to(self.device)
                # Sometimes it only tries to sample for one environment, this might be due to frequency mismatch of the env (but not sure)
                if(active_idx.size()<torch.Size([num_resets])):
                    continue
                else:
                    sampled_object_state[active_idx, :2] += offset_x_axis
                    break
                        
                '''
                This code is to sample only for the particular environment
                '''
                # if(active_idx[0] == torch.tensor([0], device=self.device)):
                #     offset_xy = torch.ones(len(env_ids), 1).to(self.device)
                #     offset_x_axis = offset_xy +  torch.tensor([offset]).to(self.device)

            # Make sure we succeeded at sampling
            # assert success, "Sampling cube locations was unsuccessful! ):"
        else:
            # We just directly sample if check_valid variable is False
            sampled_object_state[:, :2] = centered_object_xy_state.unsqueeze(0) + \
                                              2.0 * self.start_position_noise * (
                                                      torch.rand(num_resets, 2, device=self.device) - 0.5)

        # # Sample rotation value
        # if self.start_rotation_noise > 0:
        #     aa_rot = torch.zeros(num_resets, 3, device=self.device)
        #     aa_rot[:, 2] = 2.0 * self.start_rotation_noise * (torch.rand(num_resets, device=self.device) - 0.5)
        #     sampled_cube_state[:, 3:7] = quat_mul(axisangle2quat(aa_rot), sampled_cube_state[:, 3:7])

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
                self.kp * dpose - self.kd * self.states["eef_vel"]).unsqueeze(-1)

        # Nullspace control torques `u_null` prevents large changes in joint configuration
        # They are added into the nullspace of OSC so that the end effector orientation remains constant
        # roboticsproceedings.org/rss07/p31.pdf
        j_eef_inv = m_eef @ self._j_eef @ mm_inv
        u_null = self.kd_null * -qd + self.kp_null * (
                (self.franka_default_dof_pos[:7] - q + np.pi) % (2 * np.pi) - np.pi)
        u_null[:, self.num_franka_dofs:] *= 0
        u_null = self._mm @ u_null.unsqueeze(-1)
        u += (torch.eye(self.num_franka_dofs, device=self.device).unsqueeze(0) - torch.transpose(self._j_eef, 1, 2) @ j_eef_inv) @ u_null

        # Clip the values to be within valid effort range
        u = tensor_clamp(u.squeeze(-1),
                         -self._franka_effort_limits[:7].unsqueeze(0), self._franka_effort_limits[:7].unsqueeze(0))

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

    def to_pose_ik(self, dpose):
        '''print('state pose shape', self.states["wrist_3_link"].shape)
        pos_err = goal_pose - self.states["wrist_3_link"][:, :3]
        orn_err = self.orientation_error(goal_orientation)'''
        dpose = dpose.unsqueeze(-1)
        u = self._compute_ik(dpose)
        self._q += u
        self._qd = torch.zeros_like(self._qd)


    def orientation_error(self, desired, current):
        '''
        input: desired orientation - quaterion, current orientation - quaterion
        output: orientation err - euler
        '''
        cc = quat_conjugate(current)
        q_r = quat_mul(desired, cc)
        return q_r[:3] * torch.sign(q_r[3]).unsqueeze(-1)

    def pre_physics_step(self, actions):
        
        '''
        Camera access in the pre physics step to compute the force using suction cup deformation score
        '''
        # communicate physics to graphics system
        self.gym.step_graphics(self.sim)
        # render the camera sensors
        self.gym.render_all_camera_sensors(self.sim)

        # for i in range(0, self.num_envs):
        for i in range(self.num_envs):
            # retrieving rgb iamge
            rgb_camera_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, self.envs[i], self.camera_handles[i][0], gymapi.IMAGE_COLOR)
            torch_rgb_tensor = gymtorch.wrap_tensor(rgb_camera_tensor)
            rgb_image = torch_rgb_tensor.to(self.device)
            rgb_image_copy = torch.reshape(rgb_image, (rgb_image.shape[0], -1, 4))[..., :3]

            depth_camera_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, self.envs[i], self.camera_handles[i][0], gymapi.IMAGE_DEPTH)
            torch_depth_tensor = gymtorch.wrap_tensor(depth_camera_tensor)
            depth_image = torch_depth_tensor.to(self.device)
            depth_image = -depth_image
            # retrieving depth and mask
            mask_camera_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, self.envs[i], self.camera_handles[i][0], gymapi.IMAGE_SEGMENTATION)
            torch_mask_tensor = gymtorch.wrap_tensor(mask_camera_tensor)
            segmask = torch_mask_tensor.to(self.device)
            
            '''
            Point cloud with tensor
            '''
            # cam_vinv = torch.inverse((torch.tensor(self.gym.get_camera_view_matrix(self.sim, self.envs[i], self.camera_handles[i][0])))).to(self.device)
            cam_proj = torch.tensor(self.gym.get_camera_proj_matrix(self.sim, self.envs[i], self.camera_handles[i][0]), device=self.device)

            width = self.camera_properties.width
            height = self.camera_properties.height
            fx = width/(2/cam_proj[0, 0])
            fy = height/(2/cam_proj[1, 1])
            cx = width/2
            cy = height/2
            self.camera_info = CameraInfo(width, height, fx, fy, cx, cy)
            
            '''
            Here the function is called which will calculate the conical spring score for each object which is denoted by item_id
            ''' 
            object_id = torch.tensor(2).to(self.device)
            total_objects = 3+len(self.object_models)
            if(len(torch.unique(segmask)) == total_objects+1):
                '''
                DexNet 3.0
                '''
                if(self.frame_count == 30):
                    try:
                        camera_intrinsics = CameraIntrinsics(frame="camera", fx=self.camera_info.fx, fy=self.camera_info.fy, cx=self.camera_info.cx, cy=self.camera_info.cy, skew=0.0, height=self.camera_info.height, width=self.camera_info.width)
                        
                        segmask_dexnet = segmask.detach().clone()
                        segmask_numpy = np.zeros_like(segmask_dexnet.cpu().numpy().astype(np.uint8))

                        segmask_numpy_temp = np.zeros_like(segmask_dexnet.cpu().numpy().astype(np.uint8))
                        segmask_numpy_temp[segmask_dexnet.cpu().numpy().astype(np.uint8) == object_id.cpu().numpy()] = 1

                        segmask_numpy[segmask_dexnet.cpu().numpy().astype(np.uint8) == object_id.cpu().numpy()] = 255
                        segmask_dexnet = BinaryImage(segmask_numpy, frame=camera_intrinsics.frame)
                        
                        depth_image_dexnet = depth_image.detach().clone()
                        depth_image_dexnet -= 0.5
                        noise_image = torch.normal(0, 0.0009, size=depth_image_dexnet.size()).to(self.device)
                        depth_image_dexnet = depth_image_dexnet + noise_image
                        
                        depth_numpy = depth_image_dexnet.cpu().numpy()
                        depth_numpy_temp = depth_numpy*segmask_numpy_temp
                        depth_numpy_temp[depth_numpy_temp == 0] = 0.75
                        
                        depth_img_dexnet = DepthImage(depth_numpy_temp, frame=camera_intrinsics.frame)

                        dexnet_object = dexnet3(depth_img_dexnet, segmask_dexnet, None, camera_intrinsics)
                        action, grasps_and_predictions = dexnet_object.inference()

                        print(f"Quality is {action.q_value} grasp location is {action.grasp.center.x}, {action.grasp.center.y}")
                        '''
                        Suction Cup Deformation
                        '''
                        depth_image_suction = depth_image.detach().clone()
                        suction_score_object = calcualte_suction_score(depth_image_suction, segmask, rgb_image_copy, self.camera_info, grasps_and_predictions, object_id, action)
                        score, xyz_point = suction_score_object.calculator()
                        self.object_coordiante_camera = xyz_point.clone()
                        force_object = calcualte_force(score)
                        force = force_object.regression()
                        print(f"suction deforamtion score --> {score}, Force along z axis --> {force}")
                    except:
                        print('DEX net exception')
                        self.frame_count -= 1
            '''
            To save rgb image, depth image and segmentation mask (comment this section if you do not want to visualize as it slows down the processing)
            '''
            self.frame_count += 1
            # plt.imsave(f'{self.current_directory}/../Data/segmask_{self.frame_count}_{i}.png', segmask.detach().cpu().numpy(), cmap=cm.gray)
            # iio.imwrite(f"{self.current_directory}/../Data/rgb_frame_{self.frame_count}_{i}.png", rgb_image_copy.detach().cpu().numpy())
            # np.save(f"{self.current_directory}/../Data/depth_frame_{self.frame_count}_{i}.npy", depth_image.detach().cpu().numpy())

        '''
        Commands to the arm for eef control
        '''
        poses_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        poses = gymtorch.wrap_tensor(poses_tensor).view(self.num_envs, -1, 13)
        
        # Transformation of base_link from world coordiante frame (wb)
        rotation_matrix_base_link = euler_angles_to_matrix(torch.tensor([180, 0, 0]).to(self.device), "XYZ", degrees=True)
        translation_base_link = torch.tensor([0, 0, 2.020]).to(self.device)
        T_base_link = transformation_matrix(rotation_matrix_base_link, translation_base_link)

        # Transformation for camera (wc --> wb*bc)
        rotation_matrix_camera_offset = euler_angles_to_matrix(torch.tensor([180, 0, 0]).to(self.device), "XYZ", degrees=True)
        T_base_link_to_camera = transformation_matrix(rotation_matrix_camera_offset, self.camera_base_link_translation)

        T_world_to_camera_link = torch.matmul(T_base_link, T_base_link_to_camera)

        # Transformation for object from camera (wo --> wc*co)
        rotation_matrix_camera_to_object = euler_angles_to_matrix(torch.tensor([0, 0, 0]).to(self.device), "XYZ", degrees=True)
        T_camera_to_object = transformation_matrix(rotation_matrix_camera_to_object, self.object_coordiante_camera)
        
        # Transformation from base link to object
        T_world_to_object = torch.matmul(T_world_to_camera_link, T_camera_to_object)

        # Transformation for pre grasp pose (wp --> wo*op)
        rotation_matrix_pre_grasp_pose = euler_angles_to_matrix(torch.tensor([0, 0, 0]).to(self.device), "XYZ", degrees=True)
        translation_pre_grasp_pose = torch.tensor([-0.25, 0, 0]).to(self.device)
        T_pre_grasp_pose = transformation_matrix(rotation_matrix_pre_grasp_pose, translation_pre_grasp_pose)

        # Transformation of object with base link to pre grasp pose
        T_world_to_pre_grasp_pose = torch.matmul(T_world_to_object, T_pre_grasp_pose)

        # Transformation for pre grasp pose (pe --> inv(we)*wp --> ew*wp)
        rotation_matrix_ee_pose = quaternion_to_matrix(poses[0][self.multi_body_idx['ee_link']][3:7])
        translation_ee_pose = poses[0][self.multi_body_idx['ee_link']][:3]
        T_world_to_ee_pose = transformation_matrix(rotation_matrix_ee_pose, translation_ee_pose)
        
        T_ee_pose_to_pre_grasp_pose = torch.matmul(torch.inverse(T_world_to_ee_pose), T_world_to_pre_grasp_pose)
        
        # Orientation error
        action_orientation = matrix_to_euler_angles(T_ee_pose_to_pre_grasp_pose[:3, :3], "XYZ")
        
        # # Visualization
        '''
        use visulization only to view the trasnformations otherwise keep it disable as it is creating issue for segmentation of object
        '''
        # self.gym.clear_lines(self.viewer)
        # axes_geom = gymutil.AxesGeometry(0.4)
        # # Create a wireframe sphere
        # sphere_rot = gymapi.Quat.from_euler_zyx(0.5 * math.pi, 0, 0)
        # sphere_pose = gymapi.Transform(r=sphere_rot)
        # sphere_geom = gymutil.WireframeSphereGeometry(0.02, 12, 12, sphere_pose, color=(1, 1, 0))

        # top_drawer_grasp = gymapi.Transform()
        # camera_trans = T_world_to_pre_grasp_pose[:3, 3].detach().cpu().numpy()
        # top_drawer_grasp.p = gymapi.Vec3(camera_trans[0], camera_trans[1], camera_trans[2])
        # top_drawer_rot = R.from_matrix(T_world_to_pre_grasp_pose[:3, :3].detach().cpu().numpy())
        # top_drawer_quat = top_drawer_rot.as_quat()
        # top_drawer_grasp.r = gymapi.Quat(top_drawer_quat[0], top_drawer_quat[1], top_drawer_quat[2], top_drawer_quat[3])
        # # camera_rotation_x = gymapi.Quat.from_axis_angle(gymapi.Vec3(1, 0, 0), np.deg2rad(180))
        # gymutil.draw_lines(axes_geom, self.gym, self.viewer, self.envs[0], top_drawer_grasp)
        # gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[0], top_drawer_grasp)

        # top_drawer_grasp = gymapi.Transform()
        # camera_trans = T_ee_pose_to_pre_grasp_pose[:3, 3]
        # top_drawer_grasp.p = gymapi.Vec3(camera_trans[0], camera_trans[1], camera_trans[2])
        # top_drawer_rot = R.from_matrix(T_ee_pose_to_pre_grasp_pose[:3, :3])
        # top_drawer_quat = top_drawer_rot.as_quat()
        # top_drawer_grasp.r = gymapi.Quat(top_drawer_quat[0], top_drawer_quat[1], top_drawer_quat[2], top_drawer_quat[3])
        # # camera_rotation_x = gymapi.Quat.from_axis_angle(gymapi.Vec3(1, 0, 0), np.deg2rad(180))
        # gymutil.draw_lines(axes_geom, self.gym, self.viewer, self.envs[0], top_drawer_grasp)
        # gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[0], top_drawer_grasp)

        # top_drawer_grasp = gymapi.Transform()
        # top_drawer_grasp.p = gymapi.Vec3(0, 0, 0)
        # top_drawer_grasp.r = gymapi.Quat(0, 0, 0, 1)
        # # camera_rotation_x = gymapi.Quat.from_axis_angle(gymapi.Vec3(1, 0, 0), np.deg2rad(180))
        # gymutil.draw_lines(axes_geom, self.gym, self.viewer, self.envs[0], top_drawer_grasp)
        # gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[0], top_drawer_grasp)

        if(self.action_contrib == 1):
            actions = torch.tensor(self.num_envs * [[T_ee_pose_to_pre_grasp_pose[0][3], T_ee_pose_to_pre_grasp_pose[1][3], T_ee_pose_to_pre_grasp_pose[2][3], 0.1*action_orientation[0], 0.1*action_orientation[1], 0.1*action_orientation[2], 1]], dtype=torch.float)
        else:
            actions = torch.tensor(self.num_envs * [[0.1, T_ee_pose_to_pre_grasp_pose[1][3], T_ee_pose_to_pre_grasp_pose[2][3], 0.1*action_orientation[0], 0.1*action_orientation[1], 0.1*action_orientation[2], 1]], dtype=torch.float)
        
        if(self.frame_count >= 30):
            if((torch.max(torch.abs(actions[0][:6]))) <= 0.003):
                self.action_contrib = 0
        
        self.actions = actions.clone().to(self.device)
        # Split arm and gripper command
        u_arm, u_gripper = self.actions[:, :-1], self.actions[:, -1]

        # Control arm (scale value first)
        u_arm = u_arm * self.cmd_limit / self.action_scale
        if self.control_type == "osc":
            u_arm = torch.clip(self._compute_osc_torques(dpose=u_arm), min=-10, max=10)
        self._arm_control[:, :] = u_arm

        # Control gripper
        '''u_fingers = torch.zeros_like(self._gripper_control)
        u_fingers[:, 0] = torch.where(u_gripper >= 0.0, self.franka_dof_upper_limits[-2].item(),
                                    self.franka_dof_lower_limits[-2].item())
        u_fingers[:, 1] = torch.where(u_gripper >= 0.0, self.franka_dof_upper_limits[-1].item(),
                                    self.franka_dof_lower_limits[-1].item())
        # Write gripper command to appropriate tensor buffer
        self._gripper_control[:, :] = u_fingers'''

        # Deploy actions
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self._pos_control))
        self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self._effort_control))

    def post_physics_step(self):

        self.gym.refresh_force_sensor_tensor(self.sim)
        try:
            _fsdata = self.gym.acquire_force_sensor_tensor(self.sim)
            fsdata = gymtorch.wrap_tensor(_fsdata)
            print("force along z axis: ", fsdata[0][2])
        except:
            pass

        self.progress_buf += 1

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        self.compute_observations()
        self.compute_reward(self.actions)

        # debug viz
        if self.viewer and self.debug_viz:
            self.gym.clear_lines(self.viewer)
            self.gym.refresh_rigid_body_state_tensor(self.sim)

            # Grab relevant states to visualize
            eef_pos = self.states["eef_pos"]
            eef_rot = self.states["eef_quat"]
            cubeA_pos = self.states["cubeA_pos"]
            cubeA_rot = self.states["cubeA_quat"]
            cubeB_pos = self.states["cubeB_pos"]
            cubeB_rot = self.states["cubeB_quat"]

            # Plot visualizations
            for i in range(self.num_envs):
                for pos, rot in zip((eef_pos, cubeA_pos, cubeB_pos), (eef_rot, cubeA_rot, cubeB_rot)):
                    px = (pos[i] + quat_apply(rot[i], to_torch([1, 0, 0], device=self.device) * 0.2)).cpu().numpy()
                    py = (pos[i] + quat_apply(rot[i], to_torch([0, 1, 0], device=self.device) * 0.2)).cpu().numpy()
                    pz = (pos[i] + quat_apply(rot[i], to_torch([0, 0, 1], device=self.device) * 0.2)).cpu().numpy()

                    p0 = pos[i].cpu().numpy()
                    self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], px[0], px[1], px[2]], [0.85, 0.1, 0.1])
                    self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], py[0], py[1], py[2]], [0.1, 0.85, 0.1])
                    self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], pz[0], pz[1], pz[2]], [0.1, 0.1, 0.85])

        
#####################################################################
###=========================jit functions=========================###
#####################################################################
@torch.jit.script
def compute_franka_reward(
    reset_buf, progress_buf, actions, states, reward_settings, max_episode_length
):
    # type: (Tensor, Tensor, Tensor, Dict[str, Tensor], Dict[str, float], float) -> Tuple[Tensor, Tensor]

    # Compute per-env physical parameters
    target_height = states["cubeB_size"] + states["cubeA_size"] / 2.0
    cubeA_size = states["cubeA_size"]
    cubeB_size = states["cubeB_size"]

    # distance from hand to the cubeA
    d = torch.norm(states["cubeA_pos_relative"], dim=-1)
    #d_lf = torch.norm(states["cubeA_pos"] - states["eef_lf_pos"], dim=-1)
    #d_rf = torch.norm(states["cubeA_pos"] - states["eef_rf_pos"], dim=-1)
    dist_reward = 1 - torch.tanh(10.0 * (d) / 3) #1 - torch.tanh(10.0 * (d + d_lf + d_rf) / 3)

    # reward for lifting cubeA
    cubeA_height = states["cubeA_pos"][:, 2] - reward_settings["table_height"]
    cubeA_lifted = (cubeA_height - cubeA_size) > 0.04
    lift_reward = cubeA_lifted

    # how closely aligned cubeA is to cubeB (only provided if cubeA is lifted)
    offset = torch.zeros_like(states["cubeA_to_cubeB_pos"])
    offset[:, 2] = (cubeA_size + cubeB_size) / 2
    d_ab = torch.norm(states["cubeA_to_cubeB_pos"] + offset, dim=-1)
    align_reward = (1 - torch.tanh(10.0 * d_ab)) * cubeA_lifted

    # Dist reward is maximum of dist and align reward
    dist_reward = torch.max(dist_reward, align_reward)

    # final reward for stacking successfully (only if cubeA is close to target height and corresponding location, and gripper is not grasping)
    cubeA_align_cubeB = (torch.norm(states["cubeA_to_cubeB_pos"][:, :2], dim=-1) < 0.02)
    cubeA_on_cubeB = torch.abs(cubeA_height - target_height) < 0.02
    gripper_away_from_cubeA = (d > 0.04)
    stack_reward = cubeA_align_cubeB & cubeA_on_cubeB & gripper_away_from_cubeA

    # Compose rewards

    # We either provide the stack reward or the align + dist reward
    rewards = torch.where(
        stack_reward,
        reward_settings["r_stack_scale"] * stack_reward,
        reward_settings["r_dist_scale"] * dist_reward + reward_settings["r_lift_scale"] * lift_reward + reward_settings[
            "r_align_scale"] * align_reward,
    )

    # Compute resets
    reset_buf = torch.where((progress_buf >= max_episode_length - 1) | (stack_reward > 0), torch.ones_like(reset_buf), reset_buf)

    return rewards, reset_buf
