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

        self.debug_viz = self.cfg["env"]["enableDebugVis"]

        self.up_axis = "z"
        self.up_axis_idx = 2

        self.franka_hand = "wrist_3_link"
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../configs")+'/collision_primitives_3d.yml') as file:
            self.world_params = yaml.load(file, Loader=yaml.FullLoader)

        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)

        # Franka defaults
        self.franka_default_dof_pos = to_torch(
            [1.0, -1.0, 2.83, 0.58, 1.67, 1.74], device=self.device
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
        #asset_options.disable_gravity = True
        asset_options.thickness = 0.001
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_EFFORT
        asset_options.use_mesh_materials = True
        franka_asset = self.gym.load_asset(self.sim, asset_root, franka_asset_file, asset_options)

        franka_dof_stiffness = to_torch([800, 800, 800, 800, 800, 800], dtype=torch.float, device=self.device)
        franka_dof_damping = to_torch([40, 40, 40, 40, 40, 40], dtype=torch.float, device=self.device)

        # Create table asset
        table_pos = [0.0, 0.0, 1.0]
        table_thickness = 0.05
        table_opts = gymapi.AssetOptions()
        table_opts.fix_base_link = True
        table_asset = self.gym.create_box(self.sim, *[1.2, 1.2, table_thickness], table_opts)

        # Create table stand asset
        '''table_stand_height = 0.1
        table_stand_pos = [-0.5, 0.0, 1.0 + table_thickness / 2 + table_stand_height / 2]
        table_stand_opts = gymapi.AssetOptions()
        table_stand_opts.fix_base_link = True
        table_stand_asset = self.gym.create_box(self.sim, *[0.2, 0.2, table_stand_height], table_opts)'''

        self.cubeA_size = 0.050
        self.cubeB_size = 0.070

        # Create cubeA asset
        cubeA_opts = gymapi.AssetOptions()
        cubeA_asset = self.gym.create_box(self.sim, *([self.cubeA_size] * 3), cubeA_opts)
        cubeA_color = gymapi.Vec3(0.6, 0.1, 0.0)

        # Create cubeB asset
        cubeB_opts = gymapi.AssetOptions()
        cubeB_asset = self.gym.create_box(self.sim, *([self.cubeB_size] * 3), cubeB_opts)
        cubeB_color = gymapi.Vec3(0.0, 0.4, 0.1)

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
        franka_start_pose.p = gymapi.Vec3(2.208, 0.002, 2.020) #gymapi.Vec3(-0.45, 0.0, 1.0 + table_thickness / 2 + table_stand_height)
        from scipy.spatial.transform import Rotation as R
        r = R.from_euler('zyx', [0, 0, 180], degrees=True).as_quat()
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

        # compute aggregate size
        num_franka_bodies = self.gym.get_asset_rigid_body_count(franka_asset)
        num_franka_shapes = self.gym.get_asset_rigid_shape_count(franka_asset)
        max_agg_bodies = num_franka_bodies + 27 + 2    # 1 for table, table stand, cubeA, cubeB
        max_agg_shapes = num_franka_shapes + 27 + 2     # 1 for table, table stand, cubeA, cubeB

        self.frankas = []
        self.envs = []

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
            self._cubeA_id = self.gym.create_actor(env_ptr, cubeA_asset, cubeA_start_pose, "cubeA", i, 2, 0)
            self._cubeB_id = self.gym.create_actor(env_ptr, cubeB_asset, cubeB_start_pose, "cubeB", i, 4, 0)
            # Set colors
            self.gym.set_rigid_body_color(env_ptr, self._cubeA_id, 0, gymapi.MESH_VISUAL, cubeA_color)
            self.gym.set_rigid_body_color(env_ptr, self._cubeB_id, 0, gymapi.MESH_VISUAL, cubeB_color)
            self.gym.set_rigid_body_segmentation_id(env_ptr, self._cubeA_id, 0, 1)
            self.gym.set_rigid_body_segmentation_id(env_ptr, self._cubeB_id, 0, 2)

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
            camera_properties = gymapi.CameraProperties()
            camera_properties.enable_tensors = True
            camera_properties.horizontal_fov = 75.0
            camera_properties.width = 1080
            camera_properties.height = 720
            camera_handle = self.gym.create_camera_sensor(env_ptr, camera_properties)
            local_transform = gymapi.Transform()
            local_transform.p = gymapi.Vec3(-0.54, 0.05, 0.6)
            camera_rotation_x = gymapi.Quat.from_axis_angle(gymapi.Vec3(1, 0, 0), np.deg2rad(90))
            camera_rotation_y = gymapi.Quat.from_axis_angle(gymapi.Vec3(1, 0, 0), np.deg2rad(90))
            local_transform.r = camera_rotation_x* camera_rotation_y

            camera_actor_handle = self.gym.get_actor_handle(env_ptr, 0)
            camera_body_handle = self.gym.get_actor_rigid_body_handle(env_ptr, camera_actor_handle, 0)
            self.gym.attach_camera_to_body(camera_handle, env_ptr, camera_body_handle, local_transform, gymapi.FOLLOW_TRANSFORM)
            self.camera_handles[i].append(camera_handle)
            self.body_states.append(self.gym.get_env_rigid_body_states(env_ptr, gymapi.STATE_ALL)['pose'])
            # self.projection_matrix = np.matrix(self.gym.get_camera_proj_matrix(self.sim, env_ptr, camera_handle))
            # self.view_matrix = np.matrix(self.gym.get_camera_view_matrix(self.sim, env_ptr, camera_handle))
            # self.current_directory = os.getcwd()
            # np.save(f'{self.current_directory}/camera_properties/projection_matrix.npy', self.projection_matrix)
            l_color = gymapi.Vec3(1, 1, 1)
            l_ambient = gymapi.Vec3(0.2, 0.2, 0.2)
            l_direction = gymapi.Vec3(-1, -1, 1)
            self.gym.set_light_parameters(self.sim, 0, l_color, l_ambient, l_direction)
            l_direction = gymapi.Vec3(-1, 1, 1)
            self.gym.set_light_parameters(self.sim, 1, l_color, l_ambient, l_direction)
        # Setup init state buffer
        self._init_cubeA_state = torch.zeros(self.num_envs, 13, device=self.device)
        self._init_cubeB_state = torch.zeros(self.num_envs, 13, device=self.device)

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
            "hand": self.gym.find_actor_rigid_body_handle(env_ptr, franka_handle, self.franka_hand),
            # Cubes
            "cubeA_body_handle": self.gym.find_actor_rigid_body_handle(self.envs[0], self._cubeA_id, "box"),
            "cubeB_body_handle": self.gym.find_actor_rigid_body_handle(self.envs[0], self._cubeB_id, "box"),
        }
        
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
        #self._eef_lf_state = self._rigid_body_state[:, self.handles["leftfinger_tip"], :]
        #self._eef_rf_state = self._rigid_body_state[:, self.handles["rightfinger_tip"], :]
        _jacobian = self.gym.acquire_jacobian_tensor(self.sim, "franka")
        jacobian = gymtorch.wrap_tensor(_jacobian)
        hand_joint_index = self.gym.get_actor_joint_dict(env_ptr, franka_handle)['ee_fixed_joint']
        self._j_eef = jacobian[:, hand_joint_index, :, :7]
        _massmatrix = self.gym.acquire_mass_matrix_tensor(self.sim, "franka")
        mm = gymtorch.wrap_tensor(_massmatrix)
        self._mm = mm[:, :7, :7]
        self._cubeA_state = self._root_state[:, self._cubeA_id, :]
        self._cubeB_state = self._root_state[:, self._cubeB_id, :]

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
        #self._gripper_control = self._pos_control[:, 7:9]

        # Initialize indices    ------ > self.num_envs * num of actors
        self._global_indices = torch.arange(self.num_envs * (3 + 27), dtype=torch.int32,
                                           device=self.device).view(self.num_envs, -1)

    def _update_states(self):
        #print("cubeA_pos", self._cubeA_state[:, :3], "cubeB_pos", self._cubeB_state[:, :3])
        self.states.update({
            # Franka
            "q": self._q[:, :],
            "q_gripper": self._q[:, -2:],
            "eef_pos": self._eef_state[:, :3],
            "eef_quat": self._eef_state[:, 3:7],
            "eef_vel": self._eef_state[:, 7:],
            #"eef_lf_pos": self._eef_lf_state[:, :3],
            #"eef_rf_pos": self._eef_rf_state[:, :3],
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
        self._reset_init_cube_state(cube='B', env_ids=env_ids, offset=[0, -0.175], check_valid=True)
        self._reset_init_cube_state(cube='A', env_ids=env_ids, offset=[0, -0.1], check_valid=True)

        # Write these new init states to the sim states
        self._cubeA_state[env_ids] = self._init_cubeA_state[env_ids]
        self._cubeB_state[env_ids] = self._init_cubeB_state[env_ids]

        # Reset agent
        reset_noise = torch.rand((len(env_ids), self.num_franka_dofs), device=self.device)
        pos = tensor_clamp(
            self.franka_default_dof_pos.unsqueeze(0) + self.franka_dof_noise * 0.0 * (reset_noise - 0.5),
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
        multi_env_ids_cubes_int32 = self._global_indices[env_ids, -2:].flatten()
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self._root_state),
            gymtorch.unwrap_tensor(multi_env_ids_cubes_int32), len(multi_env_ids_cubes_int32))

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0

    def _reset_init_cube_state(self, cube, env_ids, offset, check_valid=True):
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
        sampled_cube_state = torch.zeros(num_resets, 13, device=self.device)

        # Get correct references depending on which one was selected
        if cube.lower() == 'a':
            this_cube_state_all = self._init_cubeA_state
            other_cube_state = self._init_cubeB_state[env_ids, :]
            cube_heights = self.states["cubeA_size"]
        elif cube.lower() == 'b':
            this_cube_state_all = self._init_cubeB_state
            other_cube_state = self._init_cubeA_state[env_ids, :]
            cube_heights = self.states["cubeB_size"]
        else:
            raise ValueError(f"Invalid cube specified, options are 'A' and 'B'; got: {cube}")

        # Minimum cube distance for guarenteed collision-free sampling is the sum of each cube's effective radius
        min_dists = (self.states["cubeA_size"] + self.states["cubeB_size"])[env_ids] * np.sqrt(2) / 2.0

        # We scale the min dist by 2 so that the cubes aren't too close together
        # min_dists = min_dists * 2.0

        # Sampling is "centered" around middle of table
        centered_cube_xy_state = torch.tensor(np.array([2.8, 0.0]), device=self.device, dtype=torch.float32)
        
        # Set z value, which is fixed height
        # sampled_cube_state[:, 2] = self._table_surface_pos[2] + cube_heights.squeeze(-1)[env_ids] / 2
        
        # For variable height
        # firs row H 1.7, second row G 1.5, third row F 1.35, ourth row 1.2
        # sampling height at which the cube will be dropped
        sampled_cube_state[:, 2] = torch.tensor(np.array([1.35]), device=self.device, dtype=torch.float32)

        # Initialize rotation, which is no rotation (quat w = 1)
        sampled_cube_state[:, 6] = 1.0

        # If we're verifying valid sampling, we need to check and re-sample if any are not collision-free
        # We use a simple heuristic of checking based on cubes' radius to determine if a collision would occur
        if check_valid:
            success = False
            # Indexes corresponding to envs we're still actively sampling for
            active_idx = torch.arange(num_resets, device=self.device)
            num_active_idx = len(active_idx)


            # retrying for 100 times
            for i in range(100):
                # Sample x y values
                sampled_cube_state[active_idx, :2] = centered_cube_xy_state
                # Setting the X axis value of the cube for cube A
                if(cube == 'A'):
                    # Set the offset for cube A on both the environment
                    offset_xy = [offset, offset]
                    offset_x_axis = torch.tensor(offset_xy).to(device=self.device)
                    # Sometimes it only tries to sample for one environment, this might be due to frequency mismatch of the env (but not sure)
                    if(active_idx.size()<torch.Size([2])):
                        # Adding the offset on each env
                        if(active_idx[0] == torch.tensor([0], device=self.device)):
                            offset_xy = [offset]
                            offset_x_axis = torch.tensor(offset_xy).to(device=self.device)
                        if(active_idx[0] == torch.tensor([1], device=self.device)):
                            offset_xy = [offset]
                            offset_x_axis = torch.tensor(offset_xy).to(device=self.device)
                else:
                    # This condition is for cube B
                    offset_xy = [offset, offset]
                    offset_x_axis = torch.tensor(offset_xy).to(device=self.device)

                    if(active_idx.size()<torch.Size([2])):
                        if(active_idx[0] == torch.tensor([0], device=self.device)):
                            offset_xy = [offset]
                            offset_x_axis = torch.tensor(offset_xy).to(device=self.device)
                        if(active_idx[0] == torch.tensor([1], device=self.device)):
                            offset_xy = [offset]
                            offset_x_axis = torch.tensor(offset_xy).to(device=self.device)
                sampled_cube_state[active_idx, :2] += offset_x_axis

                # Check if sampled values are valid
                cube_dist = torch.linalg.norm(sampled_cube_state[:, :2] - other_cube_state[:, :2], dim=-1)
                
 
                active_idx = torch.nonzero(cube_dist < min_dists, as_tuple=True)[0]
                num_active_idx = len(active_idx)
                # If active idx is empty, then all sampling is valid :D
                if num_active_idx == 0:
                    success = True
                    break
            # Make sure we succeeded at sampling
            # assert success, "Sampling cube locations was unsuccessful! ):"
        else:
            # We just directly sample if check_valid variable is False
            sampled_cube_state[:, :2] = centered_cube_xy_state.unsqueeze(0) + \
                                              2.0 * self.start_position_noise * (
                                                      torch.rand(num_resets, 2, device=self.device) - 0.5)

        # # Sample rotation value
        # if self.start_rotation_noise > 0:
        #     aa_rot = torch.zeros(num_resets, 3, device=self.device)
        #     aa_rot[:, 2] = 2.0 * self.start_rotation_noise * (torch.rand(num_resets, device=self.device) - 0.5)
        #     sampled_cube_state[:, 3:7] = quat_mul(axisangle2quat(aa_rot), sampled_cube_state[:, 3:7])

        # Lastly, set these sampled values as the new init state
        # print('env id', env_ids, 'pose', sampled_cube_state)
        this_cube_state_all[env_ids, :] = sampled_cube_state

    def _compute_osc_torques(self, dpose):
        # Solve for Operational Space Control # Paper: khatib.stanford.edu/publications/pdfs/Khatib_1987_RA.pdf
        # Helpful resource: studywolf.wordpress.com/2013/09/17/robot-control-4-operation-space-control/
        q, qd = self._q[:, :7], self._qd[:, :7]
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

    def pre_physics_step(self, actions):
        
        '''
        Camera access in the pre physics step to compute the force using suction cup deformation score
        '''
        # communicate physics to graphics system
        self.gym.step_graphics(self.sim)
        # render the camera sensors
        self.gym.render_all_camera_sensors(self.sim)
        points = []
        # for i in range(0, self.num_envs):
        for i in range(1,2):
            # retrieving rgb iamges
            rgb_image = self.gym.get_camera_image(self.sim, self.envs[i], self.camera_handles[i][0], gymapi.IMAGE_COLOR)
            rgb_image_copy = rgb_image.copy()
            rgb_image_copy = np.array(rgb_image_copy)
            rgb_image_copy = rgb_image_copy.reshape(rgb_image_copy.shape[0], -1, 4)[..., :3]
            
            # retrieving depth and mask
            depth_image = self.gym.get_camera_image(self.sim, self.envs[i], self.camera_handles[i][0], gymapi.IMAGE_DEPTH)
            segmask = self.gym.get_camera_image(self.sim, self.envs[i], self.camera_handles[i][0], gymapi.IMAGE_SEGMENTATION)

            # # refining the depth iamge
            depth_image = np.array(-depth_image)
            # # -inf implies no depth value, set it to zero. output will be black.
            depth_image[depth_image == np.inf] = 0
            # # clamp depth image to 10 meters to make output image human friendly
            depth_image[depth_image > 10] = 10.0
            # # normalizing the image, to make it visualizable
            # normalized_depth = -255.0*(depth_image/np.min(depth_image + 1e-4))
            # normalized_depth_image = im.fromarray(normalized_depth.astype(np.uint8))
            # print(depth_image)
            # adding noise to the depth iamge
            noise_image = np.random.normal(0, 0.0009, size=depth_image.shape)
            # print(noise_image)
            depth_image = depth_image + noise_image

            '''
            Point cloud with tensor
            '''
            cam_vinv = torch.inverse((torch.tensor(self.gym.get_camera_view_matrix(self.sim, self.envs[i], self.camera_handles[i][0])))).to(self.device)
            cam_proj = torch.tensor(self.gym.get_camera_proj_matrix(self.sim, self.envs[i], self.camera_handles[i][0]), device=self.device)
            camera_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, self.envs[i], self.camera_handles[i][0], gymapi.IMAGE_DEPTH)
            torch_cam_tensor = gymtorch.wrap_tensor(camera_tensor)
            segmask = np.asarray((segmask == 1), dtype=np.uint8)
            depth_buffer = torch_cam_tensor.to(self.device)
            depth_buffer[segmask == 0] = -10001

            width = 1080
            height = 720
            camera_u = torch.arange(0, width, device=self.device)
            camera_v = torch.arange(0, height, device=self.device)
            camera_v, camera_u = torch.meshgrid(camera_v, camera_u, indexing='ij')
            vinv = cam_vinv
            proj = cam_proj
            fu = 2/proj[0, 0]
            fv = 2/proj[1, 1]
            centerU = width/2
            centerV = height/2

            Z = depth_buffer
            X = -(camera_u-centerU)/width * Z * fu
            Y = (camera_v-centerV)/height * Z * fv

            depth_bar = 10
            Z = Z.view(-1)  
            valid = Z > -depth_bar
            X = X.view(-1)
            Y = Y.view(-1)


            position = torch.vstack((X, Y, Z, torch.ones(len(X), device=self.device)))[:, valid]
            position = position.permute(1, 0)
            # position = position@vinv

            points = position[:, 0:3]
            if(len(points) == 0):
                continue
            print(torch.max(points[:, 0])-torch.min(points[:, 0]), torch.max(points[:, 1])-torch.min(points[:, 1]), torch.max(points[:, 2])-torch.min(points[:, 2]))
            num_points = points.shape[0]
            print(torch.median(points, 1))
            centroid = [torch.median(points[:, 0]), torch.median(points[:, 1]), torch.median(points[:, 2])]
            print(centroid)
            if points.any():
                # pcd = o3d.geometry.PointCloud()
                # pcd.points = o3d.utility.Vector3dVector(points)
                # o3d.visualization.draw_geometries([pcd],
                #                                 zoom=0.3412,
                #                                 front=[0.4257, -0.2125, -0.8795],
                #                                 # lookat=[1.32404573, 2.77499986, -0.12350497],
                #                                 lookat = [1.32719177,  5.7750001,  -0.12350497],
                #                                 up=[-0.0694, -0.9768, 0.2024])
                
                contours = cv2.findContours(
                segmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                contours = contours[0] if len(contours) == 2 else contours[1]
                big_contour = max(contours, key=cv2.contourArea)
                rotrect = cv2.minAreaRect(big_contour)
                (center), (width, height), angle = rotrect
                center3D_depth_value = depth_image[int(center[1]), int(center[0])]
                u = -(center[0]-centerU)/(width)  # image-space coordinate
                v = (center[1]-centerV)/(height)  # image-space coordinate
                # print(center[0], center[1])
                d = depth_image[int(center[1]), int(center[0])]  # depth buffer value
                X2 = np.array([d*fu*u, d*fv*v, d, 1])  # deprojection vector
                p2 = X2  # Inverse camera view to get world coordinates
                centroid_world_coordinate = [p2[0], p2[1], p2[2]]
                # print(centroid_world_coordinate)

                base_coordinate = np.array([0.02, 0, 0], dtype=np.float32)
                suction_coordinates = [base_coordinate]
                object_base_coordinate = centroid_world_coordinate + base_coordinate
                object_suction_coordinate = [object_base_coordinate]

                for angle in range(45, 360, 45):
                    x = base_coordinate[0]*math.cos(angle*math.pi/180) - \
                        base_coordinate[1]*math.sin(angle*math.pi/180)
                    y = base_coordinate[0]*math.sin(angle*math.pi/180) + \
                        base_coordinate[1]*math.cos(angle*math.pi/180)
                    '''
                    Appending all the coordiantes in suction_cooridnates and the object_suction_coordinate is the x and y 3D cooridnate of the object suction points
                    '''
                    suction_coordinates = np.append(
                        suction_coordinates, np.array([[x, y, 0.]]), axis=0)
                    object_suction_coordinate = np.append(object_suction_coordinate, np.array(
                        [[x+centroid_world_coordinate[0], y+centroid_world_coordinate[1], 0.]]), axis=0)
                # print(points.shape[0])
                # print(suction_coordinates)
                points1= np.empty((0, 3), float)
                for suction_points in suction_coordinates:
                    diff = 999.
                    point_cloud_index = np.array([0, 0, 0])
                    for j in range(points.shape[0]):
                        # for k in range(cam_width):
                        compare = abs(points[j][0]-(centroid_world_coordinate[0]+suction_points[0])) + abs(points[j][1]-(centroid_world_coordinate[1]+suction_points[1]))
                        if(diff > compare):
                            diff = compare
                            point_cloud_index = points[j]
                    points1 = np.vstack((points1, [point_cloud_index]))

                    # print(points1)
                # pcd = o3d.geometry.PointCloud()
                # pcd.points = o3d.utility.Vector3dVector(points1)
                # o3d.visualization.draw_geometries([pcd],
                #                                 zoom=0.3412,
                #                                 front=[0.4257, -0.2125, -0.8795],
                #                                 # lookat=[1.32404573, 2.77499986, -0.12350497],
                #                                 lookat = [-0.11878591, 1.34449499, 5.7750001],
                #                                 up=[-0.0694, -0.9768, 0.2024])    



            # if(np.unique(segmask).any() >= 1):
            #     bin_id = "3F"
            #     object = calcualte_suction_score()
            #     score = object.calculator(depth_image, segmask, rgb_image_copy, cam_proj, cam_vinv, bin_id)
            #     print(score)
            '''
            To save rgb image, depth image and segmentation mask (comment this section if you do not want to visualize as it slows down the processing)
            '''
            # self.frame_count += 1
            # plt.imsave(f'{self.current_directory}/../Data/segmask_{self.frame_count}_{i}.png', np.array(segmask*30), cmap=cm.gray)
            # iio.imwrite(f"{self.current_directory}/../Data/rgb_frame_{self.frame_count}_{i}.png", rgb_image_copy)
            # np.save(f"{self.current_directory}/../Data/depth_frame_{self.frame_count}_{i}.npy", depth_image)
        
        '''
        Commands to the arm for eef control
        '''
        self.actions = actions.clone().to(self.device)

        # Split arm and gripper command
        u_arm, u_gripper = self.actions[:, :-1], self.actions[:, -1]

        # print(u_arm, u_gripper)
        # print(self.cmd_limit, self.action_scale)

        # Control arm (scale value first)
        u_arm = u_arm * self.cmd_limit / self.action_scale
        if self.control_type == "osc":
            u_arm = self._compute_osc_torques(dpose=u_arm)
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
