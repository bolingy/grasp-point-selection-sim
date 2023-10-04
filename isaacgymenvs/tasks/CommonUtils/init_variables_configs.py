from isaacgymenvs.tasks.base.vec_task import VecTask
import os
import torch
import yaml
import assets.urdf_models.models_data as md
from isaacgym.torch_utils import *

class InitVariablesConfigs(VecTask):
    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render, bin_id, data_path=None):
        self.cfg = cfg

        self.max_episode_length = self.cfg["env"]["episodeLength"]

        self.action_scale = self.cfg["env"]["actionScale"]
        self.aggregate_mode = self.cfg["env"]["aggregateMode"]

        # Bin ID (bin size)
        self.bin_id = bin_id

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

        VecTask.__init__(self, config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id,
                         headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)

        self.ur16e_default_dof_pos = to_torch(
            [-1.57, 0, 0, 0, 0, 0, 0], device=self.device
        )

        # Constants
        # TODO: explain all the constants here
        self.COOLDOWN_FRAMES = 150
        self.RETRY_OBJECT_STABILITY = torch.tensor(3)
        self.OBJECT_FALL_HEIGHT = torch.tensor(0.5)
        self.POSE_ERROR_THRESHOLD = torch.tensor(0.0055)
        self.OBJECT_MASK_THRESHOLD = 1000
        self.DEPTH_IMAGE_OFFSET = 0.2
        self.GRASP_LIMIT = 10
        self.DEFAULT_EE_VEL = torch.tensor(0.15)

        # System IDentification data results
        self.data_path = data_path or os.path.expanduser(
            "~/temp/grasp_data_05/")
        for env_number in range(self.num_envs):
            new_dir_path = os.path.join(
                self.data_path, f"{self.bin_id}/{env_number}/")
            os.makedirs(new_dir_path, exist_ok=True)

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
        self.ee_vel = torch.ones(self.num_envs)*self.DEFAULT_EE_VEL
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

        self.selected_object_env = {}

        self.track_save = torch.zeros(self.num_envs)
        self.config_env_count = torch.zeros(self.num_envs)
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

        self.env_reset_id_env = torch.ones(self.num_envs)

        self.check_object_coord_bins = {
            "3F": [113, 638, 366, 906],
            "3E": [273, 547, 366, 906],
            "3H": [226, 589, 366, 906],
        }

        self.crop_coord_bins = {
            "3F": [0, 720, 0, 1280],
            "3E": [0, 720, 0, 1280],
            "3H": [0, 720, 0, 1280],
        }

        self.object_bin_prob_spawn = {
            "3F": [1, 0.4, 1],
            "3E": [0.05, 0.45, 1],
            "3H": [0.05, 0.45, 1],
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
        self.reset_idx_init(torch.arange(self.num_envs, device=self.device))

        # Refresh tensors
        self.refresh_env_tensors()
        self.current_directory = os.getcwd()
