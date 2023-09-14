from isaacgym import gymapi
import torch
import numpy as np
from .motion_primitives import Primitives

DEFAULT_OSC_DIST = 0.3
DEFAULT_MIN_DIST_MUL = 0.05
class PrimitiveCombo():
    def __init__(self, num_envs, init_pose, device, args, prim_combo_type="swipe"):
        def get_prim_list(self, prim_combo_type, args):
            if prim_combo_type == "swipe":
                self.prim_list = ["in"]
                if prim_target_dist_y > 0:
                    self.prim_list.append("left")
                elif prim_target_dist_y < 0:
                    self.prim_list.append("right")
                prim_target_dist_y = abs(prim_target_dist_y)
                self.prim_list.append("out")
                self.move_dist_list = [args.prim_target_dist_x, prim_target_dist_y, args.prim_target_dist_x]
        self.device = device
        self.num_envs = num_envs
        self.primitive = Primitives(self.num_envs, init_pose, device=self.device)
        self.prim_combo_type = prim_combo_type
        self.prim_list = []
        self.move_dist_list = []
        self.curr_prim_num = 0
        get_prim_list(self, prim_combo_type, args)



    def move(self, curr_eef_pos, curr_eef_quat):
        if self.curr_prim_num >= len(self.prim_list):
            return torch.tensor([[0., 0., 0., 0., 0., 0.]]), True
        u_arm_temp, _, done = self.primitive.move_w_ori(self.prim_list[self.curr_prim_num], 
                                                        curr_eef_pos, 
                                                        curr_eef_quat, 
                                                        self.move_dist_list[self.curr_prim_num])
        if done:
            self.curr_prim_num += 1
        return u_arm_temp, False