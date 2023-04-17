from isaacgym import gymapi
import torch
import numpy as np

DEFAULT_DIST = 0.01

class Primitives():
    def __init__(self, num_envs, init_pose):
        self.target_pose = init_pose
        self.current_pose = init_pose
        self.min_distance_to_goal = 0.1
        self.num_envs = num_envs
        self.state = False

    def move(self, action, current_pose, target_pose):
        self.current_pose = current_pose 
        if self.state == False:
            self.target_pose = torch.clone(current_pose)
            self.target_pose[:, [1]] = self.target_pose[:, [1]] - 0.1

        if torch.linalg.norm(self.target_pose - self.current_pose) > self.min_distance_to_goal:
            pose_diff = torch.clone(self.target_pose - self.current_pose)
            if action == "right":
                print(action, torch.sum(pose_diff))
                return self.move_right(pose_diff), "right"
            elif action == "left":
                print(action, torch.sum(pose_diff))
                return self.move_left(pose_diff), "left"
            elif action == "up":
                return self.move_up(pose_diff), "up"
            elif action == "down":
                return self.move_down(pose_diff), "down"
            
        return torch.tensor(10 * [[0., 0., 0.]]), "done"

    def set_target_pose(self, target_pose):
        self.target_pose = target_pose

    def get_target_pose(self):
        return self.target_pose

    def square_pattern(self, pose: gymapi.Transform, action: str):
        if action == "up": 
            return self.move_up(pose), "right"
        elif action == "right":
            return self.move_right(pose), "down"
        elif action == "down":
            return self.move_down(pose), "left"
        elif action == "left":
            return self.move_left(pose), "up"
        else:
            return pose, "done"

    def get_cartesian_move(self, x: float, y: float, z: float) -> torch.Tensor:
        # current_pose[:,[0]] = current_pose[:,[0]] + x
        # current_pose[:,[1]] = current_pose[:,[1]] + y
        # current_pose[:,[2]] = current_pose[:,[2]] + z
        # return current_pose
        return torch.cat((x, y, z), 1)

    def rotate(self, pose: gymapi.Transform, x: float, y: float, z: float)->gymapi.Transform:
        euler = pose.r.to_euler_zyx()
        x += euler[0]
        y += euler[1]
        z += euler[2]
        pose.r = pose.r.from_euler_zyx(x, y, z)
        return pose

    def move_right(self, pose_diff, distance = -DEFAULT_DIST) -> torch.Tensor:
        movement = torch.zeros_like(pose_diff[:, [1]]) + distance
        return self.get_cartesian_move(pose_diff[:,[0]], movement, pose_diff[:,[2]])
    
    def move_left(self, pose_diff, distance= DEFAULT_DIST) -> torch.Tensor:
        movement = torch.zeros_like(pose_diff[:, [1]]) + distance
        return self.get_cartesian_move(pose_diff[:,[0]], movement, pose_diff[:,[2]])

    def move_up(self, pose_diff, distance=DEFAULT_DIST)-> torch.Tensor:
        return self.get_cartesian_move(pose, 0, distance, 0)

    def move_down(self, pose_diff, distance=-DEFAULT_DIST)-> torch.Tensor:
        movement = torch.zeros_like(pose_diff[:, [2]]) + distance
        return self.get_cartesian_move(pose_diff[:,[0]], pose_diff[:,[1]], movement)

    def move_forward(self, pose_diff, distance=-DEFAULT_DIST)-> torch.Tensor:
        movement = torch.zeros_like(pose_diff[:, [0]]) + distance
        return self.get_cartesian_move(movement, pose_diff[:,[1]], pose_diff[:,[2]])

    def move_back(self, pose_diff, distance=DEFAULT_DIST)-> torch.Tensor:
        movement = torch.zeros_like(pose_diff[:, [0]]) - distance
        return self.get_cartesian_move(movement, pose_diff[:,[1]], pose_diff[:,[2]])

    def push(self, pose_diff, distance=-0.2)-> torch.Tensor:
        return self.move_forward(pose, distance)

    def shake(self, pose_diff, distance=-0.1)-> torch.Tensor:
        pass

    def move_in():
        pass

    def lift():
        pass

    def move_out():
        pass

    def move_to_drop():
        pass

    def pluck():
        pass

    def hammer_on():
        pass

    # push VERB
    # to move someone or something away from you, or from their previous position, using part of your body, especially your hands

    # shove VERB
    # to push someone or something with force

    # heave VERB
    # to push, pull, or lift a heavy object using a lot of effort

    # thrust aside PHRASAL VERB
    # to push someone or something to one side

    # prod VERB
    # to push someone or something quickly with your finger, or with an object that has a long thin end

    # jar VERB
    # to push something firmly and suddenly against something else, usually accidentally

    # jab VERB
    # to push something with a sudden straight movement, usually with your finger, your elbow, or a narrow object

    # thrust VERB
    # to move somewhere by pushing hard and quickly

    # shoulder VERB
    # to push someone with your shoulder

    # strain VERB
    # to push against something very hard

    def get_gymapi_transform(self, pose: list) -> gymapi.Transform:
        tmp = gymapi.Transform()
        tmp.p = gymapi.Vec3(pose[0], pose[1], pose[2])
        tmp.r = gymapi.Quat(pose[3], pose[4], pose[5], pose[6])
        pose = tmp
        return pose

    def get_list_transform(self, pose: gymapi.Transform) -> list:
        return [pose.p.x, pose.p.y, pose.p.z, pose.r.x, pose.r.y, pose.r.z, pose.r.w]

    
