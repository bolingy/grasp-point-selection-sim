from isaacgym import gymapi
import torch
import numpy as np


class Primitives():
    def __init__(self, init_pose):
        self.target_pose = init_pose
        self.current_pose = init_pose
        self.min_distance_to_goal = 0.01
    def move(self, action, current_pose):
        # if action != "done" and np.linalg.norm(self.target_pose - np.ravel([self.current_pose[0], self.current_pose[0], self.current_pose[0]])) > self.min_distance_to_goal:

        if action == "right":
            return self.move_right(current_pose)

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

    def get_cartesian_move(self, current_pose: torch.Tensor, x: float, y: float, z: float) -> torch.Tensor:
        current_pose[:,[0]] = current_pose[:,[0]] + x
        current_pose[:,[1]] = current_pose[:,[1]] + y
        current_pose[:,[2]] = current_pose[:,[2]] + z
        return current_pose


    def rotate(self, pose: gymapi.Transform, x: float, y: float, z: float)->gymapi.Transform:
        euler = pose.r.to_euler_zyx()
        x += euler[0]
        y += euler[1]
        z += euler[2]
        pose.r = pose.r.from_euler_zyx(x, y, z)
        return pose

    def move_right(self, pose: torch.Tensor, distance=0.01) -> torch.Tensor:
        return torch.tensor(10 * [[0.01, 0., 0.]])
        # return self.get_cartesian_move(pose, distance, 0, 0)
    
    def move_left(self, pose: gymapi.Transform, distance=-0001)->gymapi.Transform:
        return self.get_cartesian_move(pose, distance, 0, 0)

    def move_up(self, pose: gymapi.Transform, distance=0.01)->gymapi.Transform:
        return self.get_cartesian_move(pose, 0, distance, 0)

    def move_down(self, pose: gymapi.Transform, distance=-0001)->gymapi.Transform:
        return self.get_cartesian_move(pose, 0, distance, 0)

    def move_forward(self, pose: gymapi.Transform, distance=-0.0001)->gymapi.Transform:
        return self.get_cartesian_move(pose, 0, 0, distance)

    def move_back(self, pose: gymapi.Transform, distance=0.0001)->gymapi.Transform:
        return self.get_cartesian_move(pose, 0, 0, distance)

    def push(self, pose: gymapi.Transform, distance=-0.2)->gymapi.Transform:
        return self.move_forward(pose, distance)

    def shake(self, pose: gymapi.Transform, distance=-0.1)->gymapi.Transform:
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

    
