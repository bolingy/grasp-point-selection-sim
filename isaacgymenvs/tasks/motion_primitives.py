from isaacgym import gymapi

class Primitives():
    def __init__(self):
        pass

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

    def get_cartesian_move(self, pose: gymapi.Transform, x: float, y: float, z: float)->gymapi.Transform:
        pose.p.x = pose.p.x + x
        pose.p.y = pose.p.y + y
        pose.p.z = pose.p.z + z

        return pose

    def rotate(self, pose: gymapi.Transform, x: float, y: float, z: float)->gymapi.Transform:
        euler = pose.r.to_euler_zyx()
        x += euler[0]
        y += euler[1]
        z += euler[2]
        pose.r = pose.r.from_euler_zyx(x, y, z)
        return pose

    def move_right(self, pose: gymapi.Transform, distance=0.1)->gymapi.Transform:
        return self.get_cartesian_move(pose, distance, 0, 0)
    
    def move_left(self, pose: gymapi.Transform, distance=-0.1)->gymapi.Transform:
        return self.get_cartesian_move(pose, distance, 0, 0)

    def move_up(self, pose: gymapi.Transform, distance=0.1)->gymapi.Transform:
        return self.get_cartesian_move(pose, 0, distance, 0)

    def move_down(self, pose: gymapi.Transform, distance=-0.1)->gymapi.Transform:
        return self.get_cartesian_move(pose, 0, distance, 0)

    def move_forward(self, pose: gymapi.Transform, distance=-0.1)->gymapi.Transform:
        return self.get_cartesian_move(pose, 0, 0, distance)

    def move_back(self, pose: gymapi.Transform, distance=0.1)->gymapi.Transform:
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

    def get_gymapi_transform(self, pose: list) -> gymapi.Transform:
        tmp = gymapi.Transform()
        tmp.p = gymapi.Vec3(pose[0], pose[1], pose[2])
        tmp.r = gymapi.Quat(pose[3], pose[4], pose[5], pose[6])
        pose = tmp
        return pose

    def get_list_transform(self, pose: gymapi.Transform) -> list:
        return [pose.p.x, pose.p.y, pose.p.z, pose.r.x, pose.r.y, pose.r.z, pose.r.w]