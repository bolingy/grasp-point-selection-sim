import torch
from homogeneous_trasnformation_and_conversion.rotation_conversions import *

class ErrorCalculator:
    def __init__(self) -> None:
        pass
    
    def detect_contact_non_target_object(self, env_count, _all_objects_current_pose, contact_exist):
        # TODO: be more specific about the contact
        '''
        detecting contacts with other objects before it contacts with the target object
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
        if ((_all_object_pose_error > torch.tensor(0.0075)) and contact_exist == torch.tensor(0)):
            env_list_reset_arm_pose = torch.cat(
                (env_list_reset_arm_pose, torch.tensor([env_count])), axis=0)
            env_list_reset_objects = torch.cat(
                (env_list_reset_objects, torch.tensor([env_count])), axis=0)
            print(
                f"Object in environment {env_count} moved without contact to target object by {_all_object_pose_error} meters")

            self.save_config_grasp_json(
                env_count, False, torch.tensor(0), True)

    def calculate_angle_error(self, env_count, env_list_reset_arm_pose, env_list_reset_objects):
        if (self.action_contrib[env_count] == 0):
            angle_error = quaternion_to_euler_angles(self._eef_state[env_count][3:7], "XYZ", degrees=False) - torch.tensor(
                [0, -self.grasp_angle[env_count][1], self.grasp_angle[env_count][0]]).to(self.device)
            if (torch.max(torch.abs(angle_error)) > torch.deg2rad(torch.tensor(10.0))):
                # encountered the arm insertion constraint
                env_list_reset_arm_pose = torch.cat(
                    (env_list_reset_arm_pose, torch.tensor([env_count])), axis=0)
                env_list_reset_objects = torch.cat(
                    (env_list_reset_objects, torch.tensor([env_count])), axis=0)
                print(
                    env_count, "reset because of arm insertion angular constraint")

                self.save_config_grasp_json(
                    env_count, False, torch.tensor(0), True)

            self.force_contact_flag[env_count] = torch.tensor(
                1).type(torch.bool)
        return env_list_reset_arm_pose, env_list_reset_objects