from isaacgym import gymtorch
import torch
import json
import numpy as np


class SavingData:
    def __init__(self) -> None:
        pass

    def store_force_and_displacement(self, env_count):
        # force sensor update
        self.force_pre_physics = self.get_force_sensor_values(env_count)

        if self.action_contrib[env_count] <= 1:
            retKey = self.force_list_save.get(env_count)
            if retKey == None:
                self.force_list_save[env_count] = torch.tensor([self.force_pre_physics])
            else:
                force_list_env = self.force_list_save[env_count]
                force_list_env = torch.cat(
                    (force_list_env, torch.tensor([self.force_pre_physics]))
                )
                self.force_list_save[env_count] = force_list_env

            object_disp_env = self.object_disp_save[env_count].copy()
            for object_id in self.selected_object_env[env_count]:
                object_current_pose = (
                    self._root_state[
                        env_count, self._object_model_id[int(object_id.item()) - 1], :
                    ][:7]
                    .type(torch.float)
                    .detach()
                    .clone()
                )
                if int(object_id.item()) not in object_disp_env:
                    object_disp_env[int(object_id.item())] = torch.empty((0, 7)).to(
                        self.device
                    )
                object_disp_env[int(object_id.item())] = torch.cat(
                    [
                        object_disp_env[int(object_id.item())],
                        object_current_pose.unsqueeze(0),
                    ],
                    dim=0,
                )
            self.object_disp_save[env_count] = object_disp_env

    def save_config_grasp_json(
        self, env_count, save_force_disp_config, push_suction_deform_score, unreachable
    ):
        success = False
        oscillation = False
        penetration = False
        if not save_force_disp_config:
            end_effector_forces = []
            object_disp_json_save = {}
            self.force_list_save[env_count] = np.array([])
        else:
            end_effector_forces = self.force_list_save[env_count].tolist()
            oscillation = self.detect_oscillation(end_effector_forces)
            if push_suction_deform_score > torch.tensor(0.1) and oscillation == False:
                success = True
            if push_suction_deform_score == torch.tensor(0):
                penetration = True

            object_disp_json_save = self.object_disp_save[env_count].copy()
            for object_id in self.selected_object_env[env_count]:
                object_disp_json_save[int(object_id.item())] = object_disp_json_save[
                    int(object_id.item())
                ].tolist()

        # saving the grasp point ad its properties if it was a successfull grasp
        json_save = {
            "force_array": self.force_list_save[env_count].tolist(),
            "object_disp": object_disp_json_save,
            "grasp point": self.grasp_point[env_count].tolist(),
            "grasp_angle": self.grasp_angle[env_count].tolist(),
            "dexnet_score": self.dexnet_score[env_count].item(),
            "suction_deformation_score": self.suction_deformation_score[
                env_count
            ].item(),
            "oscillation": oscillation,
            "gripper_score": push_suction_deform_score.item(),
            "success": success,
            "object_id": self.object_target_id[env_count].item(),
            "penetration": penetration,
            "unreachable": unreachable,
        }
        new_dir_name = (
            str(env_count)
            + "_"
            + str(self.track_save[env_count].type(torch.int).item())
        )
        save_dir_json = (
            self.data_path
            + str(self.bin_id)
            + "/"
            + str(env_count)
            + "/json_data_"
            + new_dir_name
            + "_"
            + str(self.config_env_count[env_count].type(torch.int).item())
            + ".json"
        )
        with open(save_dir_json, "w") as json_file:
            json.dump(json_save, json_file)
        self.track_save[env_count] = self.track_save[env_count] + torch.tensor(1)
