import torch
import numpy as np
from isaacgym import gymtorch

class SaveData:

    def store_force_and_displacement(self, envs):
        """
        Continuously storing the force values and object displacement
        """
        for env_count in envs:
            env_count = env_count.item()
            self.force_pre_physics = self.get_force_sensor_values(env_count)

            if (self.action_contrib[env_count] <= 1):
                retKey = self.force_list_save.get(env_count)
                if (retKey == None):
                    self.force_list_save[env_count] = torch.tensor(
                        [self.force_pre_physics])
                else:
                    force_list_env = self.force_list_save[env_count]
                    force_list_env = torch.cat(
                        (force_list_env, torch.tensor([self.force_pre_physics])))
                    self.force_list_save[env_count] = force_list_env
                object_disp_env = self.object_disp_save[env_count].copy()
                for object_id in self.selected_object_env[env_count]:
                    object_current_pose = self._root_state[env_count, self._object_model_id[int(object_id.item())-1], :][:7].type(
                        torch.float).detach().clone()
                    if (int(object_id.item()) not in object_disp_env):
                        object_disp_env[int(object_id.item())] = torch.empty(
                            (0, 7)).to(self.device)
                    object_disp_env[int(object_id.item())] = torch.cat(
                        [object_disp_env[int(object_id.item())], object_current_pose.unsqueeze(0)], dim=0)
                self.object_disp_save[env_count] = object_disp_env
                
    def get_force_sensor_values(self, env_count):
        _fsdata = self.gym.acquire_force_sensor_tensor(self.sim)
        fsdata = gymtorch.wrap_tensor(_fsdata)
        return -fsdata[env_count][2].detach()
