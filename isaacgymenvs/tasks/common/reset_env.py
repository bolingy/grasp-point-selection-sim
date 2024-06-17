import torch

class EnvReset:
    def check_reset_conditions(self, envs):
        segmask = self.torch_segmask_cameras[envs].squeeze()
        # Cropping bin area from the segmask
        segmask_bin_crop = segmask
        # Calculating the number of objects spawned in the bin within the boundaries of the bin
        objects_spawned = len(torch.unique(segmask_bin_crop)) - 1
        
        for env_count in envs:
            env_count = env_count.item()
            total_objects = len(self.selected_object_env[env_count])
            object_coords_match = torch.count_nonzero(segmask) == torch.count_nonzero(
                segmask_bin_crop
            )
            total_position_error, obj_drop_status = self.calculate_objects_position_error(env_count)
            object_mask_area = self.min_area_object(segmask, segmask_bin_crop)
            
            # Check conditions for resetting the environment.
            conditions_messages = [
                (
                    object_mask_area < 1000,
                    f"Object in environment {env_count} not visible in the camera (due to occlusion) \
                        with area {object_mask_area}",
                ),
                (
                    (not object_coords_match) or (total_objects != objects_spawned),
                    f"Object in environment {env_count} extends beyond the bin's boundaries",
                ),
                # (
                #     torch.abs(total_position_error) > self.POSE_ERROR_THRESHOLD,
                #     f"Object in environment {env_count}, not visible in the camera (due to occlusion) \
                #         with area {object_mask_area}",
                # ),
                (
                    obj_drop_status,
                    f"Object falled down in environment {env_count}, where total objects are {total_objects} \
                        and only {objects_spawned} were spawned inside the bin",
                ),
            ]

            for condition, message in conditions_messages:
                
                if condition:
                    print("Resetting environment", env_count, message)
                    self.env_complete_reset[env_count] = 1
                    if self.RL_flag[env_count] == 1:
                        self.free_envs_list[env_count] = 1
                    break
            # for condition, message in conditions_messages:
            #     if condition:
            #         env_complete_reset = self.reset_env_with_log(
            #             env_count, message, env_complete_reset
            #         )
            #         break
        return segmask
    
    def min_area_object(self, segmask_check, segmask_bin_crop):
        """
        Find the object with the minimum area in the segmask.
        """
        areas = [
            (segmask_check == int(object_id.item())).sum()
            for object_id in torch.unique(segmask_bin_crop)
        ]
        object_mask_area = min(areas, default=torch.tensor(1000))
        return object_mask_area