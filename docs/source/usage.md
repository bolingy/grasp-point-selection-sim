# Getting Started

## Data Collection
BEfore running the data collection file, you need to create a folder outside your grasp-point-selection-sim folder. This folder will be used to store the data collected from the simulation. You can create this folder using the following command:
```bash
mkdir <PATH_TO_YOUR_WORKSPACE>/grasp-point-selection-sim/scenario_grasp_configurations
```

Kickstart your experience with a basic usage example:

**Usage**:
```bash
./dynamo_grasp.sh --bin-id BIN_ID --num-envs NUM_ENVS
```
**Parameters**:
```bash
BIN ID :  Identifier for the bin (e.g., 3E, 3F, 3H)
NUM ENVS :  Integer representing the number of environments to run.
            Select this number based on your computational resources.
```

### Bin Size Configuration
DYNAMO GRASP allows customization of bin sizes (such as **3E**, **3F**, and **3H**), termed as **bin_id**. These bins are centrally located within the vertical pod. You can incorporate additional bin sizes by modifying the `init_variables_configs.py` file and adding collision primitives in the `configs` folder.
#### Example Configuration Changes:
You have to change the following variables in the `init_variables_configs.py` file to add a new bin size:
```python
        # Probabilities for spawning 1, 2, or 3 objects in different bins.
        self.object_bin_prob_spawn = {
            "3F": [0.025, 0.4, 1],
            "3E": [0.05, 0.45, 1],
            "3H": [0.05, 0.45, 1],
        }

        # Heights to spawn objects in respective bins.
        self.object_height_spawn = {
            "3F": 1.3,
            "3E": 1.4,
            "3H": 1.4,
        }

        # Coords to validate object presence within bins.
        self.check_object_coord_bins = {
            "3F": [113, 638, 366, 906],
            "3E": [273, 547, 366, 906],
            "3H": [226, 589, 366, 906],
        }

        # Coords to crop images for each bin type.
        self.crop_coord_bins = {
            "3F": [0, 720, 0, 1280],
            "3E": [0, 720, 0, 1280],
            "3H": [0, 720, 0, 1280],
        }

        # Identifying bins with a tendency to spawn smaller objects.
        self.smaller_bins_set = ["3E", "3H"]
```
#### Bin Size Adjustment:
Prior to this, you have to also add collision primitives in `configs` folder. You can refer to the existing collision primitives in the `configs` folder.

Adjust the pose of horizontal planes such that you get custom bin size, you can refer to the following example:
```yaml
# <bin id> bin adjustment
      cube110:
        dims: [1, 1, 0.001]
        pose: [1.0914, 0.0502, 0.7156, 0.707, 0.707, 0.0, 0.0]
      cube111:
        dims: [1, 1, 0.001]
        pose: [1.0914, 0.0501, 0.6013, 0.707, 0.707, 0.0, 0.0]
```

Adjust the **pose** attribute's z-axis to alter bin sizes. While using other bins is possible, it's crucial to perform a reachability analysis for the robot before proceeding.
```{note} 
Remember to adapt the input arguments for the `dynamo_grasp.sh` bash script accordingly.
```