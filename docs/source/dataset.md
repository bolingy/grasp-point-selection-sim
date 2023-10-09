# Dataset

Link to the dataset: TODO <br/>
This dataset consist of input which include 4 channel such as 3D point clouds, segmentation Mask. Grasp labels consist of a zero background image with float values at each sampled point. This dataset contains data collected for 3F bin only.

The dataset is organized as follows:

```bash
*_input_data_*.npy - Input data and corresponding label image is *_label_*.npy
*_input_augment1_*.npy - Augmented input data (adding noise to point cloud) and corresponding label image is *_label_*.npy
*_input_augment2_*.npy - Augmented input data (flipping the input image), corresponding flipped label image is *_label_flip_*.npy
```

This dataset is obtained by running `pre_processing` file.

In our simulation setup we have two camera, one for main sampling points and other is embedded in the suction to measure suction deformation score more accurately.
This dataset is used with a specific back camera setup. The camera setup is as follows:
```bash
self.camera_properties_back_cam.horizontal_fov = 80
self.camera_properties_back_cam.width = 1280
self.camera_properties_back_cam.height = 786
self.camera_base_link_translation = [-0.18, 0.175, 0.6]
```
Then this is cropped and passed with resolution 640x480 to the GQCNN network for sampling.

If you want to change the camera setup, you need to change in multiple files,
1. `init_variables_configs.py` - Change the crop coordinates for each bin.<br/>
    Change two variables `check_object_coord_bins` and `crop_coord_bins` for each bin.

2. `gqcnn/cfg/examples/gqcnn_suction.yaml` - Change the configuration of input for GQCNN network.<br/>
    Change the following variables: crop_height, crop_width, image_width, image_height

3. `gqcnn/gqcnn/grasping/image_grasp_sampler.py` - Change the sampling resolution for GQCNN network. <br/>
    Change focal length and depth comprenation in the following variables: fx, fy, depth_left, depth_right
