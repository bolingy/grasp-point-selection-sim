# Dataset

Link to the dataset: []() <br/>
This dataset consist of input which include 4 channel such as 3D point clouds, segmentation Mask. Grasp labels consist of a zero background image with float values at each sampled point. This dataset contains data collected for 3F bin only.

The dataset is organized as follows:

```bash
`*_input_data_*.npy` - Input data and corresponding label image is `*_label_*.npy`
`*_input_augment1_*.npy` - Augmented input data (adding noise to point cloud) and corresponding label image is `*_label_*.npy`
`*_input_augment2_*.npy` - Augmented input data (fliping the input image), corresponding label iamge is `*_label_flip_*.npy`

```

This dataset is obtained by running `pre_processing` file. 