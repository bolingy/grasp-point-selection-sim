# Training Models
Here’s a detailed guide to help you navigate through model training and data processing. All essential scripts have been organized and stored thoughtfully to aid your experience.

The fundamental scripts utilized for training the models are neatly placed within the `training/src` directory.

## Using the Provided Dataset (Processed_Data.zip):
If you want to use the provided dataset, train your custom model using `train_grasp_model.py`.
```bash
cd training
python src/train_grasp_model.py
```
If you want to use wandb then you can uncomment code blocks to enable it.

## Using Data from the Current Simulation Setup:
- First, pre-process the data using `process_raw_files_high_res.py`.
- Then, utilize `train_grasp_model_high_res.py` for training.
```bash
cd training
python src/process_raw_files_high_res.py
python src/train_grasp_model_high_res.py
```

## For Reference:
- The pre-processing script for the previous camera configuration can be found in `process_raw_files.py` (just for reference).