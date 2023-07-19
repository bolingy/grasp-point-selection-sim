import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob
import os
import json
# depth_path = "/home/soofiyan_ws/Documents/Issac_gym_ws/System_Identification_Data/Parallelization-Data-evlaution/0/depth_image_0_63.npy"
# 24 39 60 63 64 65 75 98 101 140 175 179 202 209 214 217 230 235 237 249 290 292 295 301 308 320 321 340 350
file_id = 561
depth_image = np.load(
    "/home/soofiyan_ws/Documents/Issac_gym_ws/System_Identification_Data/Parallelization-Data/0/depth_image_0_"+str(file_id)+".npy")
segmask_image = np.load(
    "/home/soofiyan_ws/Documents/Issac_gym_ws/System_Identification_Data/Parallelization-Data/0/segmask_0_"+str(file_id)+".npy")
rgb_image = np.load(
    "/home/soofiyan_ws/Documents/Issac_gym_ws/System_Identification_Data/Parallelization-Data/0/rgb_0_"+str(file_id)+".npy")

file_pattern = "/home/soofiyan_ws/Documents/Issac_gym_ws/System_Identification_Data/Parallelization-Data/0/json_data_0_*_"+str(file_id)+".json"

# Get a list of files matching the pattern
files = glob.glob(file_pattern)
# Sort the files by creation time
files.sort(key=os.path.getmtime)

count = 0
for filename in files:
    with open(filename, 'r') as f:
        data = json.load(f)
        grasp_point = np.array([data['grasp point'][0], data['grasp point'][1]])
        # print(grasp_point)
        count += 1
        cv2.circle(rgb_image, (int(grasp_point[0]), int(grasp_point[1])), 3, (count*60,count*50,count*50), 2)


# plt.imshow(depth_image)
# plt.show()
plt.imshow(segmask_image)
plt.show()
cv2.imshow("rgb", rgb_image)
cv2.waitKey(0)