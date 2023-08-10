import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob
import os
import json
# depth_path = "/home/soofiyan_ws/Documents/Issac_gym_ws/System_Identification_Data/Parallelization-Data-evlaution/0/depth_image_0_63.npy"
# 5 challenging cases: 14, 17, 178, 279, 296
# impossible cases: 34, 154, 182, 246, 284
# regular cases: 49, 75, 100, 108, 129
# regular cases 20, 48, 78, 94, 97, 176, 189
array_config = [20, 48, 97, 176, 189]

for i in array_config:
    file_id = i
    depth_image = np.load(
        "/home/soofiyan_ws/Documents/Issac_gym_ws/System_Identification_Data/Parallelization-Data_3_methods/0/depth_image_0_"+str(file_id)+".npy")
    segmask_image = np.load(
        "/home/soofiyan_ws/Documents/Issac_gym_ws/System_Identification_Data/Parallelization-Data_3_methods/0/segmask_0_"+str(file_id)+".npy")
    rgb_image = np.load(
        "/home/soofiyan_ws/Documents/Issac_gym_ws/System_Identification_Data/Parallelization-Data_3_methods/0/rgb_0_"+str(file_id)+".npy")

    file_pattern = "/home/soofiyan_ws/Documents/Issac_gym_ws/System_Identification_Data/Parallelization-Data_3_methods/0/json_data_0_*_"+str(file_id)+".json"

    # Get a list of files matching the pattern
    files = glob.glob(file_pattern)
    # Sort the files by creation time
    files.sort(key=os.path.getmtime)

    success_count = 0
    count = 0
    for filename in files:
        with open(filename, 'r') as f:
            data = json.load(f)
            grasp_point = np.array([data['grasp point'][0], data['grasp point'][1]])
            # print(grasp_point)
            count += 1
            color = 100*(int((count%15)/5))
            success, pen, deform, osill = data['success'], data['penetration'], float(data['suction_deformation_score']), data['oscillation']
            result =  (success or (not success and pen and deform > 0.))
            print(result)
            if(result == True):
                success_count += 1
            cv2.circle(rgb_image, (int(grasp_point[0]), int(grasp_point[1])), 3, (color,color,color), 2)
            

    print(success_count)
    # plt.imshow(depth_image)
    # plt.show()
    # plt.imshow(segmask_image)
    # plt.show()
    rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
    cv2.imshow("rgb", rgb_image)
    cv2.waitKey(0)
    # save = input("Save? y/n")
    # if(save == 'y'):
    #     cv2.imwrite(f"/home/soofiyan_ws/Documents/Issac_gym_ws/System_Identification_Data/Parallelization-Data_3_methods/regular_cases_2/rgb_target_0_{file_id}.png", rgb_image)