import glob
import json
import os
import numpy as np
import cv2
import random

file_pattern = "/home/soofiyan_ws/Documents/Issac_gym_ws/System_Identification_Data/Parallelization-Data_3_methods/0/json_data_0_*.json"

# Get a list of files matching the pattern
files = glob.glob(file_pattern)
# Sort the files by creation time
files.sort(key=os.path.getmtime)

count = 0
last_number = 0
for filename in files:
    underscore_parts = filename.split('_')
    dot_parts = underscore_parts[-1].split('.')
    count += 1
    if(count == 1 and last_number == 0):
        last_number = int(dot_parts[0])
    elif(last_number != int(dot_parts[0]) and count != 16 and count != 1):
        for f in glob.glob("/home/soofiyan_ws/Documents/Issac_gym_ws/System_Identification_Data/Parallelization-Data_3_methods/0/json_data_0_*"+str(last_number)+".json"):
            os.remove(f)
        last_number = int(dot_parts[0])
        count = 1
    elif(last_number != int(dot_parts[0])):
        last_number = int(dot_parts[0])
        count = 1


file_pattern = "/home/soofiyan_ws/Documents/Issac_gym_ws/System_Identification_Data/Parallelization-Data_3_methods/0/json_data_0_*.json"

# Get a list of files matching the pattern
files = glob.glob(file_pattern)
# Sort the files by creation time
files.sort(key=os.path.getmtime)

l0, l1, l2, l3, l4 = [], [], [], [], []
min_score = [100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100]
env_count = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# prev_config_list = [48, 95, 96, 100, 117, 169, 186, 201, 209, 262, 200, 264, 295, 28, 108, 322, 166, 164, 212, 71]
prev_config_list = []

counter = 0
success_count = 0
for filename in files:
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
            underscore_parts = filename.split('_')
            dot_parts = underscore_parts[-1].split('.')
            # print(dot_parts[0])
            success, pen, deform, osill = data['success'], data['penetration'], float(data['suction_deformation_score']), data['oscillation']
            result =  (success or (not success and pen and deform > 0.))
            print(filename, result)
            if counter%15  < 5:
                l0.append(result)
            elif counter%15 < 10:
                l1.append(result)
                # if(result == False):
            elif counter%15 < 15:
                l2.append(result)
            counter += 1
            if(result):
                success_count += 1
            # if(counter >= 2600):
            #     break
            if(counter%15 == 0):
                # for i in range(len(min_score)):
                if(int(dot_parts[0]) not in prev_config_list):
                    random_number = random.uniform(0, 1)
                    segmask = np.load(f"/home/soofiyan_ws/Documents/Issac_gym_ws/System_Identification_Data/Parallelization-Data_3_methods/0/segmask_0_{int(dot_parts[0])}.npy")
                    if(max(min_score) > success_count and random_number<0.1 and len(np.unique(segmask)) > 1):
                        index = np.argmax(min_score)
                        min_score[index] = success_count
                        env_count[index] = int(dot_parts[0])
                        print(success_count)
                    success_count = 0
                    print('-------------------------------------------------------------------------------------------------------------------------------------------')
    except Exception as e:
        print(f"Failed to load file {filename}. Error: {str(e)}")


l0, l1, l2 = np.array(l0), np.array(l1), np.array(l2)
print(l0.shape, l1.shape, l2.shape)
print(l0.sum(), l1.sum(), l2.sum())
print(l0.sum()/l0.shape, l1.sum()/l1.shape, l2.sum()/l2.shape)


print(min_score, env_count)
for i in env_count:
    rgb = np.load(f"/home/soofiyan_ws/Documents/Issac_gym_ws/System_Identification_Data/Parallelization-Data_3_methods/0/rgb_0_{i}.npy")
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    cv2.imwrite(f"/home/soofiyan_ws/Documents/Issac_gym_ws/System_Identification_Data/Parallelization-Data_3_methods/regular_cases_2/rgb_0_{i}.png", rgb)
    cv2.imshow("rgb", rgb)
    cv2.waitKey(0)