import glob
import json
import os
import numpy as np
import cv2
import random

data_path = "/home/soofiyan_ws/Documents/Issac_gym_ws/System_Identification_Data/Parallelization-Data-single_model_all_points"
file_pattern = data_path+"/0/json_data_0_*.json"

# Get a list of files matching the pattern
files = glob.glob(file_pattern)
# Sort the files by creation time
files.sort(key=os.path.getmtime)

count = 0
last_number = 0
# for filename in files:
#     underscore_parts = filename.split('_')
#     dot_parts = underscore_parts[-1].split('.')
#     count += 1
#     if(count == 1 and last_number == 0):
#         last_number = int(dot_parts[0])
#     elif(last_number != int(dot_parts[0]) and count != 16 and count != 1):
#         for f in glob.glob(data_path+"/0/json_data_0_*"+str(last_number)+".json"):
#             os.remove(f)
#         last_number = int(dot_parts[0])
#         count = 1
#     elif(last_number != int(dot_parts[0])):
#         last_number = int(dot_parts[0])
#         count = 1


file_pattern = data_path+"/0/json_data_0_*.json"

# Get a list of files matching the pattern
files = glob.glob(file_pattern)
# Sort the files by creation time
files.sort(key=os.path.getmtime)

l0, l1, l2, l3, l4 = [], [], [], [], []
m0, m1, m2, m3, m4 = 0., 0., 0., 0., 0.
ml0, ml1, ml2, ml3, ml4 = [], [], [], [], []
min_score = [100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100]
env_count = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# prev_config_list = [48, 95, 96, 100, 117, 169, 186, 201, 209, 262, 200, 264, 295, 28, 108, 322, 166, 164, 212, 71]
prev_config_list = []
config_list_single = np.array([])
config_list = np.load("coonfig_list_single_method.npy")
print(config_list)
counter = 0
success_count = 0

for filename in files:
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
            underscore_parts = filename.split('_')
            dot_parts = underscore_parts[-1].split('.')
            # print(dot_parts[0])

            if(int(dot_parts[0]) in config_list):
                success, pen, deform, osill = data['success'], data['penetration'], float(data['suction_deformation_score']), data['oscillation']
                result =  (success or (not success and pen and deform > 0.))
                print(filename, result)
                if counter%5  < 5:
                    if result:
                        m0 += 1
                    l0.append(result)
                # elif counter%15 < 10:
                #     if result:  
                #         m1 += 1
                #     l1.append(result)
                #     # if(result == False):
                # elif counter%15 < 15:
                #     if result:  
                #         m2 += 1
                #     l2.append(result)
                counter += 1
                if(result):
                    success_count += 1
                # if(counter >= 1300):
                #     break
                if(counter%5 == 0):
                    ml0.append(m0/5)
                    # ml1.append(m1/5)
                    # ml2.append(m2/5)
                    m0, m1, m2 = 0, 0, 0

                    config_list_single = np.append(config_list_single, int(dot_parts[0]))
                    # for i in range(len(min_score)):
                    if(int(dot_parts[0]) not in prev_config_list):
                        random_number = random.uniform(0, 1)
                        segmask = np.load(f"{data_path}/0/segmask_0_{int(dot_parts[0])}.npy")
                        if(max(min_score) > success_count and random_number<0.1 and len(np.unique(segmask)) > 1):
                            index = np.argmax(min_score)
                            min_score[index] = success_count
                            env_count[index] = int(dot_parts[0])
                            print(success_count)
                        success_count = 0
                        print('-------------------------------------------------------------------------------------------------------------------------------------------')
    except Exception as e:
        print(f"Failed to load file {filename}. Error: {str(e)}")

# np.save("coonfig_list_single_method.npy", config_list_single)
l0, l1, l2 = np.array(l0), np.array(l1), np.array(l2)
print(l0.shape, l1.shape, l2.shape)
print(l0.sum(), l1.sum(), l2.sum())
print(l0.sum()/l0.shape, l1.sum()/l1.shape, l2.sum()/l2.shape)
# print(ml0, ml1, ml2)
ml0 = np.array(ml0)
# ml1 = np.array(ml1)
# ml2 = np.array(ml2)

print(np.mean(ml0), np.std(ml0))
# print(np.mean(ml1), np.std(ml1))
# print(np.mean(ml2), np.std(ml2))

print(min_score, env_count)
# for i in env_count:
#     rgb = np.load(f"{data_path}/0/rgb_0_{i}.npy")
#     rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
#     cv2.imwrite(f"{data_path}/regular_cases_2/rgb_0_{i}.png", rgb)
#     cv2.imshow("rgb", rgb)
#     cv2.waitKey(0)


# For mean and std dev
