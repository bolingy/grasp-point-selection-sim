import glob
import json
import os
import numpy as np

file_pattern = "/home/soofiyan_ws/Documents/Issac_gym_ws/System_Identification_Data/Parallelization-Data/0/json_data_0_*.json"

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
    elif(last_number != int(dot_parts[0]) and count != 6 and count != 1):
        for f in glob.glob("/home/soofiyan_ws/Documents/Issac_gym_ws/System_Identification_Data/Parallelization-Data/0/json_data_0_*"+str(last_number)+".json"):
            os.remove(f)
        last_number = int(dot_parts[0])
        count = 1
    elif(last_number != int(dot_parts[0])):
        last_number = int(dot_parts[0])
        count = 1

file_pattern = "/home/soofiyan_ws/Documents/Issac_gym_ws/System_Identification_Data/Parallelization-Data/0/json_data_0_*.json"

# Get a list of files matching the pattern
files = glob.glob(file_pattern)
# Sort the files by creation time
files.sort(key=os.path.getmtime)

l0, l1, l2, l3, l4 = [], [], [], [], []
counter = 0
for filename in files:
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
            success, pen, deform, osill = data['success'], data['penetration'], float(data['suction_deformation_score']), data['oscillation']
            result =  (success or (not success and pen and deform > 0.))
            print(filename, result)
            if counter%5 == 0:
                l0.append(result)
            elif counter%5 == 1:
                l1.append(result)
                # if(result == False):
            elif counter%5 == 2:
                l2.append(result)
            elif counter%5 == 3:
                l3.append(result)
            else:
                l4.append(result)
                print('--------------------------------')
            counter += 1
    except Exception as e:
        print(f"Failed to load file {filename}. Error: {str(e)}")

l0, l1, l2, l3, l4 = np.array(l0), np.array(l1), np.array(l2), np.array(l3), np.array(l4)
print(l0.shape, l1.shape, l2.shape, l3.shape, l4.shape)
print(l0.sum(), l1.sum(), l2.sum(), l3.sum(), l4.sum())
print(l0.sum()/l0.shape, l1.sum()/l1.shape, l2.sum()/l2.shape, l3.sum()/l3.shape, l4.sum()/l4.shape)
