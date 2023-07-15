import glob
import json
import os
import numpy as np

file_pattern = "/home/soofiyan_ws/Documents/Issac_gym_ws/System_Identification_Data/Parallelization-Data-evlaution/0/json_data_0_*.json"

# Get a list of files matching the pattern
files = glob.glob(file_pattern)
# Sort the files by creation time
files.sort(key=os.path.getmtime)
#./0/json_data_0_106_103.json


# # fix ordering issue
# idx1 = files.index('./0/json_data_0_106_103.json')
# idx2 = files.index('./0/json_data_0_106_104.json')

# # remove the string at idx1
# str1 = files.pop(idx1)

# # insert str1 before idx2
# files.insert(idx2, str1)


l0, l1, l2 = [], [], []
counter = 0
for filename in files:
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
            success, pen, deform, osill = data['success'], data['penetration'], float(data['suction_deformation_score']), data['oscillation']
            result =  (success or (not success and pen and deform > 0.)) and not osill
            if counter%3 == 0:
                l0.append(result)
                print(filename, result)
            elif counter%3 == 1:
                l1.append(result)
            else:
                l2.append(result)
                print('--------------------------------')
            counter += 1
    except Exception as e:
        print(f"Failed to load file {filename}. Error: {str(e)}")

l0, l1, l2 = np.array(l0), np.array(l1), np.array(l2)
print(l0.shape, l1.shape, l2.shape)
print(l0.sum(), l1.sum(), l2.sum())
print(l0.sum()/l0.shape, l1.sum()/l1.shape, l2.sum()/l2.shape)

indexes = np.where((l0 == True) & (l1 == False) & (l2 == False))[0]
print('before', indexes)
indexes = indexes * 3
print('after', indexes)
print(np.array(files)[indexes])

# for i in indexes:
#     print(files[i], )