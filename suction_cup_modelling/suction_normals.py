import numpy as np
import math
from scipy.spatial.transform import Rotation as R

xyz_points = np.array([[0.1458, 0.1007, 0.9735],
        [0.1397, 0.1135, 0.9859],
        [0.1243, 0.1191, 0.9912],
        [0.1100, 0.1135, 0.9859],
        [0.1042, 0.1007, 0.9735],
        [0.1113, 0.0887, 0.9693],
        [0.1248, 0.0826, 0.9770],
        [0.1392, 0.0887, 0.9693]])
x_a = -0.0699
y_a = 26.7482
r = R.from_euler('yxz', [x_a, y_a, 0], degrees=True)
base_coordinate = np.array([0.02, 0, 0])
suction_coordinates = [base_coordinate]

# |x --y
for angle in range(45, 360, 45):
    x = base_coordinate[0]*math.cos(angle*math.pi/180) - \
        base_coordinate[1]*math.sin(angle*math.pi/180)
    y = base_coordinate[0]*math.sin(angle*math.pi/180) + \
        base_coordinate[1]*math.cos(angle*math.pi/180)
    '''
    Appending all the coordiantes in suction_cooridnates and the object_suction_coordinate is the x and y 3D cooridnate of the object suction points
    '''
    suction_coordinates = np.concatenate((suction_coordinates, np.array([[x, y, 0]])), axis=0)

print(suction_coordinates)
suction_coordinates = suction_coordinates@r.as_matrix()
print(xyz_points)
from matplotlib import pyplot as plt
plt.rcParams["figure.figsize"] = [7.00, 3.50]
plt.rcParams["figure.autolayout"] = True
print(suction_coordinates)
# plt.xlim(0, 3)
# plt.ylim(0, 3)
plt.grid()
plt.plot(suction_coordinates[:, 1], suction_coordinates[:, 0], marker="o", markersize=2, markeredgecolor="red", markerfacecolor="green")
# plt.show()
''' suction points order
         (7)
    (6)       (8)

 (5)             (1)
    
    (4)       (2)
         (3)
'''
normalized_points = np.array([])
angle_x = x_a
angle_y = y_a
trasnformed_points = xyz_points[:,2]*math.cos(np.deg2rad(angle_x))*math.cos(np.deg2rad(angle_y)) - suction_coordinates[:, 2]
# trasnformed_points -= abs(suction_coordinates[4][0]-suction_coordinates[:,0])*math.sin(np.deg2rad(angle))
print(trasnformed_points)