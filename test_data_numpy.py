import numpy as np
import matplotlib.pyplot as plt

# depth_path = "/home/soofiyan_ws/Documents/Issac_gym_ws/System_Identification_Data/Parallelization-Data-evlaution/0/depth_image_0_63.npy"

depth_image = np.load("/home/soofiyan_ws/Documents/Issac_gym_ws/System_Identification_Data/Parallelization-Data-evlaution/0/depth_image_0_579.npy")
segmask_image = np.load("/home/soofiyan_ws/Documents/Issac_gym_ws/System_Identification_Data/Parallelization-Data-evlaution/0/segmask_0_579.npy")

plt.imshow(depth_image)
plt.show()
plt.imshow(segmask_image)
plt.show()