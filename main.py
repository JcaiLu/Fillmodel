from math import sqrt

import matplotlib.pyplot as plt
import numpy as np

length = 1  # in mm; Höhendimension x |
width = 0.5  # in mm; Breitendimension y -
thickness = 2  # in mm
wetting_time = 200  # in s
mesh_incr = 0.02  # in mm/lu
k_equal = 0.05  # mm/s^0.5
time_incr = 0.02  # in s/tu
k_trans_max = 0.01  # in %
threshold = 1  # in %
time_incr_max = 2 * k_equal * mesh_incr  # in s/tu

# Grid setup
length_mesh_steps = int(length / mesh_incr) + 1
width_mesh_steps = int(width / mesh_incr) + 1
wetting_time_steps = int(wetting_time / time_incr)

# Initialize Wetting array

Wetting_last = np.zeros((length_mesh_steps, width_mesh_steps))
Wetting_new = np.zeros((length_mesh_steps, width_mesh_steps))
Wetting_sum = np.zeros(wetting_time_steps)

# Boundary conditions
Wetting_boundary = np.zeros((length_mesh_steps, width_mesh_steps))

neighbors = [
            (-1, -1), (0, -1), (1, -1),
            (-1, 0), (1, 0),
            (-1, 1), (0, 1), (1, 1)
    ]

def ausbreitung(array, sz1, sz2, abs_a, abs_b, threshold):
    _array = array[sz1, sz2]
    for d1, d2 in neighbors:
        new_sz1 = sz1 + d1
        new_sz2 = sz2 + d2
        if 0 <= new_sz1 < array.shape[0] and 0 <= new_sz2 < array.shape[1]:
            if array[new_sz1, new_sz2] >= threshold:
                abs_coeff = abs_a if d1 == 0 or d2 == 0 else abs_b
                _array += array[new_sz1, new_sz2] * abs_coeff
    return _array

# Main simulation loop
# 1. Set the boundary condition
Wetting_boundary[-1, :] = 1

# 2. set for the first time step
Wetting_last[:,:] = Wetting_boundary[:,:]

# 3. run the main loop
for i in range(1, wetting_time_steps):
    # 3.1 calculate the time-dependen parameter
    Act_time_Step = time_incr * i  # in ms
    k_trans_new = 0.5 * k_equal * (time_incr / mesh_incr) * (1 / sqrt(time_incr * i))
    abs_a = (1 / (1 + sqrt(2))) * k_trans_new
    abs_b = (1 / (sqrt(2) * (1 + sqrt(2)))) * k_trans_new

    # 3.2 calculate the new value of the array
    for j in range(0,length_mesh_steps ):
        for k in range(0,width_mesh_steps ):
            if Wetting_last[j, k] < 1:
                Wetting_new[j, k] = ausbreitung(Wetting_last, j, k, abs_a, abs_b, threshold)
                if Wetting_new[j, k] > 1:
                    Wetting_new[j, k] = 1
            else:
                Wetting_new[j, k] = Wetting_last[j, k]
            # print(j, k, Wetting_new[j,k]) ## for Checking the value

    # 3.3 transfer the new array to the last array
    Wetting_last = Wetting_new

    # 3.4 calculate the sum of the Matrix
    Wetting_sum[i] = np.sum(Wetting_new)

    # 3.5 Displaying the wetting process
    if i % 100 == 0:
        plt.cla()
        plt.imshow(Wetting_last[:, :], cmap='gray')
        plt.show()
        plt.pause(0.1)
    if i % 10 == 0:
        print("Wetting Prozess .......", i / wetting_time_steps * 100, "%")

# 4. Analysis and plotting
x = np.arange(wetting_time_steps) * time_incr
A = Wetting_sum
B = A / Wetting_last.shape[0] / Wetting_last.shape[1]  # in %
C = k_equal * np.sqrt(np.arange(wetting_time_steps) * time_incr) / length  # in %
E = C - B
Grad_B = np.gradient(B, np.sqrt(x))
Grad_C = np.gradient(C, np.sqrt(x))

plt.figure()
plt.plot(x, B, x, C, x, E)
plt.figure()
plt.plot(np.sqrt(x), B, np.sqrt(x), C, np.sqrt(x), Grad_B, np.sqrt(x), Grad_C)
plt.show()
