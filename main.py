from math import sqrt
import matplotlib.pyplot as plt
import numpy as np
import time
from numba import jit
import cv2


# Input Parameter of user
length = 70  # in mm; Höhendimension x |
width = 50  # in mm; Breitendimension y -
thickness = 2  # in mm
wetting_time = 20  # in s

# Parameter of system
mesh_incr = 0.2  # in mm/lu
k_equal = 0.5  # mm/s^0.5
time_incr = 0.2  # in s/tu
k_trans_max = 0.01  # in %
threshold = 1  # in %
time_incr_max = 2 * k_equal * mesh_incr  # in s/tu
init_Boundary_Condition = 1  # 1: Wetting only from bottom; 2: Wetting from each side
Wetting_sum_threshold = 1

# parameter for output
time_incr_eff = 0

# 内置函数math.ceil() 来向上取整
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

# use numba to celebrate the program
@jit
def ausbreitung(array, sz1, sz2, abs_a, abs_b, threshold):
    _array = array[sz1, sz2]
    neighbors = [
        (-1, -1), (0, -1), (1, -1),
        (-1, 0), (1, 0),
        (-1, 1), (0, 1), (1, 1)
    ]
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
if init_Boundary_Condition == 1:
    Wetting_boundary[-1, :] = 1
elif init_Boundary_Condition == 2:
    Wetting_boundary[-1, :] = 1
    Wetting_boundary[0, :] = 1
    Wetting_boundary[:, -1] = 1
    Wetting_boundary[:, 0] = 1

# 2. set for the first time step
Wetting_last[:, :] = Wetting_boundary[:, :]

###########################
# Time Test
##########################
# 3. run the main loop
for i in range(1, wetting_time_steps):
    ###########################
    # Time Test
    T1 = time.time()
    ##########################
    # 3.1 calculate the time-dependen parameter
    Act_time_Step = time_incr * i  # in ms
    k_trans_new = 0.5 * k_equal * (time_incr / mesh_incr) * (1 / sqrt(time_incr * i))
    abs_a = (1 / (1 + sqrt(2))) * k_trans_new
    abs_b = (1 / (sqrt(2) * (1 + sqrt(2)))) * k_trans_new

    # 3.2 calculate the new value of the array
    for j in range(0, length_mesh_steps):
        for k in range(0, width_mesh_steps):
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
    if Wetting_sum[i] - Wetting_sum[i - 1] < Wetting_sum_threshold:
        time_incr_eff = i
        break
    # 3.5 Displaying the wetting process
    if i % 50 == 0:
        # use OpenCV
        cv2.imshow("demo",Wetting_last)
        cv2.resizeWindow("demo",[width*50, length*50])
        cv2.waitKey(50)

    # print information of loop
    if i % 10 == 0:
        print("****************************************")
        T2 = time.time()
        print("Time for ", i, "st loop needs", '%.2f' % (T2 - T1), "seconds")

        rate = Wetting_sum[i] / Wetting_new.shape[0] / Wetting_new.shape[1] * 100
        print("Wetting Percent", '%.3f' % rate, "%")

        print("****************************************")


# Main loop end
#############################################################################

# 4. Analysis and plotting
x = np.arange(wetting_time_steps) * time_incr
rSumWetting = Wetting_sum
rSumWettingRate = rSumWetting / Wetting_last.shape[0] / Wetting_last.shape[1]  # in %
rTheorieSumWetting = k_equal * np.sqrt(np.arange(wetting_time_steps) * time_incr) / length  # in %
rError = rTheorieSumWetting - rSumWettingRate
Grad_B = np.gradient(rSumWettingRate, np.sqrt(x))
Grad_C = np.gradient(rTheorieSumWetting, np.sqrt(x))

x_plot = x[1:time_incr_eff]
A_plot = rSumWetting[1:time_incr_eff]
B_plot = rSumWettingRate[1:time_incr_eff]
C_plot = rTheorieSumWetting[1:time_incr_eff]
E_plot = rError[1:time_incr_eff]
Grad_B_plot = Grad_B[1:time_incr_eff]
Grad_C_plot = Grad_C[1:time_incr_eff]

plt.figure('Benetzungskurve')
plt.title('Benetzungskurve')
plt.plot(x, rSumWettingRate, x, rTheorieSumWetting, x, rError)
plt.legend(['Simulation', 'Theorie','Error'])
plt.show()

plt.figure('Gradient Benetzungskurve')
plt.title('Gradient Benetzungskurve')
plt.plot(np.sqrt(x), rSumWettingRate, np.sqrt(x), rTheorieSumWetting, np.sqrt(x), Grad_B, np.sqrt(x), Grad_C)
plt.legend(['Simulation', 'Theorie','Error'])
plt.show()
