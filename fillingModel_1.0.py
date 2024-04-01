import sys
from math import sqrt
import matplotlib.pyplot as plt
import numpy as np
import time
import cv2
import torch
from torch import nn
import math

# Input Parameter of user
# Elektroltyt
OS = 0  # Oberflächenspannung ---mN/m
vi = 2.5 * 1e-6  # Viskosität ---Pa.s
di = 0  # Dichte---kg/m^3----g/mm^3
al = 0  # Druckkoeffizient----10^-3/bar
tha = 0  # Kontaktwinkel_Anode---grad
ths = 0  # Kontaktwinkel_Separator---grad
thk = 0  # Kontaktwinkel_Kathode---grad

# Anode
ba = 70 * 1e-3  # Breite des Zellkörpers---m
ha = 50 * 1e-3  # Höhe des Zellkörpers--- m
da = 2 * 1e-6  # Dicke----m
tw = 10  # Zeit der Benetzung--- s

# Befüllungsmethode
P = 0.1    #Druck ---bar
Pa = 0.08  # Anfangsdruck----bar

# Parameter of system/Simulation
l_incr = 0.01  # in mm/lu
t_incr = 0.01  # in s/tu
k_equal = 0  # mm/s^0.5
k_trans_max = 0.01  # in %
threshold = 0.1  # in %
time_incr_max = 2 * k_equal * l_incr  # in s/tu
init_Boundary_Condition = 2  # 1: Wetting only from bottom; 2: Wetting from each side
wetting_sum_threshold = 1


# Theoretische Bewegungsgeschwindigkeit
def k_operation(r, th):  # K-Wert
    vip = vi * math.exp(al * (P - Pa))  # Viskositätsänderungen durch Druck
    k = (OS * r * math.cos(th / 180 * math.pi) / (2 * vip)) ** 0.5  # mm/s^0.5
    return k

k_equal = k_operation(0, tha)
# parameter for output
time_incr_eff = 0

# Grid setup
length_mesh_steps = int(ba / l_incr) + 1
width_mesh_steps = int(ha / l_incr) + 1
time_steps = int(tw / t_incr)

wetting_last = torch.zeros([length_mesh_steps, width_mesh_steps], dtype=torch.float)
wetting_new = torch.zeros([length_mesh_steps, width_mesh_steps], dtype=torch.float)
wetting_sum = torch.zeros(time_steps, dtype=torch.float)
wetting_Foto = torch.zeros([length_mesh_steps, width_mesh_steps], dtype=torch.float)
# Boundary conditions
wetting_boundary = torch.zeros([length_mesh_steps, width_mesh_steps], dtype=torch.float)

# Main simulation loop
# 1. Set the boundary condition
if init_Boundary_Condition == 1:
    wetting_boundary[-1, :] = 1
elif init_Boundary_Condition == 2:
    wetting_boundary[-1, :] = 1
    wetting_boundary[0, :] = 1
    wetting_boundary[:, -1] = 1
    wetting_boundary[:, 0] = 1

# 2. set for the first time step
wetting_last[:, :] = wetting_boundary[:, :]
wetting_last = wetting_last.unsqueeze(0).unsqueeze(0)
# other Parameter
wetting_1 = torch.ones([length_mesh_steps, width_mesh_steps], dtype=torch.float)
wetting_0 = torch.zeros([length_mesh_steps, width_mesh_steps], dtype=torch.float)
###########################
# Time Test
##########################
# 3. run the main loop
for i in range(1, time_steps):
    ###########################
    # Time Test
    T1 = time.time()
    ##########################
    # 3.1 calculate the time-dependen parameter
    Act_time_Step = t_incr * i  # in ms
    k_trans_new = 0.5 * k_equal * (t_incr / l_incr) * (1 / sqrt(t_incr * i))
    abs_a = (1 / (1 + sqrt(2))) * k_trans_new
    abs_b = (1 / (sqrt(2) * (1 + sqrt(2)))) * k_trans_new
    # 3.1 calculate the time-dependen kernel
    kernel_2d = torch.tensor([[abs_b, abs_a, abs_b],
                              [abs_a, 0, abs_a],
                              [abs_b, abs_a, abs_b]])
    kernel = kernel_2d.unsqueeze(0).unsqueeze(0)
    conv = nn.Conv2d(1, 1, 3, padding=1, bias=False)
    # update the weight in each loop
    conv.weight = nn.Parameter(kernel)
    # 3.2 calculate the new value of the array
    wetting_LessThanOne = torch.where(wetting_last < 1, wetting_last, 0.0)
    wetting_MoreThanOne = torch.where(wetting_last >= 1, 1.0, 0.0)
    wetting_add = conv(wetting_MoreThanOne) - wetting_MoreThanOne * 8.0
    wetting_new = wetting_add.where(wetting_add > 0, 0.0) + wetting_LessThanOne + wetting_MoreThanOne
    wetting_Foto = torch.where(wetting_new < 1, wetting_new, 1.0)
    # 3.3 transfer the new array to the last array
    wetting_last = wetting_Foto
    print(wetting_last)
    # 3.4 calculate the sum of the Matrix
    wetting_sum[i] = torch.sum(wetting_last)
    # 3.5 Displaying the wetting process
    if i % 1 == 0:
        # use OpenCV
        wetting_show = wetting_Foto.squeeze()
        wetting_show_cv = wetting_show.detach().numpy()
        cv2.namedWindow('demo', cv2.WINDOW_NORMAL)
        cv2.imshow("demo", wetting_show_cv)
        cv2.waitKey(50)
    # print information of loop
    if i % 1 == 0:
        print("****************************************")
        T2 = time.time()
        print("Time for ", i, "st loop needs", '%.2f' % (T2 - T1), "seconds")
        rate = wetting_sum[i] / wetting_new.shape[2] / wetting_new.shape[3] * 100
        print("Wetting Percent", '%.3f' % rate, "%")
        print("****************************************")
    # 3.6 wetting process end
    if wetting_sum[i] - wetting_sum[i - 1] < wetting_sum_threshold | i == time_steps - 1:
        time_incr_eff = i
        break

# Main loop end
#############################################################################

# 4. Analysis and plotting
# x = np.arange(wetting_time_steps) * time_incr
# rSumWetting = wetting_sum
# rSumWettingRate = rSumWetting / wetting_last.shape[0] / wetting_last.shape[1]  # in %
# rTheorieSumWetting = k_equal * np.sqrt(np.arange(wetting_time_steps) * time_incr) / length  # in %
# rError = rTheorieSumWetting - rSumWettingRate
# Grad_B = np.gradient(rSumWettingRate, np.sqrt(x))
# Grad_C = np.gradient(rTheorieSumWetting, np.sqrt(x))
#
# x_plot = x[1:time_incr_eff]
# A_plot = rSumWetting[1:time_incr_eff]
# B_plot = rSumWettingRate[1:time_incr_eff]
# C_plot = rTheorieSumWetting[1:time_incr_eff]
# E_plot = rError[1:time_incr_eff]
# Grad_B_plot = Grad_B[1:time_incr_eff]
# Grad_C_plot = Grad_C[1:time_incr_eff]
#
# plt.figure('Benetzungskurve')
# plt.title('Benetzungskurve')
# plt.plot(x, rSumWettingRate, x, rTheorieSumWetting, x, rError)
# plt.legend(['Simulation', 'Theorie','Error'])
# plt.show()
#
# plt.figure('Gradient Benetzungskurve')
# plt.title('Gradient Benetzungskurve')
# plt.plot(np.sqrt(x), rSumWettingRate, np.sqrt(x), rTheorieSumWetting, np.sqrt(x), Grad_B, np.sqrt(x), Grad_C)
# plt.legend(['Simulation', 'Theorie','Error'])
# plt.show()
sys.exit()
