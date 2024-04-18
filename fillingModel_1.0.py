import os
import sys
from math import sqrt
import matplotlib.pyplot as plt
import numpy as np
import time
import cv2
import torch
from torch import nn
import math
import datetime
import re

# Input Parameter of user
# Elektroltyt
OS = 0.034 * 1000  # Oberflächenspannung ---mN/m
di = 1.3  # Dichte---kg/m^3
Vi = 2.5 * di  # Viskosität ---mPa.s
al = 2.0 * 0.001  # Druckkoeffizient----10^-3/bar
tha = 0.8182232853354976 * 180 / np.pi  # Kontaktwinkel_Anode---grad
ths = 0.96 * 180 / np.pi  # Kontaktwinkel_Separator---grad
thk = 1.0884720504995502 * 180 / np.pi  # Kontaktwinkel_Kathode---grad
qElektrolyte = 0.0 # Volumen der Elektroltyt---ml
# Anode
ba = 70 * 1e-3  # Breite des Zellkörpers---mm
ha = 50 * 1e-3   # Höhe des Zellkörpers--- mm
da = 2    # Dicke----mm
tw = 1  # Zeit der Benetzung--- s
ra = 1.57  #Effektiver Porenradius ----mm


# Befüllungsmethode
P = 0.1  # Druck ---bar
Pa = 0.08  # Anfangsdruck----bar

# Parameter of system/Simulation
l_incr = 0.0001  # in mm/lu
t_incr = 0.001  # in s/tu
k_equal = 0.5  # mm/s^0.5
k_trans_max = 0.01  # in %
threshold = 0.1  # in %
time_incr_max = 2 * k_equal * l_incr  # in s/tu
init_Boundary_Condition = 1  # 1: Wetting only from bottom; 2: Wetting from each side
wetting_sum_threshold = 10

# Parameter for video
theTime = str(datetime.datetime.now().strftime('%m-%d %H:%M,f'))
videoname = re.sub(u"([^\u0030-\u0039])", "", theTime)
video_dir = 'G:/video/' + videoname + '.avi'

if not os.path.exists(video_dir):
    print('check the output path of videos')
fps = 30
num = 0
img_size = (500,700)
fourcc = cv2.VideoWriter_fourcc(*'mp4v') #opencv3.0
videoWriter = cv2.VideoWriter(video_dir, fourcc, fps, img_size)

# Theoretische Bewegungsgeschwindigkeit
def k_operation(r, th):  # K-Wert g/s^0.5
    Vip = Vi * math.exp(al * (P - Pa))  # Viskositätsänderungen durch Druck
    k = (OS * r * math.cos(th / 180 * math.pi) / (2 * Vip)) ** 0.5  # mm/s^0.5
    return k

def ResizeWithAspectRatio(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(image, dim, interpolation=inter)

k_equal = k_operation(ra, tha)
# parameter for output
time_incr_eff = 0.0
rate = 0.0

# Grid setup
length_mesh_steps = int(ba / l_incr) + 1
width_mesh_steps = int(ha / l_incr) + 1
time_steps = int(tw / t_incr)
time_steps_real = 0

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
        cv2.resizeWindow("demo", 500, 500)
        cv2.waitKey(50)
        videoWriter.write(wetting_show_cv)
    # print information of loop
    if i % 1 == 0:
        print("****************************************")
        T2 = time.time()
        print("Time for ", i, "st loop needs", '%.2f' % (T2 - T1), "seconds")
        rate = wetting_sum[i] / wetting_new.shape[2] / wetting_new.shape[3] * 100
        print("Wetting Percent", '%.3f' % rate, "%")
        print("****************************************")
    # 3.6 wetting process end
    time_steps_real = i
    if rate >= 100:
        break

# Main loop end
#############################################################################

# 4. Analysis and plotting# 4. Analysis and plotting
x = np.arange(time_steps_real) * t_incr
rSumWetting = wetting_sum
rSumWettingRate = rSumWetting / wetting_last.shape[2] / wetting_last.shape[3]  # in %
rTheorieSumWetting = k_equal * np.sqrt(np.arange(time_steps) * t_incr) / ha  # in %
print(type(rSumWettingRate))
rSumWettingRate_numpy = rSumWettingRate.detach().numpy()
rError = rTheorieSumWetting - rSumWettingRate_numpy

# if time_steps > time_steps_real:
t = time_steps_real
# elif time_steps > rSumWettingRate_numpy.length():
#     t = rSumWettingRate_numpy.length()

Grad_B = np.gradient(rSumWettingRate_numpy[0:t], np.sqrt(x))
Grad_C = np.gradient(rTheorieSumWetting[0:t], np.sqrt(x))

x_plot = x[0:t]
A_plot = rSumWetting[0:t]
B_plot = rSumWettingRate[0:t]
C_plot = rTheorieSumWetting[0:t]
E_plot = abs(rError[0:t])

plt.figure('Benetzungskurve')
plt.title('Benetzungskurve')
plt.plot(x_plot, rSumWettingRate_numpy[0:t], x_plot, rTheorieSumWetting[0:t], x_plot, rError[0:t])
plt.legend(['Simulation', 'Theorie', 'Error'])
plt.show()

plt.figure('Gradient Benetzungskurve')
plt.title('Gradient Benetzungskurve')
plt.plot(np.sqrt(x), rSumWettingRate_numpy[0:t], np.sqrt(x), rTheorieSumWetting[0:t], np.sqrt(x), Grad_B[0:t],np.sqrt(x), Grad_C[0:t])
plt.legend(['Simulation', 'Theorie', 'Grad_Simulation', 'Grad_Theorie'])
plt.show()
videoWriter.release()
sys.exit()
