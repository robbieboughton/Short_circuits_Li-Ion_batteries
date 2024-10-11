# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 10:28:18 2024

@author: robbi
"""


import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

"Battery model function"

def single_battery_model(t, T, I, R, h, c, v, T_env):
    Q_gen = (I ** 2) * R
    delta_T = T - T_env
    Q_loss = h * delta_T
    dT_dt = (Q_gen - Q_loss) / (c*v)
    return dT_dt

"Single cell model iteration based"

# # Defining and initialising Variables
# l = 0.07
# w = 0.02
# v = w*w*l
# c = 7 # should be 1x10^6
# h = 8 # was 13 but got values below 20 (shouldn't happen)
# I = np.random.normal(3, 0.5, 100)
# T_env = 20
# T = np.full(100,20)
# R = 1.42
# for i in range(99):
#     Q_gen = (I[i]**2)*R
#     delta_T = T[i]-T_env
#     Q_loss = h*delta_T
#     # Differential equation
#     dT_dt = (Q_gen-Q_loss)/c
#     T[i+1] = T[i]+dT_dt
#     print(dT_dt)
# print(T)

"Single cell model ODE solver"

# # Defining and initialising Variables
# l = 0.07
# w = 0.02
# v = w*w*l
# c = 1e6 # should be 1x10^6
# h = 8 # was 13 but got values below 20 (shouldn't happen)
# I = 2.9
# T_env = 20
# T = np.full(100,20)
# R = 1.42
# t_span = (0, 99)
# T0 = T_env

# sol = solve_ivp(single_battery_model, t_span, [T0], args=(I, R, h, c, v, T_env), t_eval=np.arange(100))

# # Plot the result
# plt.plot(sol.t, sol.y[0], label="Battery Temperature")
# plt.xlabel('Time (seconds)')
# plt.ylabel('Temperature (°C)')
# plt.title('Battery Temperature Over Time (Constant Current)')
# plt.legend()
# plt.grid()
# plt.show()


"Now the indexed model"

# Defining and initialising Variables
l = 0.07
w = 0.02
v = l*w*w
N = 5 # 5 cells
t = 100 # timesteps
c = 1e6 
h = 8
h_bb = 13
I = 2.7
T_env = 20
# T = np.full((t,N),20)
R = np.full(N,1.42)
t_span = (0, 499)
T0 = np.full(N, T_env)

# for i in range(t-1):
#     dT_dt = np.full(N,0)
#     for j in range(N):
#         Q_gen = ((I[i]/N)**2)*R[j] # Dividing current by 5 to assume current const. over whole pack
#         if j == 0:
#             Q_loss_eb = h*(T[i,j]-T_env)*(3*(l*w)+2*(w*w))
#             Q_loss_bb = h_bb*(T[i,j]-T[i,j+1])
#             Q_loss = Q_loss_eb+Q_loss_bb
#         elif j == N-1:
#             Q_loss_eb = h*(T[i,j]-T_env)*(3*(l*w)+2*(w*w))
#             Q_loss_bb = h_bb*(T[i,j]-T[i,j-1])
#             Q_loss = Q_loss_eb+Q_loss_bb
#         else:
#             Q_loss_bb = h_bb*(l*w)*((T[i,j]-T[i,j-1]) + (T[i,j]-T[i,j+1]))
#             Q_loss_eb = h*((2*w*w)+(2*l*w))*(T[i,j]-T_env)
#             Q_loss = Q_loss_eb+Q_loss_bb
#         # Differential equation
#         dT_dt[j] = (Q_gen-Q_loss)/c
#     T[i+1,:] = T[i,:]+dT_dt

"Indexed model using ODE solver"

def battery_pack_model(t, T, I, R, h, h_bb, c, l, w, T_env, N):
    # Reshape T into an N-element array (one temp for each battery)
    T = np.reshape(T, (N,))
    dT_dt = np.zeros(N)  # Initialize array for temperature derivatives
    
    for j in range(N):
        Q_gen = ((I/N)**2)*R[j] # Dividing current by 5 to assume current const. over whole pack
        if j == 0:
            Q_loss_eb = h*(T[j]-T_env)*(3*(l*w)+2*(w*w))
            Q_loss_bb = h_bb*(T[j]-T[j+1])*(l*w)
            Q_loss = Q_loss_eb+Q_loss_bb
        elif j == N-1:
            Q_loss_eb = h*(T[j]-T_env)*(3*(l*w)+2*(w*w))
            Q_loss_bb = h_bb*(T[j]-T[j-1])*(l*w)
            Q_loss = Q_loss_eb+Q_loss_bb
        else:
            Q_loss_bb = h_bb*(l*w)*((T[j]-T[j-1]) + (T[j]-T[j+1]))
            Q_loss_eb = h*((2*w*w)+(2*l*w))*(T[j]-T_env)
            Q_loss = Q_loss_eb+Q_loss_bb
            # Differential equation for temperature change
        dT_dt[j] = (Q_gen - Q_loss) / (c*v)

    return dT_dt

sol = solve_ivp(battery_pack_model, t_span, T0, args=(I, R, h, h_bb, c, l, w, T_env, N), t_eval=np.arange(500))
print(sol.y)

final_temperatures = sol.y[:, -1]
batteries = np.arange(1, N + 1)  # Battery indices

plt.bar(batteries, final_temperatures, color='b', alpha=0.7)
plt.xlabel('Battery Number')
plt.ylabel('Final Temperature (°C)')
plt.title('Final Temperatures of Each Battery')
plt.xticks(batteries)
plt.ylim([25,26])

plt.show()
