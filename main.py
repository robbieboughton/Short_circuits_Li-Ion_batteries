# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 10:28:18 2024

@author: robbi
"""


import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import math as m

"Battery model function"

def single_battery_model(t, T, I, R, h, c, v, T_env):
    Q_gen = (V ** 2)/R
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
# l = 0.065
# w = 0.018
# v = w*w*l
# c = 1899335
# h = 8 # was 13 but got values below 20 (shouldn't happen)
# I = 6.7
# V = 0.3
# T_env = 20
# R = 0.05
# t_span = (0, 299)
# T0 = T_env

# sol = solve_ivp(single_battery_model, t_span, [T0], args=(I, R, h, c, v, T_env), t_eval=np.arange(0,300),max_step = 0.1)

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
l = 0.065
w = 0.018
v = l*w*w
N = 15
t = 3600
c = 1899335 # Estimated s.h.c of 40 J/K 
h = 8
h_bb = 10
voltages = np.arange(0.1,1.1,0.1)
# V = 0.1
T_env = 20
t_span = (0, t)
T0 = np.full(N, T_env)
n_sc = m.floor(N/2)
t_dg = 900
data = {}

"Time-dependent resistance function"

def time_dependent_resistance(n_sc,t,t_dg):
    # print(t)
    R = np.full(N,0.05)
    R[n_sc] *= np.exp(-t/t_dg)

    return R

"Indexed model using ODE solver"

def battery_pack_model(t, T, h, h_bb, c, l, w, T_env, N, V):
    # Reshape T into an N-element array (one temp for each battery)
    T = np.reshape(T, (N,))
    # print(t)

    R = time_dependent_resistance(n_sc,t,t_dg)
    dT_dt = np.zeros(N)  # Initialize array for temperature derivatives
    # temperature_data.append(T.copy())
    # resistance_data.append(R.copy())

    
    # print(R[n_sc])
    for j in range(N):
        Q_gen = (V**2)/R[j]
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


# sol = solve_ivp(battery_pack_model, t_span, T0, args=(h, h_bb, c, l, w, T_env, N), atol = 1e-8, rtol = 1e-7)
# print(sol.t[-1])
# print(sol.success, sol.message)

# # plt.plot(sol.t, sol.y[2,:])

# final_temperatures = sol.y[:, -1]
# batteries = np.arange(1, N + 1)  # Battery indices

# plt.bar(batteries, final_temperatures, color='b', alpha=0.7)
# plt.xlabel('Battery Number')
# plt.ylabel('Final Temperature (°C)')
# plt.title('Final Temperatures of Each Battery')
# plt.xticks(batteries)
# plt.ylim([25,30])

# plt.show()
"Saving data for different voltages"

for V in voltages:
    temperature_data = []
    resistance_data = []
    # Solve the model for the given voltage
    sol = solve_ivp(
        battery_pack_model, t_span, T0, args=(h, h_bb, c, l, w, T_env, N, V), atol=1e-8, rtol=1e-7
    )
    temperature_data.append(sol.y.T)  # Store temperature data (timesteps × batteries)
    resistance_data.append(time_dependent_resistance(n_sc, sol.t, t_dg))  # Store resistance data

    # Store the results for this voltage
    data[V] = {
        "temperature_data": np.array(temperature_data),  # Shape: (timesteps, batteries)
        "resistance_data": np.array(resistance_data),    # Optional: Include resistance
    }


"Formatting saved data"

np.savez("battery_data_1:15.npz", **data)


