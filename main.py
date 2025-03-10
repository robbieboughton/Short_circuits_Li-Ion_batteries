# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 10:28:18 2024

@author: robbi
"""


import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import math as m
import pandas as pd
import os

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

"Time-dependent resistance function"

def time_dependent_resistance(n_sc,t,t_dg):

    R = np.full(N,0.05)
    R[n_sc] *= np.exp(-t/t_dg)

    return R

"Indexed model using ODE solver"

def battery_pack_model(t, T, h, h_bb, c, l, w, T_env, N, s_c):
    T = np.reshape(T, (N,))

    if s_c == 1:
        R = time_dependent_resistance(n_sc,t,t_dg)
    else:
        R = np.full(N, 0.05)
    dT_dt = np.zeros(N)  # Initialize array for temperature derivatives

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

"Model for testing"

# # Defining and initialising Variables
# l = 0.065
# w = 0.018
# v = l*w*w
# N = 15
# t = 3600
# c = 1899335 # Estimated s.h.c of 40 J/K 
# h = 8
# h_bb = 10
# V = 0.1
# T_env = 20
# t_span = (0, t)
# T0 = np.full(N, T_env)
# n_sc = m.floor(N/2)
# s_c = 1
# t_dg = 900

# sol = solve_ivp(battery_pack_model, t_span, T0, args=(h, h_bb, c, l, w, T_env, N), atol = 1e-8, rtol = 1e-7)
# print(sol.t[-1])
# print(sol.success, sol.message)

# final_temperatures = sol.y[:, -1]
# batteries = np.arange(1, N + 1)  # Battery indices

# plt.bar(batteries, final_temperatures, color='b', alpha=0.7)
# plt.xlabel('Battery Number')
# plt.ylabel('Final Temperature (°C)')
# plt.title('Final Temperatures of Each Battery')
# plt.xticks(batteries)


# plt.show()

"Data saving"

# Defining and initialising Variables
l = 0.065
w = 0.018
v = l*w*w
N = 15
t = 3600
c = 1899335 # Estimated s.h.c of 40 J/K 
h = 8
h_bb = 10
voltages = np.arange(0.05,1.05,0.1)
T_env = 20
t_span = (0, t)
T0 = np.full(N, T_env)
n_sc = m.floor(N/2)
s_c = 0
t_dg_values = [500, 1500, 2500]
dataset = []

num_timesteps = 150
t_eval = np.linspace(t_span[0], t_span[1], num_timesteps)

"Saving data"

if not os.path.exists("short_circuit3"):
    os.makedirs("short_circuit3")
if not os.path.exists("no_short_circuit3"):
    os.makedirs("no_short_circuit3")
    
    
for t_dg in t_dg_values: 
    for V in voltages:
        temperature_data = []
        resistance_data = []
        temperature_df = pd.DataFrame(columns=['timestep'] + [f'battery_{i+1}' for i in range(N)])
        resistance_df = pd.DataFrame(columns=['timestep'] + [f'battery_{i+1}' for i in range(N)])
        
        sol = solve_ivp(
            battery_pack_model, t_span, T0, args=(h, h_bb, c, l, w, T_env, N, s_c), t_eval = t_eval, atol=1e-8, rtol=1e-7,)
        temperature_data.append(sol.y.T)  # Store temperature data (timesteps × batteries)
        resistance_at_timesteps = []
        for t in sol.t:
            if s_c == 1:
                resistance_at_timesteps.append(time_dependent_resistance(n_sc, t, t_dg))
            else:
                resistance_at_timesteps.append(np.full(N, 0.05))
        resistance_data.append(resistance_at_timesteps)
        
    
        for timestep_idx, timestep in enumerate(sol.t):
            # for battery_idx in range(N):
            #     dataset.append({
            #         "voltage": V,
            #         "timestep": timestep,
            #         "battery_index": battery_idx,
            #         "temperature": sol.y.T[timestep_idx, battery_idx],
            #         "resistance": resistance_at_timesteps[timestep_idx][battery_idx],
            #         "short_circuit": s_c,
            #         "dendrite_growth_time": t_dg
            #     })
            temperature_df.loc[timestep_idx] = [timestep] + list(sol.y.T[timestep_idx])
            resistance_df.loc[timestep_idx] = [timestep] + list(resistance_at_timesteps[timestep_idx])
        avg_temperature_df = pd.DataFrame({
            'timestep': sol.t,
            'average_temperature': temperature_df.iloc[:, 1:].mean(axis=1)
        })
        # # Convert the dataset list to a Pandas DataFrame
        # df_temp = pd.DataFrame(dataset)

        # Save as a CSV file
        if s_c == 1:
            folder = "short_circuit3"
            # file_path = os.path.join(folder_name,f"short_circuit")
            # df.to_csv(f"voltage{V}_t_dg{t_dg}_temp.csv", index=False)
            temperature_df.to_csv(f"{folder}/temperature_data_V_{V}_tdg_{t_dg}.csv", index=False)
            resistance_df.to_csv(f"{folder}/resistance_data_V_{V}_tdg_{t_dg}.csv", index=False)
            avg_temperature_df.to_csv(f"{folder}/avg_temperature_data_V_{V}_tdg_{t_dg}.csv", index=False)
        elif s_c == 0:
            folder = "no_short_circuit3"
            
            temperature_df.to_csv(f"{folder}/temperature_data_V_{V}_tdg_{t_dg}.csv", index=False)
            resistance_df.to_csv(f"{folder}/resistance_data_V_{V}_tdg_{t_dg}.csv", index=False) 
            avg_temperature_df.to_csv(f"{folder}/avg_temperature_data_V_{V}_tdg_{t_dg}.csv", index=False)
            # df.to_csv("battery_data_no_isc.csv", index=False)


