# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 10:28:18 2024

@author: robbi
"""

"Single cell model"
import numpy as np
# Defining and initialising Variables
# c = 7
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

"Now the indexed model"
# Defining and initialising Variables
l = 0.07
w = 0.02
N = 5 # 5 cells
t = 100 # timesteps
c = 7
h = 8
h_bb = 13
I = np.random.normal(3, 0.5, t)
T_env = 20
T = np.full((t,N),20)
R = np.full(N,1.42)
dT_dt = np.full(N,0)
for i in range(N-1):
    for j in range(4):
        Q_gen = ((I[i]/5)**2)*R # Dividing each current by 5 to assume current const. over whole pack
        if j = 0:
            Q_loss_ab = h*(T[i,j]-T_env)*(3*(l*w)+2*(w*w))
            Q_loss_bb = h_bb*(T[i,j]-T(i,j+1))
            Q_loss = h*delta_T
        elif j = max(N-1):
            Q_loss_ab = h*(T[i,j]-T_env)*(3*(l*w)+2*(w*w))
            Q_loss_bb = h_bb*(T[i,j]-T(i,j-1))
            Q_loss = Q_loss_ab+Q_loss_bb
        else:
            Q_loss_bb = h_bb*(l*w)*(T[i,j]-T[i,j-1]) + (T[i,j]-T[i,j+1])
            Q_loss_ab = h*(w*w)*(T[i,j]-T_env)*2
            Q_loss = Q_loss_ab+Q_loss_bb
        # Differential equation
        dT_dt[j] = dT_dt[j]+(Q_gen-Q_loss)/c
        T[i+1] = T[i]+dT_dt[j]
        print(dT_dt)
print(T)



