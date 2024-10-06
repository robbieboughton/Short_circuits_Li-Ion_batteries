# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 10:28:18 2024

@author: robbi
"""
import numpy as np
# Defining and initialising Variables
c = 400
h = 13
I = np.random.normal(3, 0.5, 100)
T_env = 20
T = np.full(100,20)
for i in range(100)
    Q_gen = (I(i)^2)*R
    delta_T = T(i)-T_env
    Q_loss = h*delta_T
    # Differential equation
    dT_dt = (Q_gen-Q_loss)/c
    T(i+1) = T(i)+dT_dt

