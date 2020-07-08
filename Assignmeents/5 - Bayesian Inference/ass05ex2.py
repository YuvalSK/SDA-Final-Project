# -*- coding: utf-8 -*-
"""
Created on Wed May 13 13:38:58 2020
@author: samoilov Yuval
"""
import numpy as np
import matplotlib.pyplot as plt
import math

plot_duration = 360
teta = math.pi*2
samp = teta/plot_duration
rates = [0,2*math.pi/3,4*math.pi/3]
rmax = 55
rmin = 0
teta_array = np.arange(0,teta,samp)

# change from polar to cartesian
r = 1
fig, ax = plt.subplots(figsize=(16,4), nrows=2, ncols=1)
#ex1
for j,rate in enumerate(rates):    
    spk_array = np.zeros_like(teta_array) 
    for i,t in enumerate(teta_array):
        avg_rate = rmax*np.cos(t - rate)
        spk_array[i] = max(avg_rate,0) 
    ax[0].plot(teta_array, spk_array,label=f'{j+1}')
    
    if j==2:
        #ex2
        x = np.abs(r*np.cos(rates))
        y = np.abs(r*np.sin(rates))
        dir_vec = np.array([x, y])
        pop_vec = np.arange(0,teta,samp) 
        vec = (spk_array-rmin) * teta_array/ rmax
        vec_t = np.vstack((vec, vec)).T
        pop_vec = vec_t.dot(dir_vec) 
        ax[1].plot(teta_array, pop_vec ,label=f'{j+1}')
        opt_teta = np.tanh(y/x)
        print(f'{opt_teta[1]:.2f} is the optimal theta')
    
ax[1].plot(teta_array,teta_array,label='optimal',linestyle='dashed')
ax[1].set_xlabel('Radians (pai)')
ax[1].set_ylabel('Spike Rates')
ax[1].set_title('population vector')
ax[0].legend(loc='upper right')
ax[0].set_xlabel('Radians (pai)')
ax[0].set_ylabel('Spike Rates')
ax[0].set_title('Inter Neurons')

plt.tight_layout()
plt.savefig('ass05ex02')