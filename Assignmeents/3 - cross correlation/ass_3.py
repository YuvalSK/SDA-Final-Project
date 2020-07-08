# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 17:01:45 2020
SDA third HW section 3, Due to 30/03/2020
@author: Samoilov-Katz Yuval id 204025258
"""
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

samp = 1000
duration = 120
#sec 2 subsec 1
mat_data = sio.loadmat('Q3data.mat')
spk_time = np.array(mat_data['spikeTime']*samp)
stm_time = np.array(mat_data['stimTime']*samp)

stm_array = np.zeros((samp*duration, 1))
spk_array = np.zeros((samp*duration, 1))

for s in spk_time[0]:
    spk_array[int(s)] = 1
    
for t in stm_time[0]:
    stm_array[int(t)] = 1
    
time_array = np.arange(0, duration, 1/samp)

fig, ax = plt.subplots(figsize=(14,6), nrows=3, ncols=1)

ax[0].plot(time_array,spk_array)
avg_spk = np.mean(spk_array) * samp
avg_spk1 = np.mean(spk_array[:10501]) * samp
avg_spk2 = np.mean(spk_array[10501:109501]) * samp
avg_spk3 = np.mean(spk_array[109501:]) * samp

ax[0].set_ylabel('Sp')
ax[0].set_title(f'Neuron r(t) = {avg_spk:.3f} sp/sec\nbefore:{avg_spk1:.3f},during:{avg_spk2:.3f},after:{avg_spk3:.3f}')
ax[0].set_xlabel('sec')

bin_size = 100
bin_array = np.zeros_like(spk_array, dtype=np.float32)
for i in np.arange(0, samp*duration, bin_size):
    bin_array[i:i+bin_size] = spk_array[i:i+bin_size].sum() / bin_size

ax[1].plot(time_array,bin_array)
ax[1].set_ylabel('Sp/sec')
ax[1].set_title('Binned Spike train')
ax[1].set_xlabel('bin')

ax[2].plot(time_array,stm_array) 
avg_stm = np.mean(stm_array)* samp
ax[2].set_ylabel('Sp')
ax[2].set_title(f'Stimulation r(t) = {avg_stm:.3f} sp/sec')
ax[2].set_xlabel('sec')

plt.tight_layout()
plt.savefig('ass3ex3')


