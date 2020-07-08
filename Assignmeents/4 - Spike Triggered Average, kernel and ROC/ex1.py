# -*- coding: utf-8 -*-
"""
Created on Monday May 3 
SDA fourth HW section, Due to 7/05/2020
@author: Samoilov-Katz Yuval id 204025258
"""
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import pycorrelate

samp = 1000
duration = 200

mat_data = sio.loadmat('STA.mat')
spk_train = mat_data['spikeTrain'][0]
stm_sig = mat_data['stimSig'][0]

time_array = np.arange(0, duration, 1/samp)
fig, ax = plt.subplots(figsize=(14,6), nrows=3, ncols=1)

#normalization of stimulus so that N(t) = s(t)- avg(s(t))
avg_sig = np.mean(stm_sig)
Nstm_sig = stm_sig - avg_sig
ax[0].plot(time_array,Nstm_sig)
ax[0].set_ylabel('Normalized (Au)')
ax[0].set_title(f'Stimulus Au(t) = {avg_sig* samp:.3f} Au/sec')
ax[0].set_xlabel('sec')

ax[1].plot(time_array,spk_train)
avg_spk = np.mean(spk_train)
ax[1].set_ylabel('sp')
ax[1].set_title(f'Neuron r(t) = {avg_spk*samp:.3f} sp/sec')
ax[1].set_xlabel('sec')

cbins = np.arange(-500,500,1)
# STA by the reverse correlation of stimulus and spk train <C(t)> = 1/avg(t) * cross_corr()
C_t = np.flip(pycorrelate.pcorrelate(stm_sig, spk_train, bins=cbins)) / avg_spk
ax[2].plot(cbins[:-1], C_t / duration * samp)
delta_t = len(C_t.nonzero()[-1])-1
Hz = samp / delta_t
ax[2].set_title(f'STA (stimulus--->Spike)\ndelta t={delta_t:.0f}(msec) optimal freq={Hz:.3f}(Hz)')
ax[2].set_ylabel('Au')
ax[2].set_xlabel('Offset [ms]')

plt.tight_layout()

plt.savefig('ass04ex1')

