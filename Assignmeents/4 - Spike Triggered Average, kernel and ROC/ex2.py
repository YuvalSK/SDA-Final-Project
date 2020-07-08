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
duration = 60

mat_data = sio.loadmat('kernel.mat')
spk_raw = mat_data['resp']
stm_sig = mat_data['stim'][0]
spk_array = np.mean(spk_raw, axis=0)
time_array = np.arange(0, duration, 1/samp)

fig, ax = plt.subplots(figsize=(12,6), nrows=3, ncols=1)
avg_sig = np.mean(stm_sig)
Nstm_sig = stm_sig - avg_sig
ax[0].plot(time_array,Nstm_sig)
ax[0].set_ylabel('Normalized (Mu)')
ax[0].set_title(f'Stimulus Mu(t) = {avg_sig* samp:.3f} MAu/sec')
ax[0].set_xlabel('sec')

bins=[100, 300, 800, 1000, 1300]
for i, b in enumerate(bins):
    win_array = np.ones(b)/b
    conv_array = np.convolve(spk_array, win_array, 'same')
    ax[1].plot(time_array,conv_array,label=b)
avg_spk = np.mean(spk_array) 
ax[1].set_ylabel('conv r(t)')
ax[1].set_title(f'convolution best bins {100} Neuron r(t) = {avg_spk*100:.3f}% rate')
ax[1].set_xlabel('sec')
fig.legend(loc = 'center right')

cbins = np.arange(-100,100,1)
# kernel <D(t)> = E(t)/V(t) * <C(t)>
D_t = np.flip(pycorrelate.pcorrelate(stm_sig, spk_array, bins=cbins)) * avg_spk / np.var(stm_sig)
ax[2].plot(cbins[:-1], D_t / avg_spk)
ax[2].set_title(f'linear kernel (stimulus--->Spike)')
ax[2].set_ylabel('Norm(Mu) by rate')
ax[2].set_xlabel('Offset [ms]')

plt.tight_layout()
plt.savefig('ass04ex2')
