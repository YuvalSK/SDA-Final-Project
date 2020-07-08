# -*- coding: utf-8 -*-
"""
Stochastic point process - Fano Factor and Cv for Diff window sizes
Created on Thu Mar 20 2020
SDA sec HW, Due to 23/04/2020
@author: Samoilov-Katz Yuval id 204025258
"""
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import math

#ex1
# Generate a "spike train" of a neuron 
samp = 1000 #msec
mat_data = sio.loadmat('ex2Question1.mat') # import our beautiful neuron
data = np.array(mat_data['spikeTimes']*samp) # msec data
duration = 10688 # sec

# clean data for spikes and times
raw_array = data.astype(np.int32)
spk_array = np.zeros(duration*samp)
spk_array[raw_array] = 1

#prepare for presenting data
fig, ax = plt.subplots(figsize=(14,6), nrows=3, ncols=1)

#calculate FF and Cv using real and requested bin sizes
win_sizes = [duration, 1000, 25]
for a, bin_size in enumerate(win_sizes):
    if a == 0:
        time_array = np.arange(0, duration, 1/samp)
        ax[a].plot(time_array[0:bin_size], spk_array[0:bin_size])
        avg = np.mean(spk_array)
        ax[a].set_ylabel('Spikes')
        ax[a].set_title(f'Neuron Chaviv r(t) = {avg:.3f} sp/sec')
        ax[a].set_xlabel('10^3 sec')
    else:
        bin_array = np.zeros(math.ceil(samp*100/bin_size))
        for i, s in enumerate(np.arange(0, samp*100, bin_size)):
            bin_array[i] = spk_array[s:s+bin_size].sum()
        hist_vals, hist_bins = np.histogram(bin_array, bins = np.arange(0, bin_array.max()+2))
        bin_mean, bin_var = np.mean(bin_array), np.var(bin_array)
        ff = bin_var / bin_mean
        cv = np.sqrt(bin_var) / bin_mean
        ax[a].bar(hist_bins[0:-1],hist_vals)
        ax[a].set_xlabel('Spikes/Bin')
        ax[a].set_ylabel('Count')
        ax[a].set_title(f'T={bin_size} ms Mean {bin_mean:.2f} FF={ff:.2f} cv = {cv:.2f}')

plt.tight_layout()
plt.savefig('ass02ex01')