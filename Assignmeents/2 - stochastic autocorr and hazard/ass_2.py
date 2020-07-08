# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 17:21:00 2020
Single poisson process - TIH, Survaivor, Hazard and Autocorr
Created on Thu Mar 20 2020
SDA sec HW, Due to 23/04/2020
@author: Samoilov-Katz Yuval id 204025258
"""
import math
import numpy as np
import matplotlib.pyplot as plt
import pycorrelate

# Generate a "spike train" of a neuron 
samp = 1000
rate = 55 / samp
duration = 90
rp = 4 #reflactory msec
rc = 7 #recovery msec

#generate a random poi neuron with ref and rec periods
spk_array = []
for b in range(samp*duration):
    spike = np.random.poisson(lam=rate)
    if spike == 1:
        #reflactory period
        for t in range(rp):
            spk_array.append(0)
        #recovery period
        for i in range(rc):
            spike = np.random.poisson(lam=rate * (i+1)/rc)
            spk_array.append(spike) 
    else:
        spk_array.append(0)
spk_poi = np.array(spk_array[:samp*duration])
spk_poi[spk_poi > 1] = 0
time_array = np.arange(0, duration, 1/samp)
fig, ax = plt.subplots(figsize=(12,6), nrows=4, ncols=1)

#create TIH
bin_size = 1000
bin_array = np.zeros(math.ceil(samp*duration/bin_size))
for i, s in enumerate(np.arange(0, samp*duration, bin_size)):
    bin_array[i] = spk_poi[s:s+bin_size].sum()
hist_vals, hist_bins = np.histogram(bin_array, bins = np.arange(0, bin_array.max()+2))
bin_mean, bin_var = np.mean(bin_array), np.var(bin_array)
ff = bin_var / bin_mean
ax[0].bar(hist_bins[0:-1],hist_vals)
ax[0].set_xlabel('Spikes/Bin')
ax[0].set_ylabel('Count')
ax[0].set_title(f'TIH Poi Neuron\nFF={ff:.2f}')

#calculate Survivor and Hazard function
surv_vals = []
haz_vals=[]
cdf = 0
val = 0
for p in np.nditer(hist_vals):
    cdf +=p
    val = 1- cdf / np.sum(hist_vals)
    surv_vals.append(val)
    #calculate Hazard function
    haz_vals.append(p / np.sum(hist_vals)/ val)
haz_vals.insert(0,0) 
haz_vals[13] = 0
haz_vals[14] = 0
haz_vals[15] = 0

ax[1].bar(hist_bins[0:-1],surv_vals)
ax[1].set_xlabel('Spikes/Bin')
ax[1].set_ylabel('Probability')
ax[1].set_title(f'Survivor')

ax[2].plot(hist_bins,haz_vals)
ax[2].set_xlabel('Interval [ms]')
ax[2].set_ylabel('Probability')
ax[2].set_title(f'Hazard')

#calculate Autocorr
cbins = np.arange(-100,100,1)
x = pycorrelate.pcorrelate(spk_poi.nonzero()[0], spk_poi.nonzero()[0], bins=cbins)
normalized = (x-min(x))/(max(x)-min(x))
normalized[100]=0
ax[3].plot(cbins[:-1],normalized / duration * samp)
ax[3].set_title('Auto correlation')
ax[3].set_ylabel('Norm(Rate)\n[spikes/s]')
ax[3].set_xlabel('Offset [ms]')

plt.tight_layout()
plt.savefig('ass02ex02')