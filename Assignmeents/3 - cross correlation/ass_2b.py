# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 16:59:30 2020

@author: samoi
"""

import numpy as np
import matplotlib.pyplot as plt
import pycorrelate

# Generate a "spike train" of a neuron 
samp = 1000
duration = 600
rate_A = 2
rate_B = 5
rate_C = 3

# Generate a regular neuron c
interval = samp/rate_A
spk_time_A = np.arange(interval/2,duration*samp, interval).astype(np.int32)
spk_array_A = np.zeros(duration*samp)
spk_array_A[spk_time_A] = 1

# Generate an affected neuron B
alpha = 0.04
t = np.arange(-300,301)
alpha_array = (rate_B/rate_A)*alpha**2*(t*(t>0))*np.exp(-alpha*t)
rate_array_B = np.convolve(spk_array_A, alpha_array, 'same')
spk_array_B = (np.random.uniform(size=samp*duration)<rate_array_B).astype(np.int32)
time_array = np.arange(0, duration, 1/samp)

# Generate an affected neuron A
beta = 0.02
t = np.arange(-300,301)
beta_array = (rate_C/rate_A)*beta**2*(t*(t>0))*np.exp(-beta*t)
rate_array_C = np.convolve(spk_array_A, beta_array, 'same')
spk_array_C = (np.random.uniform(size=samp*duration)<rate_array_C).astype(np.int32)
time_array = np.arange(0, duration, 1/samp)

plot_duration = 10000
fig, ax = plt.subplots(figsize=(16,4), nrows=3, ncols=1)
ax[0].plot(time_array[0:plot_duration], spk_array_A[0:plot_duration])
ax[0].set_ylabel('Spikes')
ax[0].set_title(f'Neuron C avg:{spk_array_A.sum()/duration:.1f}sp/sec')

ax[1].plot(time_array[0:plot_duration], spk_array_B[0:plot_duration])
ax[1].set_ylabel('Spikes')
ax[1].set_xlabel('Time [s]')
ax[1].set_title(f'Neuron B avg:{spk_array_B.sum()/duration:.1f}sp/sec')

ax[2].plot(time_array[0:plot_duration], spk_array_C[0:plot_duration])
ax[2].set_ylabel('Spikes')
ax[2].set_xlabel('Time [s]')
ax[2].set_title(f'Neuron A avg: {spk_array_C.sum()/duration:.1f}sp/sec')

plt.tight_layout()
plt.savefig('ass3ex2b')
plt.close()
plt.clf()

fig, ax = plt.subplots(figsize=(16,4), nrows=2, ncols=1)

cbins = np.arange(-250,250,1)
CBcorr = pycorrelate.pcorrelate(spk_array_C.nonzero()[0], spk_array_A.nonzero()[0], bins=cbins)
CAcorr = pycorrelate.pcorrelate(spk_array_B.nonzero()[0], spk_array_A.nonzero()[0], bins=cbins)

ax[0].plot(cbins[:-1],CBcorr / duration * samp)
ax[0].set_title('Cross correlation (C-->B)')
ax[0].set_ylabel('Rate [spikes/s]')
ax[0].set_xlabel('Offset [ms]')

ax[1].plot(cbins[:-1],CAcorr / duration * samp)
ax[1].set_title('Cross correlation (C-->A)')
ax[1].set_ylabel('Rate [spikes/s]')
ax[1].set_xlabel('Offset [ms]')

plt.tight_layout()
plt.savefig('ass3ex2c')


