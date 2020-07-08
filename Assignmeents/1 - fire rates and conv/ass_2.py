# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 17:01:45 2020
SDA first HW section 2, Due to 26/03/2020
@author: Samoilov-Katz Yuval id 204025258
"""
import numpy as np
import pandas as pd
import scipy.io as sio
import scipy.signal as sig
import matplotlib.pyplot as plt

#sec 2 subsec 1
samp = 1000 #msec
duration = 60 #sec
time_array = np.arange(0, duration, 1/samp)

mat_data = sio.loadmat('ex1Question2.mat')
data = np.around(mat_data['spikeTimes'][0])
spk_data = np.zeros((samp*duration, 1))
for spike in data:
    spk_data[int(spike)] = 1

def rate_plot(x, y, ax, legend, samp=1000):
    ax.plot(x,y*samp,label=legend)
    ax.set_ylabel('Firing rate\n[sp/s]')
    ax.set_xlabel('Time(sec)')
    ax.grid()
    ax.legend(loc='right')
    
fig, ax = plt.subplots(figsize=(15,20), nrows=5, ncols=1)

#sec 2 subsec a
rate_plot(time_array, spk_data, ax[0], 'Spike train\nAvg fire rate: {0:.3f} sp/sec'.format(spk_data.mean()))

#sec 2 subsec b - bins fire rate
bin_sizes = [200, 500, 1000, 3000]
for bin_size in bin_sizes:
    bin_array = np.zeros_like(spk_data, dtype=np.float32)
    for i in np.arange(0, samp*duration, bin_size):
        bin_array[i:i+bin_size] = spk_data[i:i+bin_size].sum() / bin_size
    rate_plot(time_array, bin_array, ax[1], '{0:.1f}'.format(bin_size/1000))

#sec 2 subsec 2 c - convs comparison
def Myconv(x, y):
    '''
    Home made 2D convolution function
    input: x, y dtype=np.arrays
    output: z dtype=np.array'''
    z_conv = 0
    z_conv = np.zeros_like(x, dtype=np.float32)
    x_height = len(x)
    y_height = len(y) 
    dh = y_height//2
    
    for i in range(dh,x_height-dh):
        for j in range(y_height):
            conv_res = 0
            #conv the values
            conv_res += y[j]*x[i-dh]        
        #Save results to a new matrix z
        z_conv[i] = conv_res
    return(z_conv)

rate = 28/samp
x = (np.random.uniform(size=samp*duration)<rate).astype(np.int32)
y_size = 6
y_array = np.ones(y_size)/y_size
my_conv = Myconv(x, y_array)
rate_plot(time_array, my_conv, ax[2], 'My conv =)')
evil_conv = np.convolve(x, y_array, 'same')
rate_plot(time_array, evil_conv, ax[2], 'Numpy conv =(')

#sec 2 subsec 2 d - rectangle window
win_sizes = [200, 500, 1000, 3000]
for win_size in win_sizes:
    win_array = np.ones(win_size)/win_size
    rec_conv = np.convolve(spk_data[:,0], win_array, 'same')
    rate_plot(time_array, rec_conv, ax[3], '{0:.1f}'.format(win_size/1000))

#sec 2 subsec 2 e - gaussian window
win_sizes = [800, 450]
conv_stds = [250, 100]
for win_size, conv_std in zip(win_sizes, conv_stds):
    gauss_array = sig.gaussian(win_size, conv_std) / samp
    gauss_conv = np.convolve(spk_data[:,0], gauss_array, 'same')
    rate_plot(time_array, gauss_conv, ax[4], '(win,std)=({0:.2f},{1:.2f})'.format((win_size/1000),(conv_std/1000)))

#plt.tight_layout()
plt.suptitle('Firing rates and convolution')
plt.savefig('ex1_sec2')