# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 15:15:47 2020

@author: samoi
"""
from numpy.fft import *
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy.signal import welch, hanning, csd, coherence

mat_data = sio.loadmat('sig3.mat')
signal = mat_data['fwrsig_nospikes'][0]
spike_train = mat_data['st'].toarray()[0]
fs = 24038 #Hz
T = len(signal) / fs
time_array = np.arange(9,T,1/fs)

nblock = 2* fs
f1, Pxx_singal = welch(signal, fs, nperseg=nblock)
f2, Pxx_spike = welch(spike_train, fs, nperseg=nblock)

res = 2
pduration = 70 * res
pstart = 3 * res

norm_sig = Pxx_singal[pstart:pduration] / np.mean(Pxx_singal[pstart*10:pduration])
norm_spike = Pxx_spike[pstart:pduration] / np.mean(Pxx_spike[pstart*10:pduration])

plt.plot(f1[pstart:pduration], norm_sig,c='b',label='signal')
plt.plot(f2[pstart:pduration], norm_spike,c='r',label='neuron')
plt.title('PSD')
plt.xlabel('Freq (Hz)')
plt.ylabel('Norm[Power]')
plt.legend()
plt.savefig('ex3sec1.png')

plt.clf()
signals = [norm_sig,norm_spike]
cs = ['b','r']
ls = ['signal','neuron']
t = 5
for i, s in enumerate(signals):
    avg = np.mean(s)
    std = np.std(s/avg)
    tresh = avg + t*std
    temp = []
    for val in s:
        if val > tresh:
            temp.append(val)
        else:
            temp.append(1)
            
    plt.plot(f1[pstart:pduration],temp,c=cs[i],label=ls[i])
    
plt.title(f'important oscilations subject to threshold \navg + {t} * std')
plt.xlabel('Freq (Hz)')
plt.ylabel('Treshold Norm[Power]')
plt.legend()
plt.savefig('ex3sec2.png')

plt.clf()
f3, Cxx = csd(signal, spike_train, fs=fs, nperseg=nblock)
plt.plot(f3[pstart:pduration],Cxx[pstart:pduration])
plt.ylabel('CSD (db)')
plt.xlabel('Freq (Hz)')
plt.title('cross spectral density between signal and neuron')
plt.savefig('ex3sec3.png')

plt.clf()
f4, Cxy = coherence(signal, spike_train, fs=fs, nperseg=nblock)
plt.plot(f3[pstart:pduration],Cxy[pstart:pduration])
plt.ylabel('Coherence {corr}')
plt.title('Coherence between signal and neuron')
plt.xlabel('Freq (Hz)')
plt.savefig('ex3sec4.png')
