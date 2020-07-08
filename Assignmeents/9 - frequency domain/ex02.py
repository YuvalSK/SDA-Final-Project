# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 22:28:06 2020

@author: samoi
"""
from numpy.fft import *
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
mat_data = sio.loadmat('handel.mat')

fs = float(mat_data['Fs'][0,0]) # Hz
T = len(mat_data['y'])  # msec

y = mat_data['y']
yf = fftshift(fft(y))
xf = np.linspace(-fs/2, fs/2, T,dtype=int)

yuvi = dict(zip(xf, np.abs(yf)))
max_p = sorted(yuvi,key=yuvi.get,reverse=True)[0]
yuvi_s = sorted(yuvi,key=yuvi.get,reverse=True)[1]

A1 = yuvi[max_p][0]
A2 = yuvi[yuvi_s][0]

plt.xlabel('Frequency(Hz)')
plt.title(f'Highest powers: {A1:.2f},{A2:.2f}\n freqs: {max_p:.0f},{yuvi_s:.0f}')
plt.ylabel('Power (db)')
plt.grid()
plt.plot(xf, 20*np.log10(np.abs(yf)))
plt.axvline(x=max_p,c='r')
plt.axvline(x=yuvi_s,c='g')
plt.savefig('ass09ex02.png')
