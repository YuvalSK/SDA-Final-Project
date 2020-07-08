# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 10:06:39 2020

@author: samoi
"""

from numpy.fft import *
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
mat_data = sio.loadmat('q1data.mat')

data = mat_data['dataVec'][0]
filt = mat_data['filtVec']
fs = 10000 # Hz
xf = np.linspace(-fs/2, fs/2, fs)

y1 = []
n = len(data)
i=0
for j in range(0,n):
    temp = 0
    for k in range(0,len(filt)):
        temp+=filt[i,k] * data[j-k]
    y1.append(temp)

y2 = []
n = len(data)
i=1
for j in range(0,n):
    temp = 0
    for k in range(0,len(filt)):
        temp+=filt[i,k] * data[j-k]
    y2.append(temp)

yfd = 20* np.log10(np.abs(fftshift(fft(data))))
yf1 = 20* np.log10(np.abs(fftshift(fft(y1))))
yf2 = 20* np.log10(np.abs(fftshift(fft(y2))))

fig, ax = plt.subplots(figsize=(12,20), nrows=3, ncols=1)

iry1 = []
x = np.zeros_like(data)
x[0] = 1
n = len(x)
i=0
for j in range(0,n):
    temp = 0
    for k in range(0,len(filt)):
        temp+=filt[i,k] * x[j-k]
    iry1.append(temp)
    
iry2 = []
n = len(x)
i=1
for j in range(0,n):
    temp = 0
    for k in range(0,len(filt)):
        temp+=filt[i,k] * x[j-k]
    iry2.append(temp)

ax[0].plot(range(0,len(x)),iry1,c='b',label='f1')
ax[0].plot(range(0,len(x)),iry2,c='r',label='f2')
ax[0].legend()
ax[0].set_title(f'Impulse Response\nVec={x.T}')
ax[0].set_ylabel('y(n)')
ax[0].grid()


#ax[1].plot(range(0,len(data)),data,label='Data',c='k')
ax[1].plot(range(0,len(data)),y1,label='f1',c='b')
ax[1].plot(range(0,len(data)),y2,label='f2',c='r')
ax[1].set_title('Filters')
ax[1].set_ylabel('y(n)')
ax2 = ax[1].twinx()
ax2.plot(range(0,len(data)),data,label='Data(sp)',c='k')
ax2.set_ylabel('Data')
ax[1].legend()
ax[1].set_xlabel('10^-4 sec')
ax[1].grid()

ax[2].plot(xf,yfd,label='Data',c='k')
ax[2].plot(xf,yf1,label='f1',c='b')
ax[2].plot(xf,yf2,label='f2',c='r')
ax[2].set_title('DFT')
ax[2].set_xlabel('fs(Hz)')
ax[2].set_ylabel('Power(db)')
ax[2].grid()
ax[2].legend()

plt.tight_layout()
plt.savefig('ex1.png')