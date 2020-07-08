# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 10:54:40 2020

@author: samoi
"""

from numpy.fft import *
import numpy as np
import matplotlib.pyplot as plt

fs = 250 # Hz
T = 40000  # msec

a = 10
b = 110

x = np.linspace(0, T, fs)
y = 0.5 * (np.cos((a-b) * 2.0*np.pi*x) + np.cos((a+b) * 2.0*np.pi*x))
yf = fftshift(fft(y))
xf = np.linspace(-fs/2, fs/2, fs)

A = 20* np.log10(np.abs(np.max(yf)))

plt.xlabel('Frequency(Hz)')
plt.title(f'Amp {A:.2f}')
plt.ylabel('Power (db)')
plt.grid()
plt.plot(xf, np.abs(yf))
plt.savefig('ass09ex01sece.png')