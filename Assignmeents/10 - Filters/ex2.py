# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 10:36:58 2020

@author: samoi
"""
import numpy as np
from numpy.fft import *
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy.signal import iirnotch, butter, filtfilt, impulse, freqz
import sunau

f=sunau.Au_read('audio.au')
audio_data = np.fromstring(f.readframes(f.getnframes()), dtype=np.int16).astype(np.float64)

fs = 16384 #sampling rate [Hz]
T = len(audio_data) / fs #duration of signal [s]
x = np.arange(0,T,1/fs) # time domain scale
midp = int(x.shape[0]/2) # for freq domain

yf = fftshift(fft(audio_data))
y = 20* np.log10(abs(yf))

fig, ax = plt.subplots(figsize=(12,20), nrows=2, ncols=1)

ax[0].plot(x[:midp]*fs/T,abs(y[:midp]),label='Data',c='k')
ax[0].set_title('DFT')
ax[0].set_ylabel('Power(db)')
ax[0].grid()

n = 15
f0 = 1300.0
nyq=fs/2
cut = f0/nyq

[blow,alow] = butter(n,cut,btype='low',analog=False)
w1, h1 = freqz(blow)
ylow = filtfilt(blow,alow,y)

w0=5050.0
bw=w0/2
[b, a] = iirnotch(w0,bw,fs=fs)
w2, h2 = freqz(b)
ynf = filtfilt(b,a,y)

ax[1].plot(0.5*fs*w1/np.pi,abs(h1)/np.mean(abs(h1)),label='LPF',c='b')
ax[1].plot(0.5*fs*w2/np.pi,abs(h2),c='m', label='NF')
ax[1].set_title('Frequency Response of Filters')
ax[1].grid()

yl = 20 * np.log10(abs(ylow))
yn = 20 * np.log10(abs(ynf))
yb = 20 * np.log10(abs(yb2))

ax[0].plot(x[:midp]*fs/T,abs(yl[:midp]), label='LPF',c='b')
ax[0].plot(x[:midp]*fs/T,abs(yn[:midp]), label='NF',c='m')

ax[0].legend()

plt.tight_layout()
plt.savefig('ex2')
#plt.show()

from scipy.io.wavfile import write

p1 = filtfilt(blow,alow,audio_data)
scaled = np.int16(p1/np.max(np.abs(p1)) * 32767)
write('audio_lpf.wav', fs, scaled)

p2 = filtfilt(b,a,audio_data)
scaled = np.int16(p2/np.max(np.abs(p2)) * 32767)
write('audio_nf.wav', fs, scaled)
