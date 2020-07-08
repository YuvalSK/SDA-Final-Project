# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 17:01:45 2020
SDA first HW, Due to 26/03/2020
@author: Samoilov-Katz Yuval id 204025258
"""
import numpy as np
import pandas as pd
import scipy.io as sio
import scipy.signal as sig
import matplotlib.pyplot as plt

#ex1#
#read 3 matlab EEG data and discuss ts stationary
def autocorr(x, t=0):
    return np.corrcoef(np.array([x[:-t], x[t:]]))[0,1]

plt.clf()
mat_data = sio.loadmat('ex1Question1.mat')
# each matrix contains 300*200
subjects = ['1','2','3']
for subject in subjects:
    eeg = np.array(mat_data['eeg{}'.format(subject)])
    df = pd.DataFrame(data=eeg)
    #mean and std for each trial
    df['ex{}'.format(subject)] = df.mean(axis=1)
    df['std'] = df.std(axis=1)
    eeg_res = [df['ex{}'.format(subject)].mean(),df['std'].std()]
    #plot each experiment, and present its mean and std
    plt.subplot(3,1,int(subject))
    df['ex{}'.format(subject)].plot()
    plt.title('experiment {0}: avg:{1:.3f}, std:{2:.3f}, autocorr:{2:.3f}'.format(subject,eeg_res[0],eeg_res[1],autocorr(df['ex{}'.format(subject)])))
    plt.ylabel('Freq (Hz)')
    plt.xlabel('Time (ms)')
    


plt.tight_layout()
plt.savefig('ex1_sec1.png')
