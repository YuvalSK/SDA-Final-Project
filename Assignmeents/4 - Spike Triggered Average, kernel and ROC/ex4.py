# -*- coding: utf-8 -*-
"""
Created on Monday May 3 
SDA fourth HW section, Due to 7/05/2020
@author: Samoilov-Katz Yuval id 204025258
"""
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import sklearn.metrics as metrics

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

mat_data = sio.loadmat('Q4Data.mat')
leftR = mat_data['leftRate'] 
rightR = mat_data['rightRate']

x = rightR / np.sum(rightR)
y = np.zeros((1000,1))
y = (rightR>leftR)

fpr, tpr, threshold = metrics.roc_curve(y, x)
roc_auc = metrics.auc(fpr, tpr)

bz_80 = np.take(threshold,np.where(tpr == find_nearest(tpr, 0.8))[0])[0]*100
cz_80 = np.take(threshold,np.where(fpr == find_nearest(fpr, 0.8))[0])[0]*100

plt.title(f'ROC curve\nTPR treshold for 80%: {bz_80:.2f}%\nFPR treshold for 80%: {cz_80:.2f}%')
plt.plot(fpr, tpr, label = 'AUC = %0.3f' % roc_auc)
plt.legend()
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('1-beta (TPR)')
plt.xlabel('alpha(FPR)')

plt.tight_layout()
plt.savefig('ass04ex4')
