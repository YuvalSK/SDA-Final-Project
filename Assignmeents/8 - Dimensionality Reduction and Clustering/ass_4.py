# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 15:27:40 2020

@author: samoi
"""
import numpy as np
import pandas as pd
import scipy.io as sio
import scipy.signal as sig
import scipy.stats as sts
import matplotlib.pyplot as plt

#step 1 - initialize variables
mat_data = sio.loadmat('clusterData2.mat')['data']
n = mat_data.shape[0] 
d = mat_data.shape[1] #number of features. Here 2-D
k = 3
cov = []
for i in range(k):
    cov.append(mat_data.T.dot(mat_data))
cov = np.array(cov)


e=0.001
iterations = 60
labels=['Alef','Beit','Gimmel','Daled','Hey']
colors = ['r', 'b', 'g', 'k', 'm'] # colors for each cluster
w = np.ones((k))/k #initiate weights
ms = np.random.choice(mat_data.flatten(),(k,d))

fig, ax = plt.subplots(figsize=(12,16), nrows=3, ncols=1)


for i in range(iterations):
    #step 2.1 Expectation
    lh=[]
    for j in range(k):
        lh.append(sts.multivariate_normal.pdf(x=mat_data,mean=ms[j], cov=cov[j]))
    lh = np.array(lh)
    
    #step 2.2 Maximization
    m=[]
    for j in range(k):
        m.append((lh[j] * w[j])/ (np.sum([lh[i]*w[i] for i in range(k)],axis=0)+e))
        ms[j] = np.sum(m[j].reshape(n,1) * mat_data,axis=0) / (np.sum(m[j]+e))
        cov[j] = np.dot((m[j].reshape(n,1) * (mat_data - ms[j])).T, (mat_data - ms[j])) / (np.sum(m[j])+e)
        w[j] = np.mean(m[j])
    
    # visualize the learned clusters
    if i%2 == 0 :
        if i == iterations/2:
            i=1
        else:
            i=2
            
        lh = []
        for j in range(k):
          lh.append(sts.multivariate_normal.pdf(x=mat_data, mean=ms[j], cov=cov[j]))
        lh = np.array(lh)
        pred = np.argmax(lh, axis=0)
        for c in range(k):
          pred_k = np.where(pred == c)
          ax[i].scatter(mat_data[pred_k[0],0], mat_data[pred_k[0],1], color=colors[c], alpha=0.2, edgecolors='none', marker='s')
        ax[i].scatter(mat_data[:,0], mat_data[:,1], facecolors='none', edgecolors='grey')
        
        for j in range(k):
          ax[i].scatter(ms[j][0], ms[j][1], color=colors[j],label=labels[j])

#step 3 - visualize data and clusters over data
ax[0].scatter(mat_data[:,0],mat_data[:,1])
ax[0].set_xlabel('1-D')
ax[0].set_ylabel('2-D')
ax[0].set_title('Original Data')

ax[1].set_title(f'Clustered Data\nAfter {iterations/2} iterations')
ax[1].set_xlabel('avg')
ax[1].set_ylabel('std')
ax[1].legend()

ax[2].set_title(f'Clustered Data\nAfter {iterations} iterations')
ax[2].set_xlabel('avg')
ax[2].set_ylabel('std')

plt.tight_layout()
plt.savefig('ex8_sec4_3')  

