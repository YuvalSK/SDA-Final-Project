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
mat_data = sio.loadmat('clusterData1.mat')['data']
m = mat_data.shape[0] #number of training samples
n = mat_data.shape[1] #number of features. Here 2-D
colors = ['r', 'b', 'g', 'k', 'm'] # colors for each cluster
clusters=5 
iterations=60 #how many time to rerun the algo
res = {} #results
cen = np.array([[-1,1,1,-1,0],[-1,-1,1,1,-1]]) #initial values of centroids

#step 2 - iterate over data
for i in range(iterations):
    e_d = np.array([]).reshape(m,0)
    
    #euclidian distances for each iteration
    for k in range(clusters):
           t=np.sqrt(np.sum((mat_data-cen[:,k])**2,axis=1))
           e_d=np.c_[e_d,t]
    
    #get index of minimun eclidian distance
    min_d_i = np.argmin(e_d,axis=1)+1
    
    t_res = {} 
    # temp results for each iteration
    
    for k in range(clusters):
        t_res[k+1]=np.array([]).reshape(2,0)
    for j in range(m):
        t_res[min_d_i[j]] = np.c_[t_res[min_d_i[j]],mat_data[j]]
    for k in range(clusters):
        t_res[k+1]=t_res[k+1].T
    for k in range(clusters):
         cen[:,k]=np.mean(t_res[k+1],axis=0)
    
    res=t_res

#step 3 - visualize data and clusters over data
fig, ax = plt.subplots(figsize=(14,10), nrows=2, ncols=1)
ax[0].scatter(mat_data[:,0],mat_data[:,1])
ax[0].set_xlabel('1-D')
ax[0].set_ylabel('2-D')
ax[0].set_title('Original Data')

labels=['Alef','Beit','Gimmel','Daled','Hey']
for c in range(clusters):
    ax[1].scatter(res[c+1][:,0], res[c+1][:,1], c=colors[c],label=labels[c])
    
ax[1].scatter(cen[0,:],cen[1,:],c='y',label='Centroids',s=600) 
ax[1].set_xlabel('1-D')
ax[1].set_ylabel('2-D')
ax[1].set_title(f'Clustered Data\nAfter {iterations} iterations')
ax[1].legend()

plt.tight_layout()
plt.savefig('ex8_sec3')  

