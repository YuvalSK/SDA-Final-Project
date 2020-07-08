# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18
SDA HW Due to 18/06/2020
@author: Samoilov-Katz Yuval id 204025258
"""
import numpy as np
import pandas as pd
import scipy.io as sio
import scipy.signal as sig
import scipy.stats as sts
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn import preprocessing

mat_data = sio.loadmat('dataPCA.mat')['lfp']
# (2000,20) where N=2000 each row trials and M=20 samples per trial (4KHz)

def entropy(data):
    #probablity for entropy by bins
    pdata, _ , b_n = sts.binned_statistic(data,data,bins=10,statistic='count') 
    p_data = pdata / np.sum(pdata)
    ent_d = 0
    for i in p_data:
        if i == 0:
            pass
        else:
            ent_d += i*np.log2(i) 
    return -ent_d
    
def PCA2D(data):
    means = np.outer(np.mean(data,axis=1),np.ones(data.shape[1])) #avg over rows, for each value to be substract from
    X = data - means #minus avg over trials
    cov_without_n = X.T.dot(X) # like cov but without dividing by n
    eigen_val,eigen_vec = np.linalg.eig(cov_without_n) #extract eigean values from X*X.T
    #stores in dict to save association
    res = dict(zip(eigen_val, eigen_vec))
    s_eigen_val = sorted(eigen_val,reverse=True) #sort descending
    
    pc1 = res[s_eigen_val[0]]
    pc2 = res[s_eigen_val[1]]
    
    t = (s_eigen_val[0] + s_eigen_val[1]) / np.sum(np.abs(eigen_val)) # total variance explained by PC1 and PC2
    
    return X, pc1, pc2, s_eigen_val[0]/ np.sum(np.abs(eigen_val)) , s_eigen_val[1]/ np.sum(np.abs(eigen_val)), t

fig, ax = plt.subplots(figsize=(8,12), nrows=4, ncols=1)

norm_m, x, y, xvar, yvar, tvar = PCA2D(mat_data)

norm_data = preprocessing.scale(mat_data)
pca = PCA()
pca.fit(norm_data)
pca_data = pca.transform(norm_data)

max_v = 0
count = 0

ax[0].scatter(x,y)
ax[0].set_title('PCA Analysis using my code')
ax[0].set_xlabel(f'PC1')
ax[0].set_ylabel(f'PC2')
ax[0].grid()

px = []
py = []

for r in norm_m:
    px.append(x.dot(r))
    py.append(y.dot(r))
    
ax[1].hist(px,density=True, bins=10)

ax[1].set_xlabel(f'PC1')
ax[1].set_ylabel(f'pdf')

ax[2].hist(py,density=True, bins=10)
ax[2].set_xlabel(f'PC2')
ax[2].set_ylabel(f'pdf')

ax[3].scatter(px,py)
ax[3].set_xlabel(f'PC1-{xvar*100:.2f}% explained')
ax[3].set_ylabel(f'PC2 -{yvar*100:.2f}% explained')
ax[3].set_title(f'total variance explained by the first 2 PCs {tvar*100:.2f}%')

for i,c in enumerate(norm_m.T):
    max_var_f = np.var(c,ddof=1)
    if max_var_f > max_v:
        max_v=max_var_f
        count = i
        
max_v_f = entropy(mat_data[:,count])
pc1_e = entropy(x.dot(mat_data.T))

print(f'entropy of max var \nfeature number{count}: {max_v_f}')
print(f'entropy of pc1: {pc1_e}')

plt.tight_layout()
plt.savefig('ex8_sec1.png')


#automatic PCA
fig, axes = plt.subplots(figsize=(8,12), nrows=2, ncols=1)    

exp_var = np.round(pca.explained_variance_ratio_ *100, decimals=2)
labels = ['PC' + str(n) for n in range(1,len(exp_var)+1)]      
pca_df = pd.DataFrame(pca_data, columns=labels)

axes[0].bar(x=range(1,len(exp_var)+1),height=exp_var, tick_label=labels,color='b')
axes[0].set_ylabel('percentage of explained Variance\n (%)')
axes[0].set_xlabel('PC')
axes[0].set_title(f'PCA Analysis sklearn')    

axes[1].scatter(pca_df.PC1,pca_df.PC2)
axes[1].set_title(f'PCA Analysis')
axes[1].set_xlabel(f'PC1 - {exp_var[0]}%')
axes[1].set_ylabel(f'PC2 - {exp_var[1]}%')
    
plt.tight_layout()
plt.savefig('ex1_sklearn_PCA.png')

