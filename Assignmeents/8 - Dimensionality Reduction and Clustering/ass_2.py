# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 18:09:44 2020

@author: samoi
"""

from scipy.linalg import cholesky
from scipy.stats import pearsonr
import numpy as np

corr = np.array([[1.0 , -.99],
                [-.99,1.0]])

up  = cholesky(corr)
rnd = np.random.normal(0,1,size=(10000,2))

ans =  rnd @ up

corr_0_1 , _ = pearsonr(ans[:,0], ans[:,1])

eigen_val,eigen_vec = np.linalg.eig(corr)