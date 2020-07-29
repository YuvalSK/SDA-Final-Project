# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 13:54:30 2020
a Linear Discriminant Analysis algorithm
@author: Samoilov-Katz Yuval
"""
import numpy as np

class LDA:
    def __init__(self,n_comps):
        '''
        comps = number of components to leave after the reduction
        ld = linear discriminators
        var = explained variance though comps
        '''
        self.n_comps = n_comps
        self.ld = None
        self.var = 0.0
        
    def fit(self,data,y):
        '''
        fit the model (on data) based on tags (y)
        '''
        # preprocess the data
        fs = data.shape[1] #features
        cs = np.unique(y) #classes
        avg_t = np.mean(data,axis=0) #total average
        
        #initialize to deal with groups sum 
        sum_within = np.zeros((fs ,fs )) # sum inside each group
        sum_between = np.zeros((fs ,fs )) # sum between groups
        
        #dealing with each class seperatly
        for c in cs:
            #sum of class c of possible classes cs
            X_c = data[y==c]
            avg_c = np.mean(X_c,axis=0)
            sum_within += (X_c - avg_c).T.dot(X_c - avg_c)
            
            n_c = X_c.shape[0]
            avg_d = (avg_c - avg_t).reshape(fs,1)
            sum_between += n_c * (avg_d).dot(avg_d.T)
        
        #eigen values and vectors
        M = np.linalg.inv(sum_within).dot(sum_between)
        eigval, eigvec = np.linalg.eig(M)
        
        #sorting them from high to low
        eigvec = eigvec.T
        i = np.argsort(abs(eigval))[::-1]
        eigval = eigval[i]
        eigvec = eigvec[i]
        
        #save the number of components required
        self.ld = eigvec[0:self.n_comps]
        self.var = eigval
        
    def transform(self,data):
        #project upon the new dimensions
        return np.dot(data,self.ld.T)
    
    def explained_var(self):
        return self.var[0:self.n_comps] / np.sum(self.var)
        
        