# -*- coding: utf-8 -*-
"""
SDA sixth HW section, Due to 25/05/2020
@author: Samoilov-Katz Yuval id 204025258
"""
import logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
import time
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import math

mat_data = sio.loadmat('gs_Data.mat')
data = mat_data['observations']

def LL(x, m=0.5, array=False):
    if array:
        product = 0
        for i in x:
            product += ((i-m)**2)+((i+m)**2)
        return ((-1/2)*(1/math.sqrt(2*math.pi)))*product
    return (1/math.sqrt(2*math.pi))*(np.exp(-0.5*(x-m)**2)+np.exp(-0.5*(x+m)**2))

w = (3 - math.sqrt(5))/2
c , a = 1., 0.
b = a + (c - a) * w
fa, fb, fc = [LL(data[0],x,array=True) for x in [a, b,c]]
epsilon = 0.01

'''
x = np.arange(a,c,epsilon)
y = LL(x)
plt.plot(x,y)
'''

count = 1
while (c-a) > epsilon:
    logging.info(f'Iteration {count}: LL(A={a:.2f}) = {fa:.2f}, LL(B={b:.2f}) = {fb:.2f}, LL(C={c:.2f}) = {fc:.2f}') 
    if (c-b > b-a):
        m = c - (c - a) * w
        fm = LL(data[0],m,array=True)
        logging.info(f'm between A and B:  LL(m={m:.2f}) = {fm:.2f}')
        if fm < fb:
            c, fc = m, fm
        else:
            a, fa = b, fb
            b, fb = m, fm
    else:
        m = a + (c - a) * w
        fm = LL(data[0],m,array=True)
        logging.info(f'm between B and C:  LL(m={m:.2f}) = {fm:.2f}')
        if fm < fb:
            c, fc = b, fb
            b, fb = m, fm
        else:                        
            a, fa = m, fm
    count += 1
