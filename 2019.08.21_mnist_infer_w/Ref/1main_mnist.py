#!/usr/bin/env python
# coding: utf-8
#import matplotlib.pyplot as plt
import numpy as np
import emachine as EM

np.random.seed(0)
##==============================================================================
# 2018.07.24: read MNIST data    
def read_mnist_data(file_name='mnist_train.csv'):    
    s0 = np.loadtxt(file_name,delimiter=',')
  
    s1 = s0[:,1:]        
    l,n = s1.shape
    s2 = np.full((l,n),-1.)
    #s2[s1>127.5] = 1.
    s2[s1>1.] = 1.

    # cut the boudaries 
    n1 = int(np.sqrt(n))
    # 2D to 3D
    s = s2.reshape(l,n1,n1) 
    # resize
    s = s[:,4:24,4:24]
    # 3D to 2D
    s = s.reshape(s.shape[0],-1)
    
    return s,s0[:,0]

# read training data   
# from personal PC
s,slabel = read_mnist_data(file_name='../MNIST_data/mnist_train.csv') 
# from biowulf
#s,slabel = read_mnist_data(file_name='/data/hoangd2/MNIST_data/mnist_train.csv')
#print(s.shape)

label = 8
i = slabel==label
slabel1 = slabel[i]
s1 = s[i]
print(s1.shape)

ops = EM.operators(s1)
print(ops.shape)

max_iter = 100
n_ops = ops.shape[1]

eps_list = np.linspace(0.5,0.9,5)
n_eps = len(eps_list)
E_eps = np.zeros((n_eps))
w_eps = np.zeros((n_eps,n_ops))
for i,eps in enumerate(eps_list):    
    w_eps[i,:],E_eps[i] = EM.fit(ops,eps=eps,max_iter=max_iter)
    print(eps,E_eps[i])

#optimal eps
ieps = np.argmax(E_eps[:])
print('optimal eps:',ieps,eps_list[ieps])
w = w_eps[ieps]

np.savetxt('w.dat',w,fmt='%f')
