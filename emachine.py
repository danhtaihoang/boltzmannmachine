#!/usr/bin/env python
# coding: utf-8

import numpy as np
#=========================================================================================
# convert s[n_seq,n_var] to ops[n_seq,n_ops]
def operators(s):
    #generate terms in the energy function
    n_seq,n_var = s.shape
    ops = np.zeros((n_seq,n_var+int(n_var*(n_var-1)/2.0)))

    jindex = 0
    for index in range(n_var):
        ops[:,jindex] = s[:,index]
        jindex +=1

    for index in range(n_var-1):
        for index1 in range(index+1,n_var):
            ops[:,jindex] = s[:,index]*s[:,index1]
            jindex +=1
            
    return ops
#=========================================================================================
# 
def energy_ops(ops,w):
    return np.sum(ops*w[np.newaxis,:],axis=1)
#=========================================================================================
# 
def generate_seq(n_var,n_seq,n_sample=30,g=1.0):
    n_ops = n_var+int(n_var*(n_var-1)/2.0)
    #w_true = g*(np.random.rand(ops.shape[1])-0.5)/np.sqrt(float(n_var))
    w_true = np.random.normal(0.,g/np.sqrt(n_var),size=n_ops)
    
    samples = np.random.choice([1.0,-1.0],size=(n_seq*n_sample,n_var),replace=True)
    ops = operators(samples)

    #sample_energy = energy_ops(ops,w_true)
    sample_energy = ops.dot(w_true)

    p = np.exp(sample_energy)
    p /= np.sum(p)
    out_samples = np.random.choice(np.arange(n_seq*n_sample),size=n_seq,replace=True,p=p)
    
    return w_true,samples[out_samples]
#=========================================================================================
# find coupling w from sequences s
# input: ops[n_seq,n_ops]
# output: w[n_ops], E_av
def fit(ops,eps=0.1,max_iter=151,alpha=0.1):
    E_av = np.zeros(max_iter)
    n_ops = ops.shape[1]
    cov_inv = np.eye(ops.shape[1])

    np.random.seed(13)
    w = np.random.rand(n_ops)-0.5    
    for i in range(max_iter):                 
        #energies_w = energy_ops(ops,w)
        energies_w = ops.dot(w)

        probs_w = np.exp(energies_w*(eps-1))
        z_data = np.sum(probs_w)
        probs_w /= z_data
        ops_expect_w = np.sum(probs_w[:,np.newaxis]*ops,axis=0)

        E_av[i] = energies_w.mean()
        w += alpha*cov_inv.dot((ops_expect_w - w*eps))        
        
    return w,-E_av[-1]
#=========================================================================================
