#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as nplin
import itertools

def operators(s, RBM=0):
    #generate terms in the energy function
    n_seq,n_var = s.shape
    if RBM==0:
        out = np.zeros((n_seq,n_var+int(n_var*(n_var-1)/2.0)))
    else:
        out = np.zeros((n_seq,n_var+(n_var-RBM)*RBM))
    jindex = 0
    for index in range(n_var):
        out[:,jindex] = s[:,index]
        jindex +=1
    if RBM==0:
        for index in range(n_var-1):
            for index1 in range(index+1,n_var):
                out[:,jindex] = s[:,index]*s[:,index1]
                jindex +=1
    else:
        for index in range(n_var-RBM):
            for index1 in range(n_var-RBM,n_var):
                out[:,jindex] = s[:,index]*s[:,index1]
                jindex +=1
    return out
def energy_ops(ops,w):
    return np.sum(ops*w[np.newaxis,:],axis=1)#/np.power(w.shape[0],0.25)
#def energy(w,seqs):
#    return np.mean(seqs*seqs.dot(w),axis=1)
def generate_seqs(n_var,n_seq,n_sample=30,large=False,num_large=4,RBM=0):
    samples = np.random.choice([1.0,-1.0],size=n_var*n_seq*n_sample,replace=True).reshape(n_seq*n_sample,n_var)
    ops = operators(samples,RBM=RBM)
    n_ops = ops.shape[1]
##    w_true = np.random.rand(n_var+int(n_var*(n_var-1)/2.0))-0.5
    w_true = (np.random.rand(ops.shape[1])-0.5)/np.sqrt(float(n_var))
##     if RBM > 0: w_true[n_var:] *= np.sqrt(float(n_var))
    if large and num_large>0: #putting in some large couplings
        indices_large = np.random.choice(range(n_var,n_ops),size=num_large,replace=False)
        for l_index in range(num_large):
            w_true[indices_large[l_index]] *= 10.0

    sample_energy = energy_ops(ops,w_true)
    p = np.exp(sample_energy)
    p = p/np.sum(p)
    out_samples = np.random.choice(np.arange(n_seq*n_sample),size=n_seq,replace=True,p=p)
    return w_true,samples[out_samples],p[out_samples],sample_energy[out_samples]

def hopfield_model(s):
    ops = operators(s)
    w=np.mean(ops,axis=0)
    print('hopfield error ',nplin.norm(w-w_true))
    return w

def boltzmann_machine_exact(w,s,s_all,max_iter=151,alpha=5e-2,cov=False):
    n_seq,n_var = s.shape
    ops = operators(s)
    cov_inv = np.eye(ops.shape[1])
    ops_obs = np.mean(ops,axis=0)
    ops_model = operators(s_all)

    for iterate in range(max_iter):
        energies_w = energy_ops(ops_model,w)
        probs_w = np.exp(energies_w)
        probs_w /= np.sum(probs_w)
        if iterate%int(max_iter/5.0)==0: 
            print(iterate,nplin.norm(w-w_true)) #,nplin.norm(spin_cov_w-spin_cov_obs))
        w += alpha*cov_inv.dot(ops_obs - np.sum(ops_model*probs_w[:,np.newaxis],axis=0))
    print('final ',iterate,nplin.norm(w-w_true))

def boltzmann_machine_sample(w,s,max_iter=151,n_partition=1,alpha=5e-2,cov=False):
    n_seq,n_var = s.shape
    ops = operators(s)
    cov_inv = np.eye(ops.shape[1])

    ops_obs = np.mean(ops,axis=0)
    samples_part = np.random.choice([1.0,-1.0],size=n_var*n_seq*n_partition,replace=True).reshape(n_seq*n_partition,n_var)
    ops_model = operators(samples_part)

    for iterate in range(max_iter):
        energies_w = energy_ops(ops_model,w)
        probs_w = np.exp(energies_w)
        probs_w /= np.sum(probs_w)
        if iterate%int(max_iter/5.0)==0: 
            print(iterate,nplin.norm(w-w_true)) #,nplin.norm(spin_cov_w-spin_cov_obs))
        w += alpha*cov_inv.dot(ops_obs - np.sum(ops_model*probs_w[:,np.newaxis],axis=0))
    print('final ',iterate,nplin.norm(w-w_true))

def eps_machine(w,s,eps_scale=.01,max_iter=151,alpha=0.1,cov=False):
    n_seq,n_var = s.shape
    ops = operators(s)
    n_ops = ops.shape[1]
    cov_inv = np.eye(ops.shape[1])

    for iterate in range(max_iter):
        obs_energies_w = energy_ops(ops,w)
        energies_w = obs_energies_w.reshape(n_seq,1)
        probs_w = np.exp(-energies_w*(1-eps_scale)).reshape(n_seq)
        probs_w /= np.sum(probs_w)
        ops_expect_w = np.sum(probs_w[:,np.newaxis]*ops.reshape(n_seq,n_ops),axis=0)
        if iterate%int(max_iter/5.0)==0:
            invprob_w = np.exp(-energies_w).reshape(n_seq)
            invprob_w /= np.sum(invprob_w)
            ops_expectation = np.sum(invprob_w[:,np.newaxis]*ops.reshape(n_seq,n_ops),axis=0)
            print(eps_scale,iterate,nplin.norm(w-w_true),np.log(np.mean(ops_expectation*ops_expectation)))
        #sec_order = np.tanh(w*eps_scale)
        sec_order = w*eps_scale
        w += alpha*cov_inv.dot((ops_expect_w - sec_order))
        
    print('final ',eps_scale,iterate,nplin.norm(w-w_true))

n_var,n_seq = 25,10000 #60, 6000
np.random.seed(0)
w_true,seqs,probs,energies = generate_seqs(n_var,n_seq)
#s_all = np.asarray(list(itertools.product([1.0, -1.0], repeat=n_var)))
#s_all.shape

for n_part in range(10000,20001,20000):
    print('n_seq ',n_part)
    print('epsilon ')
    for eps in [0.05,0.1,0.2,0.3,0.6]:
        np.random.seed(13)
        w_trial3 = np.random.rand(n_var+int(n_var*(n_var-1)/2.0))-0.5
        eps_machine(w_trial3,seqs[:n_part],eps_scale=eps,cov=False)
 #       eps_machine_unlearning(w_trial3,seqs[:n_part],eps_scale=eps,cov=False)
        
    print('Hopfield ')
    w_hopfield=hopfield_model(seqs[:n_part])

    np.random.seed(13)
    w_trial1 = np.random.rand(n_var+int(n_var*(n_var-1)/2.0))-0.5
    print('Boltzmann (sample) ')
    boltzmann_machine_sample(w_trial1,seqs[:n_part],cov=False)
    
#    np.random.seed(13)
#    w_trial2 = np.random.rand(n_var+int(n_var*(n_var-1)/2.0))-0.5
#    print('Boltzmann (exact) ')
#    boltzmann_machine_exact(w_trial2,seqs[:n_part],s_all,cov=False)
    
fig, ax = plt.subplots()    
lims = [np.min([w_true, w_trial1]), np.max([w_true, w_trial1])]
ax.scatter(w_true,w_hopfield,label='Hopfield', alpha=0.4)
ax.scatter(w_true,w_trial1,label='BM_sample',alpha=0.4)
#ax.scatter(w_true,w_trial2,label='BM_exact',alpha=0.4)
ax.scatter(w_true,w_trial3,label='eps',alpha=0.4)
ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
ax.legend()
plt.show()
