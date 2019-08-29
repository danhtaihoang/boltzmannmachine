#!/usr/bin/env python
# coding: utf-8

import numpy as np
#import matplotlib.pyplot as plt
import emachine as EM
import itertools

np.random.seed(0)
#=========================================================================================
# data
s0 = np.loadtxt('../MNIST_data/mnist_test.csv',delimiter=',')
seq = s0[:,1:] 
label = s0[:,0]
#print(seq.shape,label.shape)

# select only 1 digit
digit = 8
t = 2
i = label == digit
label1 = label[i]

seq1 = seq[i][t]
print(digit,seq1.shape)

# convert to binary
seq1 = np.sign(seq1-1.5)

w = np.loadtxt('w.txt')
cols_active = np.loadtxt('cols_selected.txt').astype(int)
cols_conserved = np.setdiff1d(np.arange(28*28),cols_active)

#=========================================================================================
# select hidden
#hidden = np.loadtxt('cols_hidden.dat').astype(int)
#n_hidden = len(hidden)

# select hidden as random
n_hidden = 100
hidden = np.random.choice(np.arange(28*28),n_hidden,replace=False)
hidden_active = np.intersect1d(hidden,cols_active)
hidden_conserved = np.intersect1d(hidden,cols_conserved)

n_hidden_active = len(hidden_active)
n_hidden_conserved = len(hidden_conserved)
print('n_hidden_active:',len(hidden_active))

#n_hidden_active = 16
#n_hidden_conserved = 184
# hidden from active cols
#cols_active = np.loadtxt('cols_selected.txt').astype(int)
#hidden_active = np.random.choice(cols_active,n_hidden_active,replace=False)
#print(len(hidden_active))

# hidden from conserved cols
#cols_conserved = np.setdiff1d(np.arange(28*28),cols_active)
#hidden_conserved = np.random.choice(cols_conserved,n_hidden_conserved,replace=False)
#print(len(hidden_conserved))
# hidden
#hidden = np.hstack([hidden_active,hidden_conserved])

#=========================================================================================
seq_all = np.asarray(list(itertools.product([1.0, -1.0], repeat=n_hidden_active)))
n_possibles = seq_all.shape[0]
print('number of possible configs:',n_possibles)

active_hidden_indices = np.intersect1d(cols_active,hidden_active,return_indices=True)[1]

# consider only one test image
#t = 2
seq_active = seq1[cols_active]

seq_active_possibles = np.tile(seq_active,(n_possibles,1))
seq_active_possibles[:,active_hidden_indices] = seq_all

# recover hidden
npart = 128
ns = int(n_possibles/npart)

energy = np.full(n_possibles,-100000.)
for i in range(npart):
    i1 = int(i*ns)
    i2 = int((i+1)*ns)
    if i%5 == 0: print('ipart:',i)
    ops = EM.operators(seq_active_possibles[i1:i2])
    energy[i1:i2] = ops.dot(w)
    
#ops = EM.operators(seq_active_possibles)
#energy = ops.dot(w)
#i = np.argmax(energy)
#seq_hidden_part = seq_all[i]

#-------------------------------------
## recover hidden
seq_hidden = seq1.copy()
seq_hidden[hidden] = 0.

seq_recover = seq1.copy()

cols_neg = np.loadtxt('cols_neg.txt').astype(int)
cols_pos = np.loadtxt('cols_pos.txt').astype(int)
hidden_neg = np.intersect1d(hidden_conserved,cols_neg)
hidden_pos = np.intersect1d(hidden_conserved,cols_pos)

seq_recover[hidden_neg] = -1.
seq_recover[hidden_pos] = 1.

# the best seq
seq_recover[hidden_active] = seq_all[np.argmax(energy)] 

# expectation values
seq_recover_av = seq_recover.copy()
energy_max = energy.max()
seq_recover_av[hidden_active] = (seq_all*np.exp(energy - energy_max)[:,np.newaxis]).sum(axis=0)/(np.exp(energy-energy_max)).sum()
print(seq_recover_av[hidden_active])

#=========================================================================================
np.savetxt('seq1_v2_%s.dat'%n_hidden,seq1,fmt='%i')
np.savetxt('seq_hidden_v2_%s.dat'%n_hidden,seq_hidden,fmt='%i')
np.savetxt('seq_recover_v2_%s.dat'%n_hidden,seq_recover,fmt='%i')
np.savetxt('seq_recover_av_v2_%s.dat'%n_hidden,seq_recover_av,fmt='%i')

#nx,ny = 3,1
#nfig = nx*ny
#fig, ax = plt.subplots(ny,nx,figsize=(nx*3.5,ny*2.8))
#ax[0].imshow(seq1[t].reshape(28,28),interpolation='nearest')
#ax[1].imshow(seq_hidden.reshape(28,28),interpolation='nearest')
#ax[2].imshow(seq_recover.reshape(28,28),interpolation='nearest')

#plt.tight_layout(h_pad=0.7, w_pad=1.5)
#plt.savefig('fig4_50_random.pdf', format='pdf', dpi=100)
