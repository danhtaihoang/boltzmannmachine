##==============================================================================
import numpy as np
import sys
#import timeit
#from scipy import linalg
#import matplotlib.pyplot as plt

#from myfunction_mnist import gen_w
#from myfunction_mnist import gen_sh
#from myfunction_mnist import gen_s
#from myfunction_mnist import fit_interaction
#from myfunction_mnist import s_av_based_hidden
from myfunction_mnist import read_mnist_data
from myfunction_mnist import predict_interactions
#from myfunction_mnist import predict_interactions_MC
#from myfunction_mnist import validate
#from myfunction_mnist import validate_s
#from myfunction_mnist import classify_s
#from myfunction_mnist import fit_interaction_including_obs_obs

#seed = 1

l = 60000
nloop = 300
nupdate_hidden = 200

#nlabel = 10
#nh = 4

#label = sys.argv[1]
#label = int(label)

nh = sys.argv[1]
nh = int(nh)

seed = sys.argv[2]
seed = int(seed)
#seed = 1

np.random.seed(seed)

print(nh,seed)      

##========================================================================================
# read training data   
# from personal PC
#s,slabel = read_mnist_data(file_name='/home/tai/MNIST_data/mnist_train.csv') 
# from biowulf
s,slabel = read_mnist_data(file_name='/data/hoangd2/MNIST_data/mnist_train.csv') 

n = s.shape[1]
s = s[:l,:]
slabel = slabel[:l]

#----------------------------------------------------------------
# ignore positions that shows more than 95% conservation
frequency = [(s[:,i] == -1).sum()/float(l) for i in range(n)]
cols = [i for i in range(n) if 0.05 < frequency[i] and frequency[i] < 0.95]
s = s[:,cols]
n = s.shape[1]
np.savetxt('cols_select.txt',cols,fmt='% 4.0f',newline='')
#----------------------------------------------------------------
# working with each digit independently
#s = s[slabel == label]
#l = s.shape[0]

#start_time = timeit.default_timer()

# supervised learning (label is known --> sh is known)
#sh = np.zeros((l,nh))
#for t in range(l):
#    sh[t,int(slabel[t])] = 1.
#w,h0 = fit_interaction(s,sh,nloop)

# unsepervised learning (label is unknown)
w,h0,ih0,cost,L = predict_interactions(s,nh,nupdate_hidden,nloop)
#w,h0,ih0,cost,L = predict_interactions_MC(s,nh,nupdate_hidden,nloop)

#ext_name = '%02d_%03d_%03d.dat'%(label,nh,seed)
ext_name = '%03d_%03d.dat'%(nh,seed)
np.savetxt('W/w_%s'%ext_name,w,fmt='% 12.8f')
np.savetxt('W/h0_%s'%ext_name,h0,fmt='% 12.8f')
np.savetxt('W/ih0_%s'%ext_name,ih0,fmt='% 3.0f',newline='')
np.savetxt('cost/cost_%s'%ext_name,zip(cost,L),fmt='% 15.10f')



    

