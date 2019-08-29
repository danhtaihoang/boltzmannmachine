##========================================================================================
import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
#=============================================================================== 
# June 29.2018: redefine sign function
# note: np.sign(0) = 0 but here sign(0) = 1
def sign(x): return 1. if x >= 0 else -1.
##==============================================================================
# June 12.2018: cross_cov
# a,b -->  <(a - <a>)(b - <b>)>   
##-------------------------------------------   
def cross_cov(a,b):
   da = a - np.mean(a, axis=0)
   db = b - np.mean(b, axis=0)
   return np.matmul(da.T,db)/a.shape[0]

##==============================================================================
def gen_w(n,nh): # now, only for nh=2    
    list_positive = [[7,12,17],[8,12,16],[11,12,13],[6,12,18],[6,8,11,13,16,18]]
    w = np.empty((n,nh))
    
    for ih in range(nh):
        w1 = -np.random.rand(n) # negative        
        for i in list_positive[ih]:
            w1[i] = -w1[i]        
        w[:,ih] = w1[:]    
        
    return w
#plt.imshow(w1.reshape(n1,n1))
    
##==============================================================================
# 2018.07.16: generate configuration of hidden sh as random
def gen_sh(l,nh):
    # initial sh
    sh = np.zeros((l,nh)) 
    for t in range(l):
        ih=np.random.randint(0,nh)
        sh[t,ih] = 1.
    
    return sh

#----------------------------------------------
# 2018.07.16: generate configuration of observed s based on w and hidden sh
def gen_s(w,h0,sh):
    n = int(w.shape[0])
    l = int(sh.shape[0])

    # s based on w and sh    
    h = h0 + np.matmul(sh,w.T)
    p = 1/(1+np.exp(-2*h))
    
    return np.sign(p-np.random.rand(l,n))

##==============================================================================
# 2018.07.16: average value of s acting by each hidden sh   
def s_av_based_hidden(s,sh):
    l,n = s.shape
    nh = sh.shape[1]    
    sp_av = np.empty((nh,n))
    
    for ih in range(nh):
        rows_hidden = [t for t in range(l) if sh[t,ih]==1]
        sp = s[rows_hidden]
        
        #print(sp.shape)
        sp_av[ih,:] = np.mean(sp,axis=0)
        
    return sp_av

##==============================================================================
# 2018.07.24: read MNIST data    
def read_mnist_data(file_name='mnist_train.csv'):    
    s0 = np.loadtxt(file_name,delimiter=',')
  
    s1 = s0[:,1:]        
    l,n = s1.shape
    s2 = np.full((l,n),-1.)
    #s2[s1>127.5] = 1.
    s2[s1>1.] = 1.

    #n1 = int(np.sqrt(n))

    # 2D to 3D
    #s = s2.reshape(l,n1,n1) 
    # resize
    #s = s[:,4:24,4:24]
    # 3D to 2D
    #s = s.reshape(s.shape[0],-1)
    
    return s2,s0[:,0]
##==============================================================================
# 2018.07.16: find W, H0 from s and sh
# here, only hidden effect to obs s, no interaction between obs to obs and obs to hidden    
def fit_interaction(s,sh,nloop):   
    n = s.shape[1]
    nh = sh.shape[1]
    l = s.shape[0]

    m = np.mean(sh,axis=0)
    ds = sh - m
    c = np.cov(ds,rowvar=False,bias=True)
    c_inv = linalg.pinv(c,rcond=1e-15)
    
    dst = ds.T

    # initial
    #w_ini = gen_w(n,nh)
    #w_ini = np.random.randn(n,nh)
    w_ini = np.random.normal(0.0,1.,size=(n,nh)) # initial w as Gaussian

    h_all = np.matmul(sh,w_ini.T)
    
    W = np.empty((n,nh)) ; H0 = np.empty(n)
    for i0 in range(n):
        s1=s[:,i0]
        cost = np.full(nloop,100.) 
        h = h_all[:,i0]
        for iloop in range(1,nloop):
            h_av = np.mean(h)
            hs_av = np.matmul(dst,h-h_av)/l
            w = np.matmul(hs_av,c_inv)
            h0=h_av-np.sum(w*m)
            h = np.matmul(sh,w.T) + h0
            s_model = np.tanh(h)
        
            cost[iloop]=np.mean((s1[:]-s_model[:])**2)
            #print(i0,cost[iloop])
            if cost[iloop] >= cost[iloop-1]:
                #print(i0,iloop)
                break
    
            #if s_model.any() != 0:
            #    h *= s1/s_model         
            
            # to avoid 0/0: 0/0 = 0 by this solution
            h *= np.divide(s1,s_model, out=np.zeros_like(s1), where=s_model!=0)
    
        W[i0,:] = w[:]
        H0[i0] = h0
 
    return W,H0

##==============================================================================
# 2018.07.24: including interaction between obs to obs and hidden to obs    
def fit_interaction_including_obs_obs(s,sh,nloop):       
    l,no = s.shape
    s = np.hstack((s,sh))
    n = s.shape[1]
    #nh = sh.shape[1]
    #l = s.shape[0]

    m = np.mean(s,axis=0)
    ds = s - m
    c = np.cov(ds,rowvar=False,bias=True)
    c_inv = linalg.pinv(c,rcond=1e-15)
    
    dst = ds.T

    # initial
    #w_ini = gen_w(n,nh)
    w_ini = np.random.randn(n,n)
    
    # no interaction to hidden = 0
    #w_ini[no:,:] = 0. 
    
    h_all = np.matmul(s,w_ini.T)
    
    W = np.empty((n,n)) ; H0 = np.empty(n)
    for i0 in range(no):
        s1=s[:,i0]
        cost = np.full(nloop,100.) 
        h = h_all[:,i0]
        for iloop in range(1,nloop):
            h_av = np.mean(h)
            hs_av = np.matmul(dst,h-h_av)/l
            w = np.matmul(hs_av,c_inv)
            
            w[i0] = 0. # w[i,i] = 0
            
            h0=h_av-np.sum(w*m)
            h = np.matmul(s,w.T) + h0
            s_model = np.tanh(h)
        
            cost[iloop]=np.mean((s1[:]-s_model[:])**2)
            #print(i0,cost[iloop])
            if cost[iloop] >= cost[iloop-1]: break     
            
            # to avoid 0/0: 0/0 = 0 by this solution
            h *= np.divide(s1,s_model, out=np.zeros_like(s1), where=s_model!=0)
    
        W[i0,:] = w[:]
        H0[i0] = h0
 
    return W,H0
##==============================================================================
# 2018.07.16: compre s_av and s_pred_av:    
def validate(s,sh0,w,h0,sh):
    n = s.shape[1]
    nh = sh.shape[1]
    
    s_av = s_av_based_hidden(s,sh0)
    s_pred = gen_s(w,h0,sh)
    sp_av = s_av_based_hidden(s_pred,sh)
    
    n1 = int(np.sqrt(n))
    plt.figure(figsize=(20,16))
    for ih in range(nh):
        plt.subplot(2,nh,ih+1)
        plt.imshow(s_av[ih,:].reshape(n1,n1))       
    for ih in range(nh):
        plt.subplot(2,nh,nh+ih+1)
        plt.imshow(sp_av[ih,:].reshape(n1,n1))       
    #plt.savefig('actual_predict.eps', format='eps', dpi=300)
    plt.show()

##==============================================================================
# 2018.07.16: compre s_av and s_pred_av:    
def validate_s(s,w,h0,sh):
    n = s.shape[1]
    nh = sh.shape[1]
    
    s_pred = gen_s(w,h0,sh)
    sp_av = s_av_based_hidden(s_pred,sh)
    
    n1 = int(np.sqrt(n))
    plt.figure(figsize=(20,16))     
    for ih in range(nh):
        plt.subplot(2,nh,ih+1)
        plt.imshow(sp_av[ih,:n1*n1].reshape(n1,n1))       
    plt.savefig('predict.eps', format='eps', dpi=300)
    plt.show()    
    
##==============================================================================
# 2018.07.16: classify_s based on sh config:    
def classify_s(s,sh,ncolumn,file_name):
    n = s.shape[1]
    nh = sh.shape[1]
    
    s_av = s_av_based_hidden(s,sh)
    
    n1 = int(np.sqrt(n))
    plt.figure(figsize=(20,16))

    nrow = nh/ncolumn    

    for irow in range(nrow):    
        for icolumn in range(ncolumn):
            ih = irow*ncolumn+icolumn
            plt.subplot(nrow,ncolumn,ih+1)
            plt.title('ih=%03d'%ih)
            plt.imshow(s_av[ih,:n1*n1].reshape(n1,n1))       
    plt.savefig('classify_s_%s.eps'%(file_name), format='eps', dpi=300)
    plt.show()       

##==============================================================================
# 2018.07.16: classify_s based on sh config:    
def classify_s_each_label_old(s,sh,ntype,ih_label,file_name):
    n = s.shape[1]
    #nh = sh.shape[1]
    
    s_av = s_av_based_hidden(s,sh)
    
    n1 = int(np.sqrt(n))
    plt.figure(figsize=(1.5*7,15))

    nlabel = 10
    ncol = int(max(ntype))
    for label in range(nlabel):    
        for itype in range(int(ntype[label])):
            ih = label*ncol+itype
            plt.subplot(nlabel,ncol,ih+1)
            #plt.title('ih=%03d'%ih)
            plt.axis('off')
            plt.imshow(s_av[int(ih_label[label,itype]),:n1*n1].reshape(n1,n1),interpolation='nearest')
            plt.subplots_adjust(hspace=0.05,wspace=0.05)
    plt.savefig('classify_s_%s.pdf'%(file_name), format='pdf', dpi=300)
    plt.show()  

##==============================================================================
## 2018.09.12: different color scheme for different label
def classify_s_each_label(s,sh,ntype,ih_label,file_name):
    n = s.shape[1]
    #nh = sh.shape[1]
    
    s_av = s_av_based_hidden(s,sh)
    
    n1 = int(np.sqrt(n))
    plt.figure(figsize=(1.5*7,15))

    nlabel = 10
    ncol = int(max(ntype))
    for label in range(nlabel):    
        for itype in range(int(ntype[label])):
            ih = label*ncol+itype
            plt.subplot(nlabel,ncol,ih+1)
            #plt.title('ih=%03d'%ih)
            plt.axis('off')
            plt.imshow(s_av[int(ih_label[label,itype]),:n1*n1].reshape(n1,n1),interpolation='nearest')
            plt.colorbar()
            plt.subplots_adjust(hspace=0.05,wspace=0.05)
    plt.savefig('classify_s_%s.pdf'%(file_name), format='pdf', dpi=300)
    plt.show()

##==============================================================================
# 2018.07.16: update sh then find the final sh,w,h0 based on s and nh
def predict_interactions(s,nh,nupdate_hidden,nloop):
    l,n = s.shape    
    cost = np.empty(nupdate_hidden)
    L_update = np.empty(nupdate_hidden)
    ih0_list = np.empty(l)
    
    sh = gen_sh(l,nh) # initial
    for iupdate in range(nupdate_hidden):    
        w,h0 = fit_interaction(s,sh,nloop)
        h = h0+w.T  # h[ih,i] = h0[i] +w[i,ih].T

        #stanhh = s[:,:,np.newaxis] - np.tanh(h.T[np.newaxis,:,:])
        #cost[iupdate] = np.mean(stanhh**2,axis=1) 

        #for t in range(l):
        #    for ih in range(nh):
        #        L[t,ih] = 1 - np.mean(np.log(1.+np.exp(-2*h[ih,:]*s[t,:])))
               
        L = -np.mean(np.log(1+np.exp(-2*s[:,:,np.newaxis]*h.T[np.newaxis,:,:])),axis=1)
               
        L_all = 0. 
        sh = np.zeros((l,nh))  
        for t in range(l):
            ih0 = np.argmax(L[t,:])
            sh[t,ih0] = 1.    
            L_all += L[t,ih0]
            ih0_list[t] = ih0
            
        L_update[iupdate] = L_all/l
                
        h = h0 + np.matmul(sh,w.T)
        cost[iupdate] = np.mean((s - np.tanh(h))**2)
                
        print(iupdate,cost[iupdate],L_update[iupdate])    
                           
    return w,h0,ih0_list,cost,L_update    

##==============================================================================
# 2018.07.28: update sh then find the final sh,w,h0 based on s and nh
# sh is updated based on Monte Carlo
def predict_interactions_MC(s,nh,nupdate_hidden,nloop):
    l,n = s.shape    
    cost = np.empty(nupdate_hidden)
    L_update = np.empty(nupdate_hidden)
    ih0_list = np.empty(l)
    
    sh = gen_sh(l,nh) # initial
    for iupdate in range(nupdate_hidden):    
        w,h0 = fit_interaction(s,sh,nloop)
        h = h0+w.T  # h[ih,i] = h0[i] +w[i,ih].T

        #stanhh = s[:,:,np.newaxis] - np.tanh(h.T[np.newaxis,:,:])
        #cost[iupdate] = np.mean(stanhh**2,axis=1) 

        #for t in range(l):
        #    for ih in range(nh):
        #        L[t,ih] = 1 - np.mean(np.log(1.+np.exp(-2*h[ih,:]*s[t,:])))
               
        L = -np.mean(np.log(1+np.exp(-2*s[:,:,np.newaxis]*h.T[np.newaxis,:,:])),axis=1)
             
        #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        # update hidden
        L_all = 0. 
        #sh = np.zeros((l,nh))
        
        L_dev = L.std(axis=1)
        for t in range(l):
            #ih0 = np.argmax(L[t,:])
                        
            ih = np.argmax(sh[t,:])
            
            # choice another ih0 != ih
            ih0=np.random.choice(range(0,ih)+range(ih+1,nh))
                         
            if np.exp((L[t,ih0]-L[t,ih])/L_dev[t]) > np.random.rand():
                sh[t,ih] = 0.
                sh[t,ih0] = 1.
                ih = ih0
                
            L_all += L[t,ih]
            ih0_list[t] = ih
        
        #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        L_update[iupdate] = L_all/l
                
        h = h0 + np.matmul(sh,w.T)
        cost[iupdate] = np.mean((s - np.tanh(h))**2)
                
        print(iupdate,cost[iupdate],L_update[iupdate])    
                           
    return w,h0,ih0_list,cost,L_update 
