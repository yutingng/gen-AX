#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.io as sio

import torch as torch
import torch.nn as nn
import torch.nn.functional as F
import utils, archimedean, extreme, importlib
import importlib

import time
import pickle

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Computer Modern Sans Serif"],
    "font.size":22})
# for Palatino and other serif fonts use:
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "font.size":22})


# In[ ]:


nsamples = 240 # monthly maxima
ndims = 3

path = './data/synthetic/CNSD_C_1_14.p'
U0 = utils.load_data_0(path).float();
 
labels = ["Belle-Ile", "Groix", "Lorient"]
utils.plot_U(U0, labels=labels)

stdf = extreme.stdfNSD(alpha = torch.tensor([1.,2.,3.]), rho = torch.tensor([-0.69]))


# In[ ]:


# block maximas

U0_6 = U0[np.random.permutation(nsamples),:].reshape(6,-1,ndims).max(axis=0)[0].detach().numpy()
for i in range(ndims):
    U0_6[:, i] = scipy.stats.rankdata(U0_6[:, i], 'ordinal')/40
U0_6 = torch.tensor(U0_6)


# In[ ]:


utils.plot_U1_U2(U0_6, U0)


# In[ ]:


phi = archimedean.GumbelPhi(torch.tensor(1.0))


# In[ ]:


nan_count = 0
stdf_GNN = extreme.stdfStochastic(ndims=ndims)
opt_stdf = torch.optim.Adam(stdf_GNN.parameters(), lr=1e-3)

MSEloss = torch.nn.MSELoss()
onesd = torch.ones(stdf_GNN.ndims)/stdf_GNN.ndims

n_iter = 5100
n_z = 200
n_batch = 40
n_test = 10000
time_taken = 0

for iter_stdf in range(n_iter):
    
    U0_6 = U0[np.random.permutation(nsamples),:].reshape(6,-1,ndims).max(axis=0)[0].detach().numpy()
    for i in range(ndims):
        U0_6[:, i] = scipy.stats.rankdata(U0_6[:, i], 'ordinal')/40
    U0_6 = torch.tensor(U0_6)
    
    time_start = time.time()
    opt_stdf.zero_grad()
    stdf_GNN.resample_M(n_z)
    y = U0_6.detach()
    w = utils.rand_simplex(n_batch,ndims)
    
    x = torch.min(phi.inverse(y[:,:,None].expand(-1,-1,1))/w.T[None,:,:].expand(1,-1,-1),dim=1)[0]
    A = stdf_GNN(w)[None,:].expand(n_batch,-1)
    ll = torch.log(-phi.diff(x*A))+torch.log(A)
    lloss = -torch.mean(ll)
    scaleloss = MSEloss(stdf_GNN.M.mean(dim=0),onesd)
    regloss = lloss + scaleloss
    regloss.backward()
    opt_stdf.step()
    time_end = time.time()
    time_taken += (time_end-time_start)
    
    if iter_stdf%100 == 0:
        
        print(iter_stdf, time_taken, lloss.item(), scaleloss.item())
        
        if torch.isnan(stdf_GNN.sample(1)).sum()>1: 
            ckpt = torch.load(prev_ckpt)
            stdf_GNN.load_state_dict(ckpt['model_state_dict'])
            opt_stdf.load_state_dict(ckpt['optimizer_state_dict'])
            nan_count+= 1
            
            print("nan_count:%d"%nan_count)

            if nan_count == 3:
                break
                
        torch.save({
                'iter_stdf': iter_stdf,
                'model_state_dict': stdf_GNN.state_dict(),
                'optimizer_state_dict': opt_stdf.state_dict(),
            }, './checkpoints/ckpt_EE_1exp_i%d.ckpt'%(iter_stdf))
        
        prev_ckpt = './checkpoints/ckpt_EE_1exp_i%d.ckpt'%(iter_stdf)


# In[ ]:


Uev = stdf_GNN.sample(200000).view(100,2000,ndims).max(dim=0)[0].detach().numpy()
for i in range(ndims):
    Uev[:, i] = scipy.stats.rankdata(Uev[:, i], 'ordinal')/2000
utils.plot_stdf3(extreme.stdfCFG(torch.tensor(Uev)))

U_test = utils.rand_simplex(10000,ndims)

stdfmax = extreme.stdfCFG(torch.tensor(Uev))(U_test)
stdfGT = stdf(U_test)
IRAEloss = torch.mean(torch.abs(stdfmax-stdfGT)/stdfGT).item()
print(IRAEloss)


# In[ ]:


Uev0, W = stdf.sample(2000)
utils.plot_U1_U2(torch.tensor(Uev), Uev0, labels=labels)

