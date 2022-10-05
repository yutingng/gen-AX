#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.io as sio

import torch as torch
import torch.nn as nn
import torch.nn.functional as F
import utils, archimedean, extreme
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


# In[2]:


# load data

path = './data/realworld/nutrient17.p'
U0 = utils.load_data_0(path);
nsamples = 1459
U0 = U0*nsamples/(nsamples+1)

ndims=17

labels = ["Energy", "Protein",  "Vit A - IU", "Vit A - RE", "Vit E", "Vit C", "Thiamin", "Riboflavin", "Niacin", "Vit B6", "Folate", "Vit B12", "Calcium", "Phosphorus", "Magnesium", "Iron", "Zinc"]


# In[3]:


stdfCFG = extreme.stdfCFG(U0)
stdfPickands = extreme.stdfPickands(U0)


# In[4]:


phi = archimedean.GumbelPhi(torch.tensor(1.0))


# In[5]:


nan_count = 0
stdf_GNN = extreme.stdfStochastic(ndims=ndims)
opt_stdf = torch.optim.Adam(stdf_GNN.parameters(), lr=1e-3)

MSEloss = torch.nn.MSELoss()
onesd = torch.ones(stdf_GNN.ndims)/stdf_GNN.ndims

n_iter = 5100
n_z = 200
n_batch = 200

for iter_stdf in range(n_iter):
    
    opt_stdf.zero_grad()
    stdf_GNN.resample_M(n_z)
    y = U0[np.random.randint(0,nsamples,n_batch),:].detach()
    w = utils.rand_simplex(n_batch,ndims)
    
    x = torch.min(phi.inverse(y[:,:,None].expand(-1,-1,1))/w.T[None,:,:].expand(1,-1,-1),dim=1)[0]
    A = stdf_GNN(w)[None,:].expand(n_batch,-1)
    ll = torch.log(-phi.diff(x*A))+torch.log(A)
    lloss = -torch.mean(ll)
    scaleloss = MSEloss(stdf_GNN.M.mean(dim=0),onesd)
    regloss = lloss + scaleloss
    regloss.backward()
    opt_stdf.step()
    
    if iter_stdf%100 == 0:
        
        print(iter_stdf)
        
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
            }, './checkpoints/ckpt_1exp_i%d.ckpt'%(iter_stdf))
        
        prev_ckpt = './checkpoints/ckpt_1exp_i%d.ckpt'%(iter_stdf)


# In[6]:


U = phi(-torch.log(stdf_GNN.sample(nsamples))/phi.sample_M(nsamples).reshape(-1,1))
utils.plot_U1_U2(U,U0,labels=labels) #U1 in blue, U2 in black

