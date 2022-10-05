#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.io as sio

import torch
import torch.nn as nn
import torch.nn.functional as F

import utils, archimedean, extreme, importlib
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

# https://github.com/bayesiains/nflows

from nflows.flows.base import Flow
from nflows.distributions.normal import StandardNormal
from nflows.transforms.base import CompositeTransform
from nflows.transforms.autoregressive import MaskedAffineAutoregressiveTransform
from nflows.transforms.permutations import ReversePermutation


# In[2]:


nsamples = 240 # monthly maxima
ndims = 3

path = './data/synthetic/CNSD_C_1_14.p'
U0 = utils.load_data_0(path).float();
 
labels = ["Belle-Ile", "Groix", "Lorient"]
utils.plot_U(U0, labels=labels)

stdf = extreme.stdfNSD(alpha = torch.tensor([1.,2.,3.]), rho = torch.tensor([-0.69]))


# In[3]:


num_layers = 2
base_dist = StandardNormal(shape=[ndims])

transforms = []
for _ in range(num_layers):
    transforms.append(ReversePermutation(features=ndims))
    transforms.append(MaskedAffineAutoregressiveTransform(features=ndims, hidden_features=128))
transform = CompositeTransform(transforms)


# In[4]:


flow = Flow(transform, base_dist)
optimizer = torch.optim.Adam(flow.parameters(), lr = 1e-3)

n_batch = 200
n_iter = 5100

loss_array = np.zeros(n_iter)
time_array = np.zeros(int(n_iter/100))
iter_array = np.zeros(int(n_iter/100))
time_taken = 0
    
for iter_stdf in range(n_iter):
    
    data = U0[np.random.randint(0,nsamples,n_batch),:].detach()
    
    time_start = time.time()
    
    optimizer.zero_grad()
    loss = -flow.log_prob(inputs=data).mean()
    loss.backward()
    optimizer.step()

    time_end = time.time()
    time_taken += (time_end-time_start)

    loss_array[iter_stdf] = loss.item()
    
    if iter_stdf % 100 == 0:

        time_array[int(iter_stdf/100)] = time_taken
        iter_array[int(iter_stdf/100)] = iter_stdf
        print(iter_stdf, time_taken, loss.item())


# In[5]:


# block maximas and compute IRAE

fake = flow.sample(200000)

Uev = fake.view(100,2000,ndims).max(dim=0)[0].detach().numpy()
for i in range(ndims):
    Uev[:, i] = scipy.stats.rankdata(Uev[:, i], 'ordinal')/2000
utils.plot_stdf3(extreme.stdfCFG(torch.tensor(Uev)))

U_test = utils.rand_simplex(10000,ndims)

stdfmax = extreme.stdfCFG(torch.tensor(Uev))(U_test)
stdfGT = stdf(U_test)
IRAEloss = torch.mean(torch.abs(stdfmax-stdfGT)/stdfGT).item()
print(IRAEloss)

