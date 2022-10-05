#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.io as sio

import torch
import torch.nn as nn

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


# In[2]:


nsamples = 240 # monthly maxima
ndims = 3

path = './data/synthetic/CNSD_C_1_14.p'
U0 = utils.load_data_0(path).float();
 
labels = ["Belle-Ile", "Groix", "Lorient"]
utils.plot_U(U0, labels=labels)

stdf = extreme.stdfNSD(alpha = torch.tensor([1.,2.,3.]), rho = torch.tensor([-0.69]))


# In[3]:


class Generator(nn.Module):
    def __init__(self, ndims):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(ndims, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, ndims),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.model(input)
    
class Discriminator(nn.Module):
    def __init__(self, ndims):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(ndims, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.model(input).view(-1, 1).squeeze(1)


# In[4]:


netG = Generator(ndims).float()
netD = Discriminator(ndims).float()
criterion = nn.BCELoss()

real_label = 1.0;
fake_label = 0.0
n_batch = 200
n_iter = 5000
lr=1e-3

real_labels = torch.full((n_batch,),real_label)
fake_labels = torch.full((n_batch,),fake_label)

optimizerD = torch.optim.Adam(netD.parameters(), lr=lr)
optimizerG = torch.optim.Adam(netG.parameters(), lr=lr)

time_array = np.zeros(int(n_iter/100))
iter_array = np.zeros(int(n_iter/100))
time_taken = 0

errD_real_array = np.zeros(n_iter)
errD_fake_array = np.zeros(n_iter)
errG_array = np.zeros(n_iter)

    
for iter_stdf in range(n_iter):
    
    data = U0[np.random.randint(0,nsamples,n_batch),:].detach()
    
    time_start = time.time()
    ## train discriminator

    # train with real    
    netD.zero_grad()
    output = netD(data)
    errD_real = criterion(output, real_labels.detach())
    errD_real.backward()

    # train with fake
    noise = torch.rand(n_batch,ndims)
    fake = netG(noise)
    output = netD(fake.detach())
    errD_fake = criterion(output, fake_labels.detach())
    errD_fake.backward()
    errD = errD_real + errD_fake
    optimizerD.step()

    ## train generator

    netG.zero_grad()
    output = netD(fake)
    errG = criterion(output, real_labels.detach())
    errG.backward()
    optimizerG.step()

    time_end = time.time()
    time_taken += (time_end-time_start)

    errD_real_array[iter_stdf] = errD_real.item()
    errD_fake_array[iter_stdf] = errD_fake.item()
    errG_array[iter_stdf] = errG.item()

    if iter_stdf % 100 == 0:

        time_array[int(iter_stdf/100)] = time_taken
        iter_array[int(iter_stdf/100)] = iter_stdf
        print(iter_stdf, time_taken, errD_real.item(), errD_fake.item(), errG.item())


# In[5]:


# block maximas and compute IRAE

IRAEloss_array = np.zeros(10)
for n in range(10):
    noise = torch.rand(200000,ndims)
    fake = netG(noise)

    Uev = fake.view(100,2000,ndims).max(dim=0)[0].detach().numpy()
    for i in range(ndims):
        Uev[:, i] = scipy.stats.rankdata(Uev[:, i], 'ordinal')/2000

    U_test = utils.rand_simplex(10000,ndims)

    stdfmax = extreme.stdfCFG(torch.tensor(Uev))(U_test)
    stdfGT = stdf(U_test)
    IRAEloss_array[n] = torch.mean(torch.abs(stdfmax-stdfGT)/stdfGT).item()
    print(IRAEloss_array[n])
print(IRAEloss_array.mean(),IRAEloss_array.std())


# In[6]:


utils.plot_stdf3(extreme.stdfCFG(torch.tensor(Uev)))

