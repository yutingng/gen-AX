import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import scipy
import scipy.io as sio

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.autograd as autograd

import utils
import time
import pickle

path = 'nutrient17.p'
U0 = utils.load_data_0(path).float();
nsamples = 1459
U0 = U0*nsamples/(nsamples+1)

ndims = 17

labels = ["Energy", "Protein",  "Vit A - IU", "Vit A - RE", "Vit E", "Vit C", "Thiamin", "Riboflavin", "Niacin", "Vit B6", "Folate", "Vit B12", "Calcium", "Phosphorus", "Magnesium", "Iron", "Zinc"]

# code from : https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/wgan_gp/wgan_gp.py
def compute_gradient_penalty(D, real_samples, fake_samples):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = torch.Tensor(np.random.random((real_samples.size(0), 1)))
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates).unsqueeze(-1)
    fake = Variable(torch.Tensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

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
    
for iter_stdf in range(n_iter):
    
    data = U0[np.random.randint(0,nsamples,n_batch),:].detach()
    
    # train D
    optimizerD.zero_grad()

    z = torch.randn_like(data)

    fake_data = netG(z)

    real_d = netD(data)
    fake_d = netD(fake_data.detach())

    gp = compute_gradient_penalty(netD, data.data, fake_data.data)
    d_loss = -torch.mean(real_d) + torch.mean(fake_d) + gp

    d_loss.backward()
    optimizerD.step()

    if (iter_stdf % 2) == 0:

        # train G
        optimizerG.zero_grad()

        fake_data = netG(z)

        fake_d = netD(fake_data)
        g_loss = -torch.mean(fake_d)

        g_loss.backward()
        optimizerG.step()

    if iter_stdf % 100 == 0:

        plt.close("all")
        U = netG(torch.rand(nsamples,ndims))
        utils.plot_U1_U2(U,U0,labels=labels)
