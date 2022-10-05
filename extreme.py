import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.autograd import Function

import torch.distributions as tdist

import scipy
import scipy.stats

import numpy as np

from utils import rand_exp, rand_simplex, rand_positive_stable
from archimedean import GumbelPhi

class stdfNSD(nn.Module):
    def __init__(self, alpha=torch.tensor([1.,2.,3.,4.]), rho=torch.tensor(-0.59)):
        super(stdfNSD, self).__init__()
        self.ndims = alpha.shape[0]
        self.alpha = alpha.detach().clone()
        self.rho = rho.detach().clone()
        self.M = self.sample_M(10000)
        
    def sample_M(self, n_samples):
        alpha=self.alpha
        rho=self.rho
        gamma = tdist.gamma.Gamma(alpha, 1)
        D = gamma.sample((n_samples,))
        W = D**rho / (torch.lgamma(alpha+rho).exp() / torch.lgamma(alpha).exp())
        return W/W.mean(dim=0,keepdims=True)
    
    def sample(self, n_samples):
        ndims = self.ndims
        tries = 100

        P = 1./rand_exp(tries,n_samples).cumsum(axis=0)
        M = self.sample_M(tries*n_samples).view(tries,n_samples,ndims)
        U = torch.max(P[:,:,None]*M,dim=0)[0]
        U = torch.exp(-1./U)
        
        return U, M
    
    def forward(self,x):
        M = self.M
        ret = ((x[:,:,None].expand(-1,-1,1)*M.T[None,:,:].expand(1,-1,-1)).max(dim=1)[0]).mean(dim=1)
        return ret

class stdfPickands(nn.Module):
    def __init__(self, y, phi=GumbelPhi(torch.tensor(1.0))):
        super(stdfPickands, self).__init__()
        self.y = y.detach().clone()
        self.phi = phi
        self.ndims = y.shape[0]
    
    def forward(self,w):
        y = self.y; n = self.ndims
        phi = self.phi
        
        x = torch.min(phi.inverse(y.T[None,:,:].expand(1,-1,-1))/w[:,:,None].expand(-1,-1,1),dim=1)[0]
        u = phi.inverse(torch.arange(1,n+1,1)/(n+1)).mean()
        A = u/x.mean(dim=1) 
            
        return A
    
class stdfCFG(nn.Module):
    def __init__(self, y, phi=GumbelPhi(torch.tensor(1.0))):
        super(stdfCFG, self).__init__()
        self.y = y.detach().clone()
        self.phi = phi
        self.ndims = y.shape[0]
    
    def forward(self,w):
        y = self.y; n = self.ndims
        phi = self.phi
        
        x = torch.min(phi.inverse(y.T[None,:,:].expand(1,-1,-1))/w[:,:,None].expand(-1,-1,1),dim=1)[0]
        x[x<1e-15] = 1e-15
        v = torch.log(phi.inverse(torch.arange(1,n+1,1)/(n+1))).mean()
        A = v-torch.log(x).mean(dim=1) 
            
        return torch.exp(A)

    
class stdfSL(nn.Module):
    def __init__(self, ndims=2, theta=torch.tensor(3.0)):
        super(stdfSL, self).__init__()
        self.ndims = ndims
        # theta in 1 to infinity
        # alpha = 1/theta in 0 to 1 (sigmoid activation)
        self.alpha_ps = nn.Parameter(torch.log((1./theta)/(1.-(1./theta)))) 

    def sample(self, n_samples):
        alpha = torch.sigmoid(self.alpha_ps)
        ndims = self.ndims
        S = rand_positive_stable(alpha, n_samples, 1)
        W = rand_exp(n_samples, ndims)
        return torch.exp(-1./((S / W)**alpha)) # frechet margins

    def forward(self, w):
        alpha = torch.sigmoid(self.alpha_ps)
        return torch.sum(w**(1./alpha), dim=1)**alpha
    
    
class stdfASL(nn.Module):
    def __init__(self, alphas, thetas):
        super(stdfASL, self).__init__()
        
        # thetas >= 0, columns sum to 1 (softmax activation)
        # alphas = 1/r in 0 to 1 (sigmoid activation)
        
        self.ndims = thetas.shape[1]
        self.nmixs  = thetas.shape[0]
        assert thetas.shape[0] == alphas.shape[0]
        
        self.alphas_ps = nn.Parameter(torch.log(alphas/(1.-alphas)))
        self.thetas_ps = nn.Parameter(torch.log(thetas))

    def sample(self, n_samples):
        alphas = torch.sigmoid(self.alphas_ps)
        thetas = F.softmax(self.thetas_ps,dim=0)
        ndims = self.ndims
        nmixs = self.nmixs
        
        Sm = rand_positive_stable(alphas.view(1, -1), n_samples, nmixs)
        Wm = rand_exp(n_samples, nmixs, ndims)
        Xm = thetas[None,:,:].expand(n_samples,nmixs,ndims) \
             * (Sm[:,:,None].expand(n_samples,nmixs,ndims) / Wm) ** alphas[None,:,None].expand(n_samples,nmixs,ndims)
        return torch.exp(-1./Xm.max(dim=1)[0]) # frechet margins

    def forward(self, w):
        alphas = torch.sigmoid(self.alphas_ps)
        thetas = F.softmax(self.thetas_ps,dim=0)
        
        nquery = w.shape[0]
        ndims = self.ndims
        nmixs = self.nmixs
                
        wtheta = w[:,:,None].expand(nquery,ndims,nmixs)*thetas.T[None,:,:].expand(nquery,ndims,nmixs)
        ret_m = torch.sum(wtheta ** (1./alphas[None,None,:].expand(nquery,ndims,nmixs)), dim=1) ** \
                                        alphas[None,:].expand(nquery,nmixs)
        return torch.sum(ret_m, dim=1)

    
class stdfStochastic(nn.Module):
    def __init__(self, Nz=200, ndims=2):
        super(stdfStochastic, self).__init__()
        self.ndims = ndims
        self.model = nn.Sequential(
            nn.Linear(ndims, 30),
            nn.BatchNorm1d(30),
            nn.ReLU(),
            nn.Linear(30, 30),
            nn.BatchNorm1d(30),
            nn.ReLU(),
            nn.Linear(30, ndims),
            nn.Softmax()
        )
        self.M = self.sample_M(Nz)
        
    def resample_M(self, Nz):
        self.M = self.sample_M(Nz)
        
    def sample_M(self, N):
        return self.model(torch.rand(N,self.ndims))
        
    def forward(self, d):
         
        nquery, nmix, ndims = d.size()[0], self.M.size()[0], self.ndims
        
        ret = ndims*torch.mean(torch.max(d[:,None,:].expand(nquery,nmix,ndims)
                                   *self.M[None,:,:].expand(nquery,nmix,ndims),dim=2)[0],dim=1)
            
        return ret
    
    def sample(self,n_sample):

        ndims = self.ndims
        tries = 100
        
        P = 1./rand_exp(tries,n_sample).cumsum(axis=0)
        M = ndims*self.sample_M(tries*n_sample).view(tries,n_sample,ndims)
        U = torch.max(P[:,:,None]*M,dim=0)[0]
        
        return torch.exp(-1./U)
    

def cdfEVC(stdf):
    return lambda y : torch.exp(-stdf(-torch.log(y)))

def sdfGPC(stdf):
    return lambda y : F.relu(1.-stdf(y))**(y.shape[1]-1)