'''
Contains common Archimedean generators (Clayton, Frank, Joe, Gumbel) and neural Archimedean generators.
Resources:
(i) ACNet, source code: https://github.com/lingchunkai/ACNet/blob/main/phi_listing.py
(ii) HACopula toolbox, source code: https://github.com/gorecki/HACopula
(iii) gen-AC, source code: https://github.com/yutingng/gen-AC
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from utils import newton_root, bisection_root

class PhiDM(nn.Module):
    def __init__(self, ndims):
        super(PhiDM, self).__init__()
        self.ndims = ndims
        self.r = None
        self.p_r = None
        self.F_r = None
        self.inverse = PhiInv(self)
        
    def sample(self, ndims, n):
        # McNeil, Neslehova 2010 ``d-monotone'' 
        shape = (n, ndims)
        if hasattr(self, 'sample_R'):
            R = self.sample_R(n)[:, None].expand(-1,ndims)
            e = torch.distributions.exponential.Exponential(torch.ones(shape))
            E = e.sample()
            E = E/(E.sum(dim=1).view(-1,1))
            return self.forward(R*E)
        else:
            print("sample_R not yet implemented")
            return torch.ones(shape)*float('nan')
        
    def sample_R(self, n):
        if hasattr(self,'r') and hasattr(self,'F_r'):
            
            n_r = len(self.F_r)
            idx = torch.where(torch.rand(n).reshape(-1,1)<self.F_r.reshape(1,-1),torch.zeros([n,n_r]),torch.ones([n,n_r]))

            return self.r[idx.sum(axis=1).long()]
        
    def forward(self, x):
        
        s = x.size()
        x_ = x.flatten()
        ndims = self.ndims
        
        nquery, nmix = x.numel(), self.r.numel()
        
        A = (1-x_[:, None].expand(nquery,nmix)/self.r[None, :].expand(nquery,nmix))
        A[A<=0]=0
        ret = torch.sum(A**(ndims-1)*self.p_r[None,:].expand(nquery,nmix),dim=1)        
        
        return ret.reshape(s)
    
    def ndiff(self, x, ndiff=1):
        
        s = x.size()
        x_ = x.flatten()
        ndims = self.ndims
        
        nquery, nmix = x.numel(), self.r.numel()
        
        A = (1-x_[:, None].expand(nquery,nmix)/self.r[None, :].expand(nquery,nmix))
        A[A<=0]=0
        A = A**(ndims-2)*(ndims-1)*(-1./self.r[None, :].expand(nquery,nmix))
        ret = torch.sum(A*self.p_r[None,:].expand(nquery,nmix),dim=1)
        
        return ret.reshape(s)
        
    def est_np2d(self, data):
        
        with torch.no_grad():
            
            ndims = data[[0],:].numel()
            assert ndims == 2
            
            zs = data[:,[0]].numel()
            c = torch.zeros(zs)
            for j in range(zs):
                c[j] = torch.sum(torch.prod(data<data[j,:],axis=1).double())/(zs+1)
                
            W = np.sort(c.detach().numpy())
            K = np.linspace(1/zs,1,zs)
            indices = np.where(np.diff(np.hstack([-1,W]))!=0)[0].astype(int)
            w = W[indices]
            k = K[indices]
            
            m = len(w)
            k = np.hstack([k[0],np.diff(k)])
            p = k[::-1]
            
            w = np.hstack([0,w])
            p = np.hstack([0,p])
            v = np.hstack([1,np.ones(m)])
            s = np.hstack([0,np.zeros(m)])
            for j in range(1,m):
                x = (np.sum(p[(m-j+1):(m+1)])-w[j+1])/ np.sum(v[1:(j+1)]*(p[(m-j+1):(m+1)][::-1]))
                v[1:(j+1)] *= x
                s[m-j] = x
            r = np.hstack([1,np.ones(m)])
            for j in range(2,(m+1)):
                r[j] = r[j-1]/s[j-1]
            
            self.r = r
            self.p = p
            self.Fr = np.cumsum(p)

class PhiDMStochastic(PhiDM):

    def __init__(self, ndims, Nz=100):
        super(PhiDMStochastic, self).__init__(ndims)
        self.model = nn.Sequential(
            nn.Linear(1, 10),
            nn.BatchNorm1d(10),
            nn.ReLU(),
            nn.Linear(10, 10),
            nn.BatchNorm1d(10),
            nn.ReLU(),
            nn.Linear(10, 1)
        )
        self.resample_R(Nz)
        self.ndims = ndims
        self.inverse = PhiInv(self)

    def resample_R(self, Nz):
        self.r = self.sample_R(Nz)
        self.p_r = torch.ones(Nz)/Nz
        
    def sample_R(self, N):
        return torch.exp(self.model(torch.rand(N).view(-1,1)).view(-1))


class PhiLT(nn.Module):
    def __init__(self):
        super(PhiLT, self).__init__()

    def sample(self, ndims, n):
        # Marshall-Olkin 1988 ``Families of Multivariate Distributions''
        shape = (n, ndims)
        if hasattr(self, 'sample_M'):
            M = self.sample_M(n)[:, None].expand(-1, ndims)
            e = torch.distributions.exponential.Exponential(torch.ones(shape))
            E = e.sample()
            return self.forward(E/M)
        else:
            print("sample_M not yet implemented")
            return torch.ones(shape)*float('nan')       


class PhiInv(nn.Module):
    def __init__(self, phi):
        super(PhiInv, self).__init__()
        self.phi = phi

    def forward(self, y, t0=None, max_iter=400, tol=1e-6):
        with torch.no_grad():
            t = newton_root(self.phi, y, max_iter=max_iter, tol=tol)
            #t = bisection_root(self.phi, y, ub = (torch.ones_like(y)*torch.max(self.phi.r)).detach())

        topt = t.detach().clone().requires_grad_(True)
        # clone() = make a copy where gradients flow back to original 
        # clone().detach() = prevent gradient flow back to original
        # detach().clone() = computationally more efficient
        # requires_grad_(True) this new tensor requires gradient
        
        f_topt = self.phi(topt) # approximately equal to y
        return self.FastInverse.apply(y, topt, f_topt, self.phi)

    class FastInverse(torch.autograd.Function):
        '''
        forward: avoid running newton_root repeatedly.
        backward: specify gradients of PhiInv to PyTorch.

        In the backward pass, we provide gradients w.r.t 
        (i) `y`, and (ii) `w` via `f_topt=self.phi(topt)` approx equal to y,
        i.e., the function evaluated (with the current `w`) on topt. 
        Note that this should contain *values* approximately equal to y, 
        #but will have the necessary computational graph built up,
        #(beginning with topt.requires_grad_(True) and f_topt = self.phi(topt))
        #but detached from y, i.e. unrelated to y.
        
        https://pytorch.org/tutorials/beginner/examples_autograd/two_layer_net_custom_function.html
        '''
        @staticmethod
        def forward(ctx, y, topt, f_topt, phi):
            ctx.save_for_backward(y, topt, f_topt)
            ctx.phi = phi
            return topt

        @staticmethod
        def backward(ctx, grad):
            y, topt, f_topt = ctx.saved_tensors
            phi = ctx.phi

            with torch.enable_grad():
                # Call FakeInverse once again, 
                # to allow for higher order derivatives.
                z = PhiInv.FastInverse.apply(y, topt, f_topt, phi)

                # Find phi'(z), i.e., take derivatives of phi(z) w.r.t z.
                if hasattr(phi,'ndiff'):
                    dev_z = phi.ndiff(z,ndiff=1)
                else:
                    f = phi(z)
                    dev_z = torch.autograd.grad(f.sum(), z, create_graph=True)[0]
                    print("warning: autodiff")

                # Refer to derivations for inverses.
                # Note when taking derivatives w.r.t. `w`, we make use of 
                # autograd's automatic application of the chain rule.
                # autograd finds the derivative d/dw[phi(z)], 
                # which when multiplied by the 3rd returned value,
                # gives the derivative d/dw[phi^{-1}].
                # Note that `w` is that contained by phi at f_topt.
                
                # what about gradients on "y" that is nested in phi_inverse?
                
                return grad/dev_z, None, -grad/dev_z, None  
        
        
class ClaytonPhi(PhiLT):
    def __init__(self, theta=torch.tensor(2.0)):
        super(ClaytonPhi, self).__init__()

        self.theta = nn.Parameter(theta)

    def forward(self, t):
        theta = self.theta
        ret = (1+t)**(-1./theta)
        #ret = (1+theta*t)**(-1./theta)
        return ret
    
    def inverse(self, u):
        theta = self.theta
        ret = u**(-theta)-1
        #ret = (u**(-theta)-1)/theta
        return ret
    
    def diff(self,t):
        theta = self.theta
        ret = (-1/theta)*(1+t)**(-1/theta-1.0)
        #ret = (-1)*(1+theta*t)**(-1/theta-1.0)
        return ret

    def sample_M(self, n):
        m = torch.distributions.gamma.Gamma(1./self.theta, 1.0)
        return m.sample((n,))

    def pdf(self, X):
        """ 
        Differentiate CDF
        [From Wolfram]
        d/dx((d(x^(-z) + y^(-z) - 1)^(-1/z))/(dy)) = (-1/z - 1) z (-x^(-z - 1)) y^(-z - 1) (x^(-z) + y^(-z) - 1)^(-1/z - 2)
        """
        assert X.shape[1] == 2

        Z = X[:, 0]**(-self.theta) + X[:, 1]**(-self.theta) - 1.
        ret = torch.zeros_like(Z)
        ret[Z > 0] = (-1/self.theta-1.) * self.theta * -X[Z > 0, 0] ** (-self.theta-1) * X[Z > 0, 1] ** (
            -self.theta-1) * (X[Z > 0, 0] ** (-self.theta) + X[Z > 0, 1] ** (-self.theta) - 1) ** (-1./self.theta-2)

        return ret

    def cdf(self, X):
        assert X.shape[1] == 2

        return (torch.max(X[:, 0]**(-self.theta) + X[:, 1]**(-self.theta) - 1, torch.zeros(X.shape[0])))**(-1./self.theta)    
    
    
class FrankPhi(PhiLT):
    def __init__(self, theta):
        super(FrankPhi, self).__init__()

        self.theta = nn.Parameter(theta)

    def forward(self, t):
        theta = self.theta
        ret = -1/theta * torch.log(torch.exp(-t)*(torch.exp(-theta)-1)+1)
        return ret
    
    def diff(self,t):
        theta = self.theta
        ret = (torch.exp(-t)*(torch.exp(-theta)-1))/(theta * (torch.exp(-t)*(torch.exp(-theta)-1)+1))
        return ret
    
    def inverse(self, u):
        theta = self.theta
        ret = -torch.log((torch.exp(-theta*u)-1)/(torch.exp(-theta)-1))
        return ret

    def pdf(self, X):
        return None

    def cdf(self, X):
        return -1./self.theta * \
            torch.log(
                1 + (torch.exp(-self.theta * X[:, 0]) - 1) * (
                    torch.exp(-self.theta * X[:, 1]) - 1) / (torch.exp(-self.theta) - 1))


class JoePhi(PhiLT):
    """
    The Joe Generator has a derivative that goes to infinity at t = 0. 
    Hence we need to be careful when t is close to 0!
    """

    def __init__(self, theta):
        super(JoePhi, self).__init__()

        self.eps = 0
        self.eps_snap_zero = 1e-15
        self.theta = nn.Parameter(theta)

    def forward(self, t):
        eps = self.eps
        if torch.any(t < eps):
            t_ = t + eps
            
        theta = self.theta
        ret = 1-(1-torch.exp(-t))**(1/theta)
        return ret
    
    def diff(self,t):
        eps = self.eps
        if torch.any(t < eps):
            t_ = t + eps
        theta = self.theta
        ret = -torch.exp(-t)/theta*(1-torch.exp(-t))**(1/theta-1)
        return ret
    
    def inverse(self,t):
        theta = self.theta
        x = 1-torch.pow(1-t,theta)
        eps = self.eps
        if torch.any(t < eps):
            t_ = eps
        return -torch.log(x)

    def sample_M(self, n):
        U = torch.rand(n)
        ret = torch.ones_like(U)

        ginv_u = self.Ginv(U)
        cond = self.F(torch.floor(ginv_u))

        cut_indices = U <= (1./self.theta)
        z = cond < U
        j = cond >= U

        ret[z] = torch.ceil(ginv_u[z])
        ret[j] = torch.floor(ginv_u[j])
        ret[cut_indices] = 1.

        return ret

    def Ginv(self, y):
        return torch.exp(-self.theta * (torch.log(1.-y) + torch.lgamma(1.-1/self.theta)))

    def gamma(self, x):
        return torch.exp(torch.lgamma(x))

    def lbeta(self, x, y):
        return torch.lgamma(x) + torch.lgamma(y) - torch.lgamma(x+y)

    def F(self, n):
        return 1. - 1. / (n * torch.exp(self.lbeta(n, 1.-1/self.theta)))

    def pdf(self, X):
        assert X.shape[1] == 2

        X_ = -X+1.0
        X_1 = X_[:, 0]
        X_2 = X_[:, 1]

        ret = -X_1 ** (self.theta-1) * X_2 ** (self.theta-1) * \
            ((X_1**self.theta) - (X_1**self.theta - 1) * X_2**self.theta)**(1./self.theta-2) * \
            ((X_1**self.theta-1) * (X_2**self.theta-1) - self.theta)

        return ret

    def cdf(self, X):
        assert X.shape[1] == 2

        X_ = -X+1.0
        X_1 = X_[:, 0]
        X_2 = X_[:, 1]

        return 1.0 - (X_1**self.theta + X_2**self.theta - (X_1**self.theta)*(X_2**self.theta))**(1./self.theta)


class GumbelPhi(PhiLT):
    def __init__(self, theta):
        super(GumbelPhi, self).__init__()

        self.theta = nn.Parameter(theta)

    def forward(self, t):
        theta = self.theta
        ret = torch.exp(-((t) ** (1/theta)))
        return ret
    
    def inverse(self, u):
        theta = self.theta
        ret = (-torch.log(u))**theta
        return ret
    
    def diff(self,t):
        theta = self.theta
        ret = torch.exp(-((t) ** (1/theta)))*(-1/theta)*(t**(1/theta-1))
        return ret

    def pdf(self, X):
        assert X.shape[1] == 2

        u_ = (-torch.log(X[:, 0]))**(self.theta)
        v_ = (-torch.log(X[:, 1]))**(self.theta)

        return torch.exp(-(u_+v_)) ** (1/self.theta)


class IGPhi(PhiLT):
    def __init__(self, theta):
        super(IGPhi, self).__init__()

        self.theta = nn.Parameter(theta)

    def forward(self, t):
        theta = self.theta
        ret = torch.exp((1-torch.sqrt(1+2*theta**2*t))/theta)
        return ret
