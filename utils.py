import pickle
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

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


def rand_exp(*dims):
    return -torch.rand(*dims).log()

def rand_simplex(batch_size, dim):
    exp = rand_exp(batch_size, dim)
    return exp / torch.sum(exp, dim=1, keepdim=True)

def rand_positive_stable(alpha, *dims):
    U = np.pi*torch.rand(*dims)
    W = rand_exp(*dims)
    return (torch.sin(alpha*U) / (U.sin() ** (1./alpha))) * (torch.sin((1.-alpha)*U) / W) ** (1./alpha - 1)

def rand_erld(batch_size, dim):
    return rand_exp(batch_size, dim).sum(dim=1)

def I_u(U0,u):
    nsamples = U0.shape[0]
    nquery = u.shape[0]
    ndims = U0.shape[1]
    
    return (U0[:,None,:].expand(nsamples, nquery, ndims)<=u[None,:,:].expand(nsamples, nquery, ndims)).prod(axis=2).float()

def CvM(U1,U2,nquery):
    
    ndims = U1.shape[1]
    u_test = torch.rand(nquery,ndims)
    
    C_u = I_u(U1,u_test).mean(axis=0)
    D_u = I_u(U2,u_test).mean(axis=0)
    
    return ((C_u-D_u)**2).mean()

def cond_cdf(y, cdf, target_dim, cond_dims):
    
    y[y==0] +=1e-15         
    y = y.requires_grad_(True)

    # Numerator
    cur = cdf(y)
    for dim in cond_dims:
        # TODO: Take gradients with respect to one dimension of y at a time
        cur = torch.autograd.grad(cur.sum(), y, create_graph=True)[0][:, dim]
    numerator = cur

    y_trunc = y.detach().clone()
    y_trunc[:,target_dim] = 1
    y_trunc = y_trunc.requires_grad_(True)

    # Denominator
    cur = cdf(y_trunc)
    for dim in cond_dims:
        cur = torch.autograd.grad(cur.sum(), y_trunc, create_graph=True)[0][:, dim]
    denominator = cur

    return numerator/denominator

def cond_sdf(y, sdf, target_dim, cond_dims):
    
    y = y.requires_grad_(True)

    # Numerator
    cur = sdf(y)
    for dim in cond_dims:
        # TODO: Take gradients with respect to one dimension of y at a time
        cur = torch.autograd.grad(cur.sum(), y, create_graph=True)[0][:, dim]
    numerator = cur

    y_trunc = y.detach().clone()
    y_trunc[:,target_dim] = 0
    y_trunc = y_trunc.requires_grad_(True)

    # Denominator
    cur = sdf(y_trunc)
    for dim in cond_dims:
        cur = torch.autograd.grad(cur.sum(), y_trunc, create_graph=True)[0][:, dim]
    denominator = cur

    return numerator/denominator


def pdf(y, cdf, mode='cdf'):
    
    y[y==0] +=1e-15
    y = y.requires_grad_(True)
    
    cur = cdf(y)
    ndims = y.size()[1]
    for dim in range(ndims):
        # TODO: Only take gradients with respect to one dimension of y at at time
        cur = torch.autograd.grad(cur.sum(), y, create_graph=True)[0][:, dim]
    pdf = cur
    
    if mode == 'cdf':
        return pdf
    if mode == 'sdf':
        return (-1)**ndims * pdf
    

def newton_root(phi, y, t0=None, max_iter=200, tol=1e-10):
    '''
    Solve
        f(t) = y
    using the Newton's root finding method.

    Parameters
    ----------
    f: Function which takes in a Tensor t of shape `s` and outputs
    the pointwise evaluation f(t).
    y: Tensor of shape `s`.
    t0: Tensor of shape `s` indicating the initial guess for the root.
    max_iter: Positive integer containing the max. number of iterations.
    tol: Termination criterion for the absolute difference |f(t) - y|.
        By default, this is set to 1e-14,
        beyond which instability could occur when using pytorch `DoubleTensor`.

    Returns:
        Tensor `t*` of size `s` such that f(t*) ~= y
    '''
    if t0 is None:
        t = torch.zeros_like(y) # why not 0.5?
    else:
        t = t0.detach().clone()

    s = y.size()
    for it in range(max_iter):
            
        if hasattr(phi,'ndiff'):
            f_t = phi(t)
            fp_t = phi.ndiff(t,ndiff=1)
        else:
            with torch.enable_grad():
                f_t = phi(t.requires_grad_(True))
                fp_t = torch.autograd.grad(f_t.sum(), t)[0]
                print("warning: autodiff")
        
        assert not torch.any(torch.isnan(fp_t))

        assert f_t.size() == s
        assert fp_t.size() == s

        g_t = f_t - y

        # Terminate algorithm when all errors are sufficiently small.
        if (torch.abs(g_t) < tol).all():
            break

        t = t - g_t / fp_t

    # error if termination criterion (tol) not met. 
    assert torch.abs(g_t).max() < tol, "t=%s, f(t)-y=%s, y=%s, iter=%s, max dev:%s" % (t, g_t, y, it, g_t.max())
    assert t.size() == s
    
    return t


def bisection_root(phi, y, lb=None, ub=None, increasing=True, max_iter=100, tol=1e-3):
    # reduced tol to 1e-6 for 100 iterations
    '''
    Solve
        f(t) = y
    using the bisection method.

    Parameters
    ----------
    f: Function which takes in a Tensor t of shape `s` and outputs
    the pointwise evaluation f(t).
    y: Tensor of shape `s`.
    lb, ub: lower and upper bounds for t.
    increasing: True if f is increasing, False if decreasing.
    max_iter: Positive integer containing the max. number of iterations.
    tol: Termination criterion for the difference in upper and lower bounds.
        By default, this is set to 1e-10,
        beyond which instability could occur when using pytorch `DoubleTensor`.

    Returns:
        Tensor `t*` of size `s` such that f(t*) ~= y
    '''
    if lb is None:
        lb = torch.zeros_like(y)
    if ub is None:
        ub = torch.ones_like(y)

    assert lb.size() == y.size()
    assert ub.size() == y.size()
    assert torch.all(lb <= ub)

    f_lb = phi(lb)
    f_ub = phi(ub)

    for it in range(max_iter):
        t = (lb + ub)/2
        f_t = phi(t)

        if increasing:
            too_low, too_high = f_t < y, f_t > y
            lb[too_low] = t[too_low]
            ub[too_high] = t[too_high]
        else:
            too_low, too_high = f_t > y, f_t < y
            lb[too_low] = t[too_low]
            ub[too_high] = t[too_high]

        assert torch.all(ub - lb >= 0.), "lb: %s, ub: %s" % (lb, ub)

    return t    

def sample(cdf, ndims, N, lb=None, ub=None):
    """
    Conditional sampling method: 
    (i)  compute conditional CDF, 
    (ii) compute inverse of conditional CDF to sample. 
    """
    
    U = torch.rand(N, ndims)

    # don't have to sample dim 0
    for dim in range(1, ndims):
        
        print('Sampling from dim: %s' % dim)
        y = U[:, dim].detach().clone()

        def cond_cdf_func(u):
            U_ = U.detach().clone()
            U_[:, dim] = u
            U_[:, (dim+1):] = 1
            ret = cond_cdf(U_, cdf, dim, list(range(dim)))
            return ret

        # Call inverse using the conditional cdf as the function.
        U[:, dim] = bisection_root(cond_cdf_func, y, lb=lb, ub=ub, increasing=True).detach()

    return U


def sample_sdf(sdf, stdf, ndims, N, seed=142857):

    U = torch.rand(N, ndims)
    
    # dim 0 is Beta(1,ndims-1)
    m = torch.distributions.Beta(torch.tensor([1.]), torch.tensor([ndims-1.]))
    U[:,0] = m.sample((N,))[:,0]

    for dim in range(1, ndims):
        
        print('Sampling from dim: %s' % dim)
                
        y = U[:, dim].detach().clone()
        
        def stdf_func(u):
            U_ = U.detach().clone()
            U_[:, dim] = u
            U_[:, (dim+1):] = 0
            ret = stdf(U_)
            return ret

        ub = bisection_root(stdf_func, torch.ones_like(y), increasing=True).detach()
                
        def cond_sdf_func(u):
            U_ = U.detach().clone()
            U_[:, dim] = u
            U_[:, (dim+1):] = 0
            ret = cond_sdf(U_, sdf, dim, list(range(dim)))
            return ret

        # Call inverse using the conditional sdf as the function.
        U[:, dim] = bisection_root(cond_sdf_func, y, ub=ub, increasing=False).detach()

    return U


def sdf2cdf(y, sdf):
    
    out = 1
    
    for j in range(3):
        y_ = torch.zeros_like(y)
        y_[:,j] = y[:,j]

        out -= sdf(y_) 
    
    for j in range(3):
        y_ = y.clone()
        y_[:,j] = 0
        
        out += sdf(y_)
    
    out -= sdf(y)
    
    return out

def s2r(s):
    m = len(s)+1
    s = np.hstack([1,s])
    r = np.hstack([1,np.ones(m)])
    for j in range(1,m):
        r[m-j] = s[m-j]*r[m-j+1]
    r = r[1:]
    s = s[1:]
    return r

def load_data_0(path):
    f = open(path, 'rb')
    d = pickle.load(f)
    f.close()
    U = d['samples_copula']
    return U

def load_data_axm(path):
    f = open(path, 'rb')
    d = pickle.load(f)
    f.close()
    S = d['samples_simplex']
    R = d['samples_radial']
    U = d['samples_copula']
    Z = d['samples_stdfs']
    return S, R, U, Z

def load_data(path, num_train=None, num_test=None):
    '''
    Loads dataset from `path` split into Pytorch train and test of 
    given sizes. Train set is taken from the front while
    test set is taken from behind.

    :param path: path to .p file containing data.
    '''
    f = open(path, 'rb')
    all_data = pickle.load(f)['samples']
    f.close()

    ndata_all = all_data.size()[0]
    
    if (num_train is None) and (num_test is None):
        num_train = np.floor(all_data*2/3)
        num_test = np.floor(all_data/3)
    elif (num_train is None) and (num_test==0):
        num_train = ndata_all
        
    assert num_train+num_test <= ndata_all

    train_data = all_data[:num_train]
    test_data = all_data[(ndata_all-num_test):]

    return train_data, test_data

def load_data2(path_train, path_test):

    train_data, _ = load_data(path_train, num_test=0)
    test_data, _ = load_data(path_test, num_test=0)
    
    return train_data, test_data

def plot_U(U,labels=None):

    U = U.detach().numpy()
    
    ndims = U.shape[1]

    fig = plt.figure(figsize=(ndims*2.5,ndims*2.5))

    gs = gridspec.GridSpec(ndims, ndims, wspace=0.0, hspace=0.0) 

    for i in range(ndims):

        for j in range(i+1,ndims):
            ax = plt.subplot(gs[i,j])
            ax.scatter(U[:,j], U[:,i],s=1,c='blue')
            ax.axis([-0.05,1.05,-0.05,1.05])
            ax.set_aspect('equal')
            ax.set_xticklabels([]); ax.set_yticklabels([])

        ax = plt.subplot(gs[i,i])
        if labels is None:
            ax.text(0.45,0.5,'$U_%d$'%(i+1),fontsize=30); 
        else:
            ax.text(0.45-len(labels[i])/30.,0.5,'%s'%labels[i],fontsize=30); 
        ax.axis([-0.05,1.05,-0.05,1.05])
        ax.set_aspect('equal')
        ax.set_xticklabels([]); ax.set_yticklabels([])
        ax.set_yticks([0,1]); ax.set_yticklabels([0,1])
        
    plt.tight_layout()
    plt.show()
    plt.clf()
    
def plot_U1_U2(U1,U2,labels=None,filename=None):

    U1 = U1.detach().numpy()
    U2 = U2.detach().numpy()
    
    ndims = U1.shape[1]

    fig = plt.figure(figsize=(ndims*2.5,ndims*2.5))

    gs = gridspec.GridSpec(ndims, ndims, wspace=0.0, hspace=0.0) 

    for i in range(ndims):

        for j in range(i+1,ndims):
            ax = plt.subplot(gs[i,j])
            ax.scatter(U1[:,j], U1[:,i],s=1,c='blue')
            ax.axis([-0.05,1.05,-0.05,1.05])
            ax.set_aspect('equal')
            ax.set_xticklabels([]); ax.set_yticklabels([])
            
        for j in range(0,i):
            ax = plt.subplot(gs[i,j])
            ax.scatter(U2[:,j], U2[:,i],s=1,c='black')
            ax.axis([-0.05,1.05,-0.05,1.05])
            ax.set_aspect('equal')
            ax.set_xticklabels([]); ax.set_yticklabels([])

        ax = plt.subplot(gs[i,i])
        if labels is None:
            ax.text(0.35,0.45,'$U_{%d}$'%(i+1),fontsize=65); 
        else:
            ax.text(0.45-len(labels[i])/30.,0.5,'%s'%labels[i],fontsize=30); 
            #ax.text(0.35,0.45,'%s'%labels[i],fontsize=65);
        ax.axis([-0.05,1.05,-0.05,1.05])
        ax.set_aspect('equal')
        ax.set_xticklabels([]); ax.set_yticklabels([])
        if i==0:
            ax.set_yticks([0,1]); ax.set_yticklabels([0,1])
        
    plt.tight_layout()
    
    if filename is not None:
        plt.savefig(filename)
        plt.clf()
    else:
        plt.show()
        plt.clf()    

import matplotlib.tri as tri
    
def plot_stdf3(stdf, nlevels=200, subdiv=6):
    
    corners = np.array([[0, 0], [1, 0], [0.5, 0.75**0.5]])
    AREA = 0.5 * 1 * 0.75**0.5
    triangle = tri.Triangulation(corners[:, 0], corners[:, 1])
    
    refiner = tri.UniformTriRefiner(triangle)
    trimesh = refiner.refine_triangulation(subdiv=subdiv)
    
    pairs = [corners[np.roll(range(3), -i)[1:]] for i in range(3)]
    tri_area = lambda xy, pair: 0.5 * np.linalg.norm(np.cross(*(pair - xy)))

    def xy2bc(xy, tol=1.e-4):
        '''Converts 2D Cartesian coordinates to barycentric.'''
        coords = np.array([tri_area(xy, p) for p in pairs]) / AREA
        return np.clip(coords, tol, 1.0 - tol)
    
    bc = np.array([xy2bc(xy) for xy in zip(trimesh.x, trimesh.y)])
    pvals = stdf(torch.tensor(bc)).detach().numpy()
    
    #plt.tricontourf(trimesh, pvals, [0.5, 0.6, 0.7, 0.8, 0.9], cmap='autumn', extend="both")
    #plt.tricontourf(trimesh, pvals, [0.7, 0.75, 0.8, 0.85, 0.9], cmap='autumn', extend="both")
    plt.tricontourf(trimesh, pvals, nlevels, cmap='autumn', extend="both")
    plt.colorbar()
    #plt.tricontour(trimesh, pvals, [0, 0.5, 0.6, 0.7, 0.8, 0.9, 1], colors='black')
    #plt.tricontour(trimesh, pvals, [0, 0.7, 0.75, 0.8, 0.85, 0.9, 1], colors='black')
    plt.tricontour(trimesh, pvals, colors='black')
    plt.axis('equal')
    plt.xlim(0, 1)
    plt.ylim(0, 0.75**0.5)
    plt.axis('off')
    plt.show()
    plt.close('all')