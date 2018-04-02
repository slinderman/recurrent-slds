import os
import pickle

import numpy as np

from scipy.special import beta
from scipy.integrate import simps

def logistic(x):
    return 1.0 / (1+np.exp(-x))

def logit(p):
    return np.log(p / (1-p))

def one_hot(x, K):
    return np.array(x[:,None] == np.arange(K)[None, :], dtype=np.float)

def plot_plane(ax3d, normal, point=None, d=0,
               xlim=(-30,30), ylim=(-30,30), zlim=(-30,30),
               **kwargs):
    # a plane is a*x + b*y + c*z + d=0
    # [a,b,c] is the normal. Thus, we have to calculate
    # d and we're set
    if point is not None:
        d = -point.dot(normal)

    # create x,y
    xx, yy = np.meshgrid(np.linspace(*xlim),
                         np.linspace(*ylim))

    # calculate corresponding z
    zz = (-normal[0] * xx - normal[1] * yy - d) * 1. / normal[2]

    # plot the surface
    ax3d.plot_surface(xx, yy, zz, **kwargs)

def plot_2d_plane(ax, normal, point=None, d=0,
               xlim=(-30,30), ylim=(-30,30),
               **kwargs):
    # a plane is a*x + b*y + d=0
    # [a,b] is the normal. Thus, we have to calculate
    # d and we're set
    if point is not None:
        d = -point.dot(normal)

    # create x,y
    xx = np.linspace(*xlim)

    # calculate corresponding y
    yy = (-normal[0] * xx - d) * 1. / normal[1]

    # plot the surface
    ax.plot(xx, yy, **kwargs)



def psi_to_pi(psi, axis=None):
    """
    Convert psi to a probability vector pi
    :param psi:     Length K-1 vector
    :return:        Length K normalized probability vector
    """
    if axis is None:
        if psi.ndim == 1:
            K = psi.size + 1
            pi = np.zeros(K)

            # Set pi[1..K-1]
            stick = 1.0
            for k in range(K-1):
                pi[k] = logistic(psi[k]) * stick
                stick -= pi[k]

            # Set the last output
            pi[-1] = stick
            # DEBUG
            assert np.allclose(pi.sum(), 1.0)

        elif psi.ndim == 2:
            M, Km1 = psi.shape
            K = Km1 + 1
            pi = np.zeros((M,K))

            # Set pi[1..K-1]
            stick = np.ones(M)
            for k in range(K-1):
                pi[:,k] = logistic(psi[:,k]) * stick
                stick -= pi[:,k]

            # Set the last output
            pi[:,-1] = stick

            # DEBUG
            assert np.allclose(pi.sum(axis=1), 1.0)

        else:
            raise ValueError("psi must be 1 or 2D")
    else:
        K = psi.shape[axis] + 1
        pi = np.zeros([psi.shape[dim] if dim != axis else K for dim in range(psi.ndim)])
        stick = np.ones(psi.shape[:axis] + psi.shape[axis+1:])
        for k in range(K-1):
            inds = [slice(None) if dim != axis else k for dim in range(psi.ndim)]
            pi[inds] = logistic(psi[inds]) * stick
            stick -= pi[inds]
        pi[[slice(None) if dim != axis else -1 for dim in range(psi.ndim)]] = stick
        assert np.allclose(pi.sum(axis=axis), 1.)

    return pi

def pi_to_psi(pi):
    """
    Convert probability vector pi to a vector psi
    :param pi:      Length K probability vector
    :return:        Length K-1 transformed vector psi
    """
    if pi.ndim == 1:
        K = pi.size
        assert np.allclose(pi.sum(), 1.0)
        psi = np.zeros(K-1)

        stick = 1.0
        for k in range(K-1):
            psi[k] = logit(pi[k] / stick)
            stick -= pi[k]

        # DEBUG
        assert np.allclose(stick, pi[-1])
    elif pi.ndim == 2:
        M, K = pi.shape
        assert np.allclose(pi.sum(axis=1), 1.0)
        psi = np.zeros((M,K-1))

        stick = np.ones(M)
        for k in range(K-1):
            psi[:,k] = logit(pi[:,k] / stick)
            stick -= pi[:,k]
        assert np.allclose(stick, pi[:,-1])
    else:
        raise NotImplementedError

    return psi


def compute_psi_cmoments(alphas):
    K = alphas.shape[0]
    psi = np.linspace(-10,10,1000)

    mu = np.zeros(K-1)
    sigma = np.zeros(K-1)
    for k in range(K-1):
        density = get_density(alphas[k], alphas[k+1:].sum())
        mu[k] = simps(psi*density(psi),psi)
        sigma[k] = simps(psi**2*density(psi),psi) - mu[k]**2

    return mu, sigma

def get_density(alpha_k, alpha_rest):
    def density(psi):
        return logistic(psi)**alpha_k * logistic(-psi)**alpha_rest \
            / beta(alpha_k,alpha_rest)
    return density

def inhmm_entropy(params, stats):
    log_transmatrices, log_pi_0, aBl, _ = params
    E_z, E_ztztp1T, log_Z = stats
    T, K = E_z.shape
    assert aBl.shape == (T, K)
    assert E_ztztp1T.shape == (T - 1, K, K)
    assert log_transmatrices.shape == (T - 1, K, K)

    neg_entropy = np.sum(E_z[0] * log_pi_0)
    neg_entropy += np.sum(E_z * aBl)
    neg_entropy += np.sum(E_ztztp1T * log_transmatrices)
    neg_entropy -= log_Z
    return -neg_entropy


def cached(results_dir, results_name):
    def _cache(func):
        def func_wrapper(*args, **kwargs):
            results_file = os.path.join(results_dir, results_name)
            if not results_file.endswith(".pkl"):
                results_file += ".pkl"

            if os.path.exists(results_file):
                with open(results_file, "rb") as f:
                    results = pickle.load(f)
            else:
                results = func(*args, **kwargs)
                with open(results_file, "wb") as f:
                    pickle.dump(results, f)

            return results
        return func_wrapper

    return _cache