import os
import copy
import pickle
import time

import numpy as np
import numpy.random as npr
npr.seed(0)

from scipy.ndimage.filters import gaussian_filter1d

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
from matplotlib.font_manager import FontProperties

import seaborn as sns
color_names = ["windows blue",
               "red",
               "amber",
               "faded green",
               "dusty purple",
               "crimson",
               "greyish"]
colors = sns.xkcd_palette(color_names)
sns.set_style("white")
sns.set_context("paper")

from hips.plotting.colormaps import gradient_cmap
from hips.plotting.sausage import sausage_plot

from pybasicbayes.util.text import progprint_xrange
from pybasicbayes.models import FactorAnalysis
from pybasicbayes.distributions import \
    Regression, Gaussian, AutoRegression

from pyslds.models import HMMSLDS
from pyslds.util import get_empirical_ar_params
from pyhsmm.util.general import relabel_by_permutation
from autoregressive.models import ARWeakLimitStickyHDPHMM

from rslds.decision_list import DecisionList
from pypolyagamma.distributions import MultinomialRegression, BernoulliRegression
from rslds.rslds import PGRecurrentSLDS, PGRecurrentOnlySLDS, StickyPGRecurrentOnlySLDS
from rslds.util import logistic

### Global parameters
T, K, D_obs, D_latent = 10000, 2, 100, 3
mask_start, mask_stop = 750, 850
N_iters = 1000      # Number of iterations of the Gibbs sampler

CACHE_RESULTS = False
RUN_NUMBER = 5
RESULTS_DIR = os.path.join("results", "bernoulli_lorenz", "run{:03d}".format(RUN_NUMBER))

### Helper functions
def cached(results_name):
    if CACHE_RESULTS:
        def _cache(func):
            def func_wrapper(*args, **kwargs):
                results_file = os.path.join(RESULTS_DIR, results_name)
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
    else:
        _cache = lambda func: func

    return _cache

### Plotting code

def make_single_column_figure(mask, y, x_true,
                              rslds_zs, rslds_ps,
                              tlim=(500,1500)):
    """
        Show the following:
         - True data
        """
    fig = plt.figure(figsize=(3.25, 3.5))
    N_subplots = 5
    scale = 3
    gs = gridspec.GridSpec(N_subplots-1+scale, 1)

    # Set up plotting vars
    imask = 1-mask
    plt_slice = slice(*tlim)
    xticks = np.linspace(tlim[0], tlim[1], 5)
    fp = FontProperties()
    fp.set_weight("bold")

    ax1 = fig.add_subplot(gs[0, 0])
    for d in range(D_latent):
        ax1.plot(np.arange(*tlim), x_true[plt_slice, d], lw=1.5 if d > 0 else 2, color=colors[d])

    # Overlay mask
    ylim = (-2,2)
    ax1.imshow(imask[plt_slice,0][None, :], cmap="Greys", alpha=0.25, extent=tlim + ylim, aspect="auto")
    ax1.set_xticks(xticks)
    ax1.set_xticklabels([])
    ax1.set_yticks([-2., 0, 2.])
    ax1.set_ylim(ylim)
    ax1.set_xlabel("Time")
    ax1.set_ylabel("True $x$")
    ax1.set_title("Missing Data Reconstruction")

    # Observed data 3d
    ax2 = fig.add_subplot(gs[1:scale+1, 0])
    ax2.imshow(y[plt_slice].T, cmap="Greys", extent=tlim + (D_obs,0), interpolation="nearest", aspect="auto")
    ax2.imshow(imask[plt_slice].T, cmap="Greys", extent=tlim + (D_obs,0), alpha=0.25, aspect="auto")
    ax2.set_xticks(xticks, [])
    ax2.set_xticklabels([])
    ax2.set_ylabel("Observed $y$", labelpad=-1)

    # Inferred discrete states (final sample)
    ax3 = fig.add_subplot(gs[scale + 1, 0])
    im = ax3.imshow(rslds_zs[-1, plt_slice][None,:], aspect="auto", extent=tlim + (0,1), vmin=0, vmax=K - 1,
                    cmap=gradient_cmap(colors[:K]), interpolation="nearest")
    ax3.imshow(imask[plt_slice, 0][None, :], cmap="Greys", alpha=0.25, extent=tlim + (0,1), aspect="auto")
    ax3.set_xticks(xticks, [])
    ax3.set_xticklabels([])
    ax3.set_yticks([])
    ax3.set_ylabel("Inf. $z$", labelpad=17)

    # Filling in missing data
    iorslds_p = roslds.states_list[0].smooth()[0]

    n_to_plot = [0, 3]
    ax4 = fig.add_subplot(gs[scale + 2, 0])
    ax4.plot(np.arange(*tlim), iorslds_p[plt_slice, n_to_plot[0]], color=colors[3])

    # Find the spikes and thin
    spks = np.where(y[plt_slice, n_to_plot[0]] == 1)[0]
    if len(spks) > 100:
        spks_to_plot = np.random.choice(spks, size=100, replace=False)

    ax4.plot(tlim[0] + spks_to_plot, np.ones_like(spks_to_plot), 'ko', markersize=2)
    ax4.imshow(imask[plt_slice].T, cmap="Greys", extent=tlim + (0, 1.2), alpha=0.25, aspect="auto")

    ax4.set_xlim(tlim)
    ax4.set_xticks(xticks)
    ax4.set_xticklabels([])
    ax4.set_ylim(0,1.2)
    ax4.set_yticks([0,1])
    ax4.set_ylabel("Inf. $\\rho_{1}$", labelpad=5)

    # Do it again for one more output
    ax5 = fig.add_subplot(gs[scale + 3, 0])
    ax5.plot(np.arange(*tlim), iorslds_p[plt_slice, n_to_plot[1]], color=colors[3])

    # Find the spikes and thin
    spks = np.where(y[plt_slice, n_to_plot[1]] == 1)[0]
    if len(spks) > 100:
        spks_to_plot = np.random.choice(spks, size=100, replace=False)

    ax5.plot(tlim[0] + spks_to_plot, np.ones_like(spks_to_plot), 'ko', markersize=2)
    ax5.imshow(imask[plt_slice].T, cmap="Greys", extent=tlim + (0, 1.2), alpha=0.25, aspect="auto")

    ax5.set_xlim(tlim)
    ax5.set_xticks(xticks)
    ax5.set_xlabel("Time")
    ax5.set_ylim(0, 1.2)
    ax5.set_yticks([0, 1])
    ax5.set_ylabel("Inf. $\\rho_{2}$", labelpad=5)

    # plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "reconstruction.pdf"))
    plt.savefig(os.path.join(RESULTS_DIR, "reconstruction.png"), dpi=300)
    plt.show()

def make_figure(y, x_true, p_true,
                rslds, zs_rslds, x_rslds,
                rslds_p_mean, rslds_p_std,
                slds, z_slds, x_slds,
                z_rslds_gen, x_rslds_gen,
                z_slds_gen, x_slds_gen,
                tlim=(500, 1500)):
    """
    Show the following:
     - True data
    """
    fig = plt.figure(figsize=(6.5,3.5))
    gs = gridspec.GridSpec(8,3)

    fp = FontProperties()
    fp.set_weight("bold")

    # Set up plotting vars
    imask = 1 - mask
    plt_slice = slice(*tlim)
    xticks = np.linspace(tlim[0], tlim[1], 5)

    # Observed data 3d
    ax1 = fig.add_subplot(gs[:4,0], projection="3d")
    plot_latent_states(x_true[:2500], ax=ax1)
    ax1.set_title("True Latent States")
    plt.figtext(.025, 1-.075, '(a)', fontproperties=fp)

    ## Plot the inferred dynamics under the rSLDS
    # Do an orthogonal coordinate transform
    xhat_rslds, R_rslds = solve_procrustes(x_true, x_rslds[-1])
    xhat_slds, R_slds = solve_procrustes(x_true, x_slds[-1])

    ax2 = fig.add_subplot(gs[4:, 0], projection="3d")
    ts = np.random.choice(T, size=400, replace=False)
    plot_most_likely_dynamics3d(rslds.trans_distn,
                                rslds.dynamics_distns,
                                x_rslds[-1][ts],
                                R=R_rslds,
                                ax=ax2, alpha=0.25)

    # Overlay a partial trajectory
    ax2.set_title("Inferred Dynamics (rSLDS)")
    plt.figtext(.025, .62 - .075, '(b)', fontproperties=fp)

    # Plot the output vs time
    ax3a = fig.add_subplot(gs[0:2, 1])
    for d in range(D_latent):
        ax3a.plot(np.arange(*tlim), x_true[plt_slice, d], lw=1.5 if d > 0 else 2, color=colors[d])

    # Overlay mask
    ylim = (-2, 2)
    ax3a.imshow(imask[plt_slice, 0][None, :], cmap="Greys", alpha=0.25, extent=tlim + ylim, aspect="auto")
    ax3a.set_xticks(xticks)
    ax3a.set_xticklabels([])
    ax3a.set_yticks([-2., 0, 2.])
    ax3a.set_ylim(ylim)
    ax3a.set_ylabel("True $x$")
    plt.figtext(.27 + .025, 1. - .075, '(c)', fontproperties=fp)

    # Observed  Bernoulli data
    ax3b = fig.add_subplot(gs[2:4, 1])
    ax3b.imshow(y[plt_slice].T, cmap="Greys", extent=tlim + (D_obs, 0), interpolation="nearest", aspect="auto")
    ax3b.imshow(imask[plt_slice].T, cmap="Greys", extent=tlim + (D_obs, 0), alpha=0.25, aspect="auto")
    ax3b.set_xticks(xticks, [])
    ax3b.set_xticklabels([])
    ax3b.set_ylabel("Observed $y$", labelpad=-1)
    # ax3b.set_title("Observed $y$")
    # ax3b.set_xlabel("Time")
    plt.figtext(.27 + .025, .8 - .075, '(d)', fontproperties=fp)


    # Plot something... z samples?
    ax4 = fig.add_subplot(gs[4:6,1])
    plot_z_samples(zs_rslds, plt_slice=tlim, ax=ax4)
    ax4.imshow(imask[plt_slice, 0][None, :], cmap="Greys", alpha=0.25, extent=tlim + (len(zs_rslds),0), aspect="auto")
    ax4.set_xticks(xticks, [])
    ax4.set_xticklabels([])
    ax4.set_yticks([0, len(zs_rslds)])
    ax4.set_yticklabels(["0", "10$^3$"])
    ax4.set_ylabel("Sampled $z$", labelpad=-1)
    ax4.set_xlabel("")
    plt.figtext(.27 + .025, .6 - .075, '(e)', fontproperties=fp)


    # Do it again for one more output
    ax5 = fig.add_subplot(gs[6:8, 1])
    sausage_plot(np.arange(*tlim), rslds_p_mean[plt_slice, 0], 3 * rslds_p_std[plt_slice, 0], sgax=ax5, color=colors[3], alpha=0.5)
    ax5.plot(np.arange(*tlim), p_true[plt_slice,0], color='k', ls='-', lw=0.5)
    ax5.plot(np.arange(*tlim), rslds_p_mean[plt_slice, 0], color=colors[3], lw=1)

    # Find the spikes and thin
    spks = np.where(y[plt_slice, 0] == 1)[0]
    if len(spks) > 100:
        spks_to_plot = np.random.choice(spks, size=100, replace=False)

    ax5.plot(tlim[0] + spks_to_plot, np.ones_like(spks_to_plot), 'ko', markersize=1.5)
    ax5.imshow(imask[plt_slice].T, cmap="Greys", extent=tlim + (0, 1.2), alpha=0.25, aspect="auto")

    ax5.set_xlim(tlim)
    ax5.set_xticks(xticks)
    ax5.set_xlabel("Time")
    ax5.set_ylim(0, 1.2)
    ax5.set_yticks([0, 1])
    ax5.set_ylabel("Inf. $\\rho_{1}$", labelpad=6)
    plt.figtext(.27 + .025, .35 - .075, '(f)', fontproperties=fp)

    # Plot simulated SLDS data
    ax5 = fig.add_subplot(gs[:4, 2], projection="3d")
    plot_latent_states(x_slds_gen[100:].dot(R_slds), z=z_slds_gen[100:], ax=ax5)
    ax5.set_title("Generated States (SLDS)")
    plt.figtext(.66 + .025, 1. - .075, '(g)', fontproperties=fp)

    # Plot simulated rSLDS data
    ax6 = fig.add_subplot(gs[4:, 2], projection="3d")
    plot_latent_states(x_rslds_gen[100:].dot(R_rslds), z=z_rslds_gen[100:], ax=ax6)
    ax6.set_title("Generated States (rSLDS)")
    plt.figtext(.66 + .025, .6 - .075, '(h)', fontproperties=fp)

    plt.tight_layout()
    # plt.savefig(os.path.join(RESULTS_DIR, "bernoulli_lorenz.png"), dpi=200)
    # plt.savefig(os.path.join(RESULTS_DIR, "bernoulli_lorenz.pdf"))
    plt.show()

def solve_procrustes(x_true, x_inf):
    from scipy.linalg import orthogonal_procrustes
    R, scale = orthogonal_procrustes(x_inf, x_true)
    return x_inf.dot(R), R

def plot_most_likely_dynamics3d(
        reg, dynamics_distns, xyz,
        alpha=0.5, length=0.5,
        R=None,
        ax=None, figsize=(3,3)):
    # Get the probability of each state at each xy location
    inputs = np.hstack((np.zeros((xyz.shape[0], reg.D_in - D_latent)), xyz))
    prs = reg.pi(inputs)
    z = np.argmax(prs, axis=1)

    if R is None:
        R = np.eye(D_latent)

    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)

    for k in range(K):
        A = dynamics_distns[k].A[:, :D_latent]
        b = dynamics_distns[k].A[:, D_latent:]
        xyzn = xyz.dot(A.T) + b.T

        # Compute xyz -> xyz * R and xyzn -> xyzn * R
        rxyz = xyz.dot(R)
        rxyzn = xyzn.dot(R)
        dydt_m = rxyzn - rxyz

        zk = z == k
        ax.quiver(rxyz[zk, 0], rxyz[zk, 1], rxyz[zk, 2],
                  dydt_m[zk, 0], dydt_m[zk, 1], dydt_m[zk, 2],
                  color=colors[k], alpha=alpha, length=length)

    # Plot scale bar of length 2
    xlim = ax.get_xlim()
    zlim = ax.get_zlim()
    ax.plot([xlim[0] + .95 * (xlim[1] - xlim[0]),
             xlim[0] + .95 * (xlim[1] - xlim[0])],
            [-1, 1],
            [zlim[0] + .05 * (zlim[1] - zlim[0]),
             zlim[0] + .05 * (zlim[1] - zlim[0])],
            '-k', lw=1)

    ax.w_xaxis.set_pane_color(.95 * np.ones(3))
    ax.w_yaxis.set_pane_color(.95 * np.ones(3))
    ax.w_zaxis.set_pane_color(.95 * np.ones(3))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.set_xlabel('$x_1$', labelpad=-15)
    ax.set_ylabel('$x_2$', labelpad=-15)
    ax.set_zlabel('$x_3$', labelpad=-15)

    return ax

def plot_latent_states(y, z=None, ax=None, ls="-", filename=None):
    if ax is None:
        fig = plt.figure(figsize=(4, 4))
        ax = fig.add_subplot(111, projection="3d")

    if z is None:
        ax.plot(y[:, 0], y[:, 1], y[:, 2],
                lw=0.5, ls=ls, color="gray")
    else:
        zcps = np.concatenate(([0], np.where(np.diff(z))[0] + 1, [z.size]))
        for start, stop in zip(zcps[:-1], zcps[1:]):
            stop = min(z.shape[0], stop+1)
            ax.plot(y[start:stop+1, 0],
                    y[start:stop+1, 1],
                    y[start:stop+1, 2],
                    lw=0.5, ls=ls, color=colors[z[start]])

    ax.w_xaxis.set_pane_color(.95 * np.ones(3))
    ax.w_yaxis.set_pane_color(.95 * np.ones(3))
    ax.w_zaxis.set_pane_color(.95 * np.ones(3))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.set_xlabel('$x_1$', labelpad=-15)
    ax.set_ylabel('$x_2$', labelpad=-15)
    ax.set_zlabel('$x_3$', labelpad=-15)

    # Plot scale bar of length 2
    xlim = ax.get_xlim()
    zlim = ax.get_zlim()
    ax.plot([xlim[0] + .95 * (xlim[1]-xlim[0]),
             xlim[0] + .95 * (xlim[1] - xlim[0])],
            [-1, 1],
            [zlim[0] + .05 * (zlim[1] - zlim[0]),
             zlim[0] + .05 * (zlim[1] - zlim[0])],
            '-k', lw=1)


    if filename is not None:
        plt.savefig(filename)

    return ax

def plot_z_samples(zs, zref=None,
                   plt_slice=None,
                   N_iters=None,
                   title=None,
                   ax=None,
                   filename=None):

    if ax is None:
        fig = plt.figure(figsize=(10,5))
        ax = fig.add_subplot(111)

    zs = np.array(zs)
    if plt_slice is None:
        plt_slice = (0, zs.shape[1])
    if N_iters is None:
        N_iters = zs.shape[0]

    im = ax.imshow(zs[:,slice(*plt_slice)], aspect=2., vmin=0, vmax=K-1,
                     cmap=gradient_cmap(colors[:K]), interpolation="nearest",
                     extent=plt_slice + (N_iters, 0))
    ax.set_ylabel("Iteration")

    if zref is not None:
        divider = make_axes_locatable(ax)
        ax2 = divider.append_axes("bottom", size="10%", pad=0.05)

        zref = np.atleast_2d(zref)
        im = ax2.imshow(zref[:, slice(*plt_slice)], aspect=1., vmin=0, vmax=K-1,
                         cmap=gradient_cmap(colors[:K]), interpolation="nearest")
        # ax2.autoscale(False)
        ax.set_xticks([])
        ax2.set_yticks([])
        ax2.set_ylabel("True $z$", rotation=0)
        ax2.yaxis.set_label_coords(-.15, -.5)
        ax2.set_xlabel("Time")
    else:
        ax.set_xlabel("Time")

    if title is not None:
        ax.set_title(title)

    if filename is not None:
        plt.savefig(os.path.join(RESULTS_DIR, filename))

def lorenz(T=3000, dt=0.01):
    from scipy.integrate import odeint

    # Lorenz attractor parameters
    mu = np.array([-2.5, -2.5, 24])
    std = 15.
    standardize = lambda state: (state - mu) / std
    unstandardize = lambda stdstate: std * stdstate + mu

    sigma = 10.0
    rho = 28.0
    beta = 8.0 / 3.0

    def _lorenz(stdstate, t):
        # unpack the state vector
        state = unstandardize(stdstate)
        x = state[0]
        y = state[1]
        z = state[2]

        # compute state derivatives
        xd = sigma * (y - x)
        yd = (rho - z) * x - y
        zd = x * y - beta * z

        # standardize the derivatives
        # dxs / dt = dxs / dx * dx /dt
        #          = 1/std * dx / dt
        dstate = np.array([xd, yd, zd]) / std

        # return the state derivatives
        return dstate

    x0 = standardize(np.array([2.0, 3.0, 4.0]))
    t = np.arange(0.0, T * dt, dt)
    xfull = odeint(_lorenz, x0, t)

    return xfull

### Make an example with 2D latent states and 4 discrete states
@cached("simulated_lorenz")
def simulate_lorenz():
    x = lorenz(T)

    if D_latent == D_obs:
        C = np.eye(D_latent)
    else:
        C = np.random.randn(D_obs, D_latent)

    # Simulate Bernoulli observations
    b = -1.0
    psi = x.dot(C.T) + b
    p = 1./(1+np.exp(-psi))
    y = np.random.rand(T,D_obs) < p

    mask = np.ones((T, D_obs), dtype=bool)
    mask[mask_start:mask_stop] = False
    return x, p, y, C, b, mask

### Factor Analysis and PCA for dimensionality reduction
@cached("factor_analysis")
def fit_factor_analysis(y, mask=None, N_iters=100):
    print("Fitting Factor Analysis")
    model = FactorAnalysis(D_obs, D_latent)

    if mask is None:
        mask = np.ones_like(y, dtype=bool)

    # Center the data
    b = y.mean(0)
    data = model.add_data(y-b, mask=mask)
    for _ in progprint_xrange(N_iters):
        model.resample_model()

    C_init = np.column_stack((model.W, b))
    return data.Z, C_init

@cached("pca")
def fit_pca(y, whiten=True):
    print("Fitting PCA")
    from sklearn.decomposition import PCA

    # First smooth the data with a Gaussian filter
    p_smooth = gaussian_filter1d(y.astype(np.float), sigma=10, axis=0)

    # Invert the logistic transformation
    assert np.all((p_smooth >= 0) & (p_smooth <=1))

    # Clip
    p_smooth = np.clip(p_smooth, 0.01, 0.99)
    psi_smooth = np.log(p_smooth / (1-p_smooth))

    model = PCA(n_components=D_latent, whiten=whiten)
    x_init = model.fit_transform(psi_smooth)
    C_init = model.components_.T
    b_init = model.mean_[:,None]
    sigma = np.sqrt(model.explained_variance_)

    # inverse transform is given by
    # X.dot(sigma * C_init.T) + b_init.T
    if whiten:
        C_init = sigma * C_init

    return p_smooth, psi_smooth, x_init, np.column_stack((C_init, b_init))

### Make an ARHMM for initialization
@cached("arhmm")
def fit_arhmm(x, affine=True):
    print("Fitting Sticky ARHMM")
    dynamics_hypparams = \
        dict(nu_0=D_latent + 2,
             S_0=np.eye(D_latent),
             M_0=np.hstack((np.eye(D_latent), np.zeros((D_latent, int(affine))))),
             K_0=np.eye(D_latent + affine),
             affine=affine)
    dynamics_hypparams = get_empirical_ar_params([x], dynamics_hypparams)

    dynamics_distns = [
        AutoRegression(
            A=np.column_stack((0.99 * np.eye(D_latent),
                               np.zeros((D_latent, int(affine))))),
            sigma=np.eye(D_latent),
            **dynamics_hypparams)
        for _ in range(K)]

    init_distn = Gaussian(nu_0=D_latent + 2,
                          sigma_0=np.eye(D_latent),
                          mu_0=np.zeros(D_latent),
                          kappa_0=1.0)

    arhmm = ARWeakLimitStickyHDPHMM(
        init_state_distn='uniform',
        init_emission_distn=init_distn,
        obs_distns=dynamics_distns,
        alpha=3.0, kappa=10.0, gamma=3.0)

    arhmm.add_data(x)

    lps = []
    for _ in progprint_xrange(1000):
        arhmm.resample_model()
        lps.append(arhmm.log_likelihood())

    z_init = arhmm.states_list[0].stateseq
    z_init = np.concatenate(([0], z_init))

    return arhmm, z_init

### Use a DecisionList to permute the discrete states
def fit_decision_list(z, y):
    print("Fitting Decision List")
    dlist = DecisionList(K, D_latent)
    dlist.fit(y[:-1], z[1:])

    dl_reg = MultinomialRegression(1, K, D_latent)
    dl_reg.A = dlist.weights.copy()
    dl_reg.b = dlist.biases[:,None].copy()

    z_perm = \
        relabel_by_permutation(z, np.argsort(dlist.permutation))

    return z_perm, dl_reg

def make_rslds_parameters(C_init):
    init_dynamics_distns = [
        Gaussian(
            mu=np.zeros(D_latent),
            sigma=np.eye(D_latent),
            nu_0=D_latent + 2, sigma_0=3. * np.eye(D_latent),
            mu_0=np.zeros(D_latent), kappa_0=1.0,
        )
        for _ in range(K)]

    dynamics_distns = [
        Regression(
            nu_0=D_latent + 2,
            S_0=1e-4 * np.eye(D_latent),
            M_0=np.hstack((np.eye(D_latent), np.zeros((D_latent, 1)))),
            K_0=np.eye(D_latent + 1),
        )
        for _ in range(K)]

    emission_distns = \
        BernoulliRegression(D_obs, D_latent + 1,
                            A=C_init.copy())

    return init_dynamics_distns, dynamics_distns, emission_distns


def _fit_model(model, y, mask):
    # Fit the model
    i_test = np.where(np.any(mask==False, axis=1))[0]

    lps = []
    hlls = []
    z_smpls = []
    x_smpls = []
    t_start = time.time()
    for _ in progprint_xrange(N_iters):
        model.resample_model()
        lps.append(model.log_likelihood())
        z_smpls.append(model.stateseqs[0].copy())
        x = model.states_list[0].gaussian_states.copy()
        x_smpls.append(x)

        # Compute the firing probability
        C, d = model.emission_distns[0].A[:, :D_latent], \
               model.emission_distns[0].A[:, D_latent] + model.emission_distns[0].b[:, 0]
        psi = x[i_test].dot(C.T) + d[None, :]
        p = logistic(psi)
        hlls.append(np.sum(y[i_test] * np.log(p) + \
                           (1 - y[i_test]) * np.log(1 - p)))
    t_stop = time.time()

    z_smpls = np.array(z_smpls)
    x_smpls = np.array(x_smpls)
    lps = np.array(lps)
    hlls = np.array(hlls)

    return model, lps, hlls, z_smpls, x_smpls, (t_stop - t_start)


@cached("lds")
def fit_lds(inputs, z_init, x_init, y, mask, C_init):
    print("Fitting standard LDS")
    init_dynamics_distns = [
        Gaussian(
            mu=np.zeros(D_latent),
            sigma=np.eye(D_latent),
            nu_0=D_latent + 2, sigma_0=3. * np.eye(D_latent),
            mu_0=np.zeros(D_latent), kappa_0=1.0,
        )]

    dynamics_distns = [
        Regression(
            nu_0=D_latent + 2,
            S_0=1e-4 * np.eye(D_latent),
            M_0=np.hstack((np.eye(D_latent), np.zeros((D_latent, 1)))),
            K_0=np.eye(D_latent + 1),
        )]

    emission_distns = \
        BernoulliRegression(D_obs, D_latent + 1,
                            A=C_init.copy())

    lds = HMMSLDS(
        init_state_distn='uniform',
        init_dynamics_distns=init_dynamics_distns,
        dynamics_distns=dynamics_distns,
        emission_distns=emission_distns,
        alpha=3.)

    lds.add_data(y, inputs=inputs, mask=mask)

    # Initialize states
    lds.states_list[0].stateseq = z_init.copy().astype(np.int32)
    lds.states_list[0].gaussian_states = x_init.copy()

    # Initialize dynamics
    print("Initializing dynamics with Gibbs sampling")
    for _ in progprint_xrange(100):
        lds.resample_dynamics_distns()

    return _fit_model(lds, y, mask)

@cached("slds")
def fit_slds(inputs, z_init, x_init, y, mask, C_init):
    print("Fitting standard SLDS")
    init_dynamics_distns, dynamics_distns, emission_distns = \
        make_rslds_parameters(C_init)

    slds = HMMSLDS(
        init_state_distn='uniform',
        init_dynamics_distns=init_dynamics_distns,
        dynamics_distns=dynamics_distns,
        emission_distns=emission_distns,
        alpha=3.)

    slds.add_data(y, inputs=inputs, mask=mask)

    # Initialize states
    slds.states_list[0].stateseq = z_init.copy().astype(np.int32)
    slds.states_list[0].gaussian_states = x_init.copy()

    # Initialize dynamics
    print("Initializing dynamics with Gibbs sampling")
    for _ in progprint_xrange(100):
        slds.resample_dynamics_distns()

    return _fit_model(slds, y, mask)

@cached("rslds")
def fit_rslds(inputs, z_init, x_init, y, mask, dl_reg, C_init):
    print("Fitting rSLDS")
    init_dynamics_distns, dynamics_distns, emission_distns = \
        make_rslds_parameters(C_init)

    rslds = PGRecurrentSLDS(
        D_in=D_latent,
        trans_params=dict(sigmasq_A=10000., sigmasq_b=10000.,
                          A=np.hstack((np.zeros((K - 1, K)), dl_reg.A)),
                          b=dl_reg.b),
        init_state_distn='uniform',
        init_dynamics_distns=init_dynamics_distns,
        dynamics_distns=dynamics_distns,
        emission_distns=emission_distns,
        fixed_emission=False,
        alpha=3.)

    rslds.add_data(y, inputs=inputs, mask=mask)

    # Initialize states
    rslds.states_list[0].stateseq = z_init.copy()
    rslds.states_list[0].gaussian_states = x_init.copy()

    # Initialize dynamics
    print("Initializing dynamics with Gibbs sampling")
    for _ in progprint_xrange(100):
        rslds.resample_dynamics_distns()

    return _fit_model(rslds, y, mask)

@cached("roslds")
def fit_roslds(inputs, z_init, x_init, y, mask, dl_reg, C_init,
               N_iters=10000):
    print("Fitting input only rSLDS")
    init_dynamics_distns, dynamics_distns, emission_distns = \
        make_rslds_parameters(C_init)

    rslds = PGRecurrentOnlySLDS(
        trans_params=dict(sigmasq_A=10000., sigmasq_b=10000.,
                          A=np.hstack((np.zeros((K-1, K)), dl_reg.A)),
                          b=dl_reg.b),
        init_state_distn='uniform',
        init_dynamics_distns=init_dynamics_distns,
        dynamics_distns=dynamics_distns,
        emission_distns=emission_distns,
        fixed_emission=False,
        alpha=3.)

    rslds.add_data(y, inputs=inputs, mask=mask)

    # Initialize states
    rslds.states_list[0].stateseq = z_init.copy()
    rslds.states_list[0].gaussian_states = x_init.copy()

    # Initialize dynamics
    print("Initializing dynamics with Gibbs sampling")
    for _ in progprint_xrange(100):
        rslds.resample_dynamics_distns()

    return _fit_model(rslds, y, mask)


@cached("sticky_inputonly_rslds")
def fit_sticky_inputonly_rslds(inputs, z_init, x_init, y, mask, dl_reg, C_init,
                        N_iters=10000):
    print("Fitting input only rSLDS")
    init_dynamics_distns, dynamics_distns, emission_distns = \
        make_rslds_parameters(C_init)

    rslds = StickyPGRecurrentOnlySLDS(
        D_in=D_latent,
        trans_params=dict(sigmasq_A=100., sigmasq_b=100., kappa=1.0,
                          A=np.hstack((np.zeros((K-1, K)), dl_reg.A)),
                          b=dl_reg.b),
        init_state_distn='uniform',
        init_dynamics_distns=init_dynamics_distns,
        dynamics_distns=dynamics_distns,
        emission_distns=emission_distns,
        fixed_emission=False,
        alpha=3.)

    rslds.add_data(y, inputs=inputs, mask=mask)

    # Initialize states
    rslds.states_list[0].stateseq = z_init.copy()
    rslds.states_list[0].gaussian_states = x_init.copy()

    # Initialize dynamics
    print("Initializing dynamics with Gibbs sampling")
    for _ in progprint_xrange(100):
        rslds.resample_dynamics_distns()

    return _fit_model(rslds, y, mask)

def run_ppc(statistic, y, models,
            N_iter=100, T_gen=1000):
    """
    Run a simple posterior predictive check to see
    if the models generate realistic data

    :param statistic: a function from y to a scalar
    """
    true_s = statistic(y)

    inputs = np.ones((T_gen, 1))
    test_ss = []
    for model in models:
        # import ipdb; ipdb.set_trace()
        print("Testing model {}".format(model))
        test_s = []
        for _ in progprint_xrange(N_iter):
            (ys_gen, _), _ = model.generate(T=T_gen, inputs=inputs, with_noise=True)
            y_gen = ys_gen[0][200:]
            test_s.append(statistic(y_gen))
        test_ss.append(np.array(test_s))

    print("True:   {}".format(true_s))
    for model, test_s in zip(models, test_ss):
        print("Model: {}".format(model))
        print("Mean:  {}".format(test_s.mean()))
        print("Std:   {}".format(test_s.std()))
        print("")


def run_ppc_on_probs(statistic, p, models,
            N_iter=100, T_gen=1000):
    """
    Run a simple posterior predictive check to see
    if the models generate realistic data

    :param statistic: a function from y to a scalar
    """
    true_s = statistic(p)

    inputs = np.ones((T_gen, 1))
    test_ss = []
    for model in models:
        # import ipdb; ipdb.set_trace()
        print("Testing model {}".format(model))
        test_s = []
        for _ in progprint_xrange(N_iter):
            (_, x_gen), _ = model.generate(T=T_gen, inputs=inputs, with_noise=True)
            # x_gen = x_gen[200:]

            psi_gen = x_gen.dot(model.emission_distns[0].A[:, :D_latent].T) + \
                      model.emission_distns[0].A[:, D_latent] + \
                      model.emission_distns[0].b.T
            p_gen = logistic(psi_gen)

            test_s.append(statistic(p_gen))
        test_ss.append(np.array(test_s))

    print("True:   {}".format(true_s))
    for model, test_s in zip(models, test_ss):
        print("Model: {}".format(model))
        print("Mean:  {}".format(test_s.mean()))
        print("Std:   {}".format(test_s.std()))
        print("")


if __name__ == "__main__":
    ## Simulate Lorenz data
    x_true, p_true, y, C, b, mask = simulate_lorenz()
    i_test = np.where(np.any(mask==False, axis=1))[0]
    inputs = np.ones((T,1))

    ## Run PCA on smoothed spike counts
    _, _, x_init, C_init = fit_pca(y)
    # x_init[i_test] = 0
    x_init[i_test] = np.random.randn(*x_init[i_test].shape)

    ## Fit a sticky ARHMM for initialization
    arhmm, z_init = fit_arhmm(x_init)
    z_init[i_test] = np.random.randint(0, K, size=i_test.size)

    ## Fit a DecisionList to get a permutation of z_init
    z_perm, dl_reg = fit_decision_list(z_init, x_init)

    ## Fit a standard LDS
    lds, lds_lps, lds_hlls, lds_z_smpls, lds_x_smpls, t_lds = \
        fit_lds(inputs, z_perm, x_init, y, mask, C_init)

    ## Fit a standard SLDS
    slds, slds_lps, slds_hlls, slds_z_smpls, slds_x_smpls, t_slds = \
        fit_slds(inputs, z_perm, x_init, y, mask, C_init)

    ## Fit a recurrent SLDS
    # rslds, rslds_lps, rslds_hlls, rslds_z_smpls, rslds_x, t_rslds = \
    #     fit_rslds(inputs, z_perm, x_init, y, mask, dl_reg, C_init)

    ## Fit an input-only recurrent SLDS
    roslds, roslds_lps, roslds_hlls, roslds_z_smpls, roslds_x_smpls, t_roslds = \
        fit_roslds(inputs, z_perm, x_init, y, mask, dl_reg, C_init)

    ## Fit a sticky input-only recurrent SLDS
    # siorslds, siorslds_lps, siorslds_hlls,  siorslds_z_smpls, siorslds_x_smpls, t_siorslds = \
    #     fit_sticky_inputonly_rslds(inputs, z_perm, x_init, y, mask, dl_reg, C_init)

    # Compute the spiking probability
    test_model, test_x_smpls = roslds, roslds_x_smpls
    C, d = test_model.emission_distns[0].A[:,:D_latent], \
           test_model.emission_distns[0].A[:,D_latent] + \
           test_model.emission_distns[0].b[:,0].T
    rslds_psi_smpls = np.array([x.dot(C[:1].T) + d[0] for x in test_x_smpls])
    rslds_p_smpls = logistic(rslds_psi_smpls)
    rslds_p_mean = rslds_p_smpls.mean(0)
    rslds_p_std = rslds_p_smpls.std(0)

    ## Generate from the model
    T_gen = 10000
    inputs = np.ones((T_gen, 1))
    (rslds_y_gen, rslds_x_gen), rslds_z_gen = test_model.generate(T=T_gen, inputs=inputs, with_noise=True)
    rslds_psi_gen = rslds_x_gen.dot(test_model.emission_distns[0].A[:, :D_latent].T) + \
                    test_model.emission_distns[0].A[:, D_latent] + \
                    test_model.emission_distns[0].b.T
    rslds_p_gen = logistic(rslds_psi_gen)

    for k in range(K):
        slds.init_dynamics_distns[k].mu = x_init[0].copy()
        slds.init_dynamics_distns[k].sigma = 1e-3 * np.eye(D_latent)

    slds_y_gen, slds_x_gen, slds_z_gen = slds.generate(T=T_gen, inputs=inputs)
    slds_psi_gen = slds_x_gen.dot(slds.emission_distns[0].A[:, :D_latent].T) + \
                   slds.emission_distns[0].A[:, D_latent] + \
                   slds.emission_distns[0].b.T
    slds_p_gen = logistic(slds_psi_gen)
    # slds_y_gen = slds_x_gen.dot(slds.emission_distns[0].A[:,:D_latent].T) + slds.emission_distns[0].A[:,D_latent]

    lds_y_gen, lds_x_gen, lds_z_gen = lds.generate(T=T_gen, inputs=inputs)
    lds_psi_gen = lds_x_gen.dot(lds.emission_distns[0].A[:, :D_latent].T) + \
                  lds.emission_distns[0].A[:, D_latent] + \
                  lds.emission_distns[0].b.T
    lds_p_gen = logistic(lds_psi_gen)

    make_figure(y, x_true, p_true,
                roslds, roslds_z_smpls, roslds_x_smpls,
                rslds_p_mean, rslds_p_std,
                slds, slds_z_smpls, slds_x_smpls,
                rslds_z_gen, rslds_x_gen,
                slds_z_gen, slds_x_gen,
                )

    ## Run a posterior predictive check
    # statistic = lambda y: y[:,0].std()
    # statistic = lambda y: np.corrcoef(y[:-10,0], y[10:,0])[0,1]
    # statistic = lambda y: np.max(np.sum(y, axis=1))

    # from pyhsmm.util.general import rle
    # def statistic(y):
    #     yv, vlen = rle(y[:,0])
    #     return np.max(vlen)

    # run_ppc(statistic, y, [lds, slds, iorslds])

    # Run a posterior predictive check on the true and generated probabilities
    # statistic = lambda p: (p[:T_gen,0] < 0.2).sum()
    # from scipy.stats import skew, kurtosis
    # statistic = lambda p: (skew(p[:T_gen, 0])**2 + 1) / kurtosis(p[:T_gen, 0])

    # Duration of time spent above 0.2
    # T_gen = 3000
    # from pyhsmm.util.general import rle
    # def statistic(p):
    #     high = p[:T_gen,0] > 0.4
    #     hv, vlen = rle(high)
    #     # print(vlen[hv==True])
    #     return np.max(vlen[hv==True])
    #
    # run_ppc_on_probs(statistic, p_true, [lds, slds, roslds], T_gen=T_gen)
    #
    # T_gen = 3000
    # from pyhsmm.util.general import rle
    # def statistic(p):
    #     high = p[:T_gen,0] > 0.4
    #     hv, vlen = rle(high)
    #     # print(vlen[hv==True])
    #     return np.max(vlen[hv==True])
    #
    # run_ppc_on_probs(statistic, p_true, [lds, slds, roslds], N_iter=1000, T_gen=T_gen)


    plt.show()
