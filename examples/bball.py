import os
import copy
import pickle

import numpy as np
import numpy.random as npr
npr.seed(1)

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
from matplotlib.font_manager import FontProperties
import matplotlib.patches as patches
import mpl_toolkits.mplot3d.art3d as art3d

import seaborn as sns
color_names = ["red",
               "windows blue",
               "medium green",
               "dusty purple",
               "orange",
               "amber",
               "clay",
               "pink",
               "greyish",
               "light cyan",
               "steel blue",
               "forest green",
               "pastel purple",
               "mint",
               "salmon",
               "dark brown"]

colors = sns.xkcd_palette(color_names)
sns.set_style("white")
sns.set_context("paper")

from hips.plotting.colormaps import gradient_cmap
from hips.plotting.layout import create_axis_at_location, remove_plot_labels

from pybasicbayes.util.text import progprint_xrange
from pybasicbayes.models import FactorAnalysis
from pybasicbayes.distributions import \
    Regression, Gaussian, DiagonalRegression, AutoRegression

from pgmult.utils import compute_psi_cmoments
from pyslds.util import get_empirical_ar_params
from pyhsmm.util.general import relabel_by_permutation
from autoregressive.models import ARWeakLimitStickyHDPHMM, ARHMM

from pinkybrain.decision_list import DecisionList
from pinkybrain.distributions import MultinomialRegression
from pinkybrain.models import MixedEmissionHMMSLDS, MixedEmissionWeakLimitStickyHDPHMMSLDS
from pinkybrain.inhmm import RecurrentARHMM, InputOnlyRecurrentARHMM, StickyInputOnlyRecurrentARHMM
from pinkybrain.inslds import InputSLDS, StickyInputSLDS, InputOnlySLDS, StickyInputOnlySLDS

### Global parameters
K, D_obs, D_latent, N_lags = 30, 2, 4, 1

mask_start, mask_stop = 0, 0

players = ["LJames", "DWade", "CBosh", "MChalmers", "RAllen"]
extra_players = ["SBattier", "CAndersen", "NCole", "UHaslem", "RLewis"]
N = len(players)
runnum = 3
data_dir = os.path.join("experiments", "aistats", "bball")
results_dir = os.path.join("experiments", "aistats", "bball", "heat", "run{:03d}".format(runnum))

USE_CACHE = False

### Helper functions
def cached(results_name):
    if USE_CACHE:
        def _cache(func):
            def func_wrapper(*args, **kwargs):
                results_file = os.path.join(results_dir, results_name)
                if not results_file.endswith(".pkl"):
                    results_file += ".pkl"

                if os.path.exists(results_file):
                    with open(results_file, "rb") as f:
                        results = pickle.load(f)
                else:
                    assert os.path.exists(results_dir)
                    results = func(*args, **kwargs)
                    with open(results_file, "wb") as f:
                        pickle.dump(results, f)

                return results
            return func_wrapper
    else:
        _cache = lambda func: func
        
    return _cache

### Plotting code

def make_figure(rarhmm, datas, ids,
                zs, usage, k_to_plot,
                state_names=None,
                mu=0, sigma=1, p_lim=0.05,
                filename=None):
    """
    Show the following:
     - True latent dynamics (for most likely state)
     - Segment of trajectory in latent space
     - A few examples of observations in 10D space
     - ARHMM segmentation of factors
     - rSLDS segmentation of factors
     - ARHMM synthesis
     - rSLDS synthesis
    """
    reg = rarhmm.trans_distn
    As = [od.A[:,:-1] for od in rarhmm.obs_distns]
    bs = [od.A[:,-1] for od in rarhmm.obs_distns]

    fig = plt.figure(figsize=(6.6,2.6))
    M = len(k_to_plot)
    # gs = gridspec.GridSpec(4,M)

    fp = FontProperties()
    fp.set_weight("bold")

    # plt.figtext(.66 + .025, .5 - .075, '(f)', fontproperties=fp)
    paw, pah = 6.0 / M, 2.0
    axw, axh = 6.0 / M - 0.1, 1.7

    for i,k in enumerate(k_to_plot):
        # ax = fig.add_subplot(gs[:-1,i])
        ax = create_axis_at_location(fig, 0.3 + i*paw, 0.8, axw, axh, ticks=False)
        draw_halfcourt(ax)
        plot_state_trajectories(k, zs, datas, ids, ax=ax, color=colors[0], N_toplot=0)
        plot_dynamics_by_probability(reg, k, As[k], b=bs[k], mu=mu, sigma=sigma, ax=ax, npts=15, color=(0, 0, 0))
        if state_names is not None and state_names[i] is not None:
            ax.set_title(state_names[i], fontsize=9)
        else:
            ax.set_title("State {}".format(i+1), fontsize=9)

        # bax = fig.add_subplot(gs[-1,i])
        bax = create_axis_at_location(fig, 0.3 + i * paw, 0.4, axw, 0.4)
        handles = []
        labels = []
        for n in range(N):
            labels.append(players[n][0] + ". " + players[n][1:])
            handles.append(bax.bar(n, usage[n, k], width=0.8, color=colors[n]))
        bax.set_xlim(-0.2, N)
        # bax.set_xticks(np.arange(N) + 0.4)
        # bax.set_xticklabels(players, rotation="90")
        bax.set_xticks([])
        bax.set_ylim(0, p_lim)
        if i == 0:
            bax.set_yticks([0, p_lim])
        else:
            bax.set_yticks([])
        bax.set_title("State Usage", fontsize=9)


    # Create legend figure
    lax = create_axis_at_location(fig, 0.75, 0.05, 6.6-1.5, 0.2, ticks=False)
    # remove_plot_labels(lax)
    plt.legend(handles, labels, ncol=N, loc="center")

    if filename is not None:
        plt.savefig(os.path.join(results_dir, filename))

    # plt.tight_layout()
    if filename is None:
        plt.savefig(os.path.join(results_dir, "bball2.png"), dpi=200)
        plt.savefig(os.path.join(results_dir, "bball2.pdf"))
    else:
        plt.savefig(os.path.join(results_dir, filename + ".png"), dpi=200)
        plt.savefig(os.path.join(results_dir, filename + ".pdf"))
    # plt.show()


def draw_halfcourt(ax, is_3d = False, alpha=1.):
    """ take in a matplotlib axis, draw a halfcourt facing left on it """
    parts = {}
    parts['bounds'] = patches.Rectangle((0,0), 47, 50, fill=False, lw=1)

    #key,
    parts['outer_key']  = patches.Rectangle((0,17), 19, 16, fill=False, lw=1)
    parts['inner_key']  = patches.Rectangle((0,19), 19, 12, fill=False, color="grey", lw=1)
    parts['jump_circle']  = patches.Circle((19, 25), radius=6, fill=False, lw=1)
    parts['restricted'] = patches.Arc( (5.25, 25), 2*4, 2*4, theta1=-90, theta2=90, lw=1)
    parts['hoop']       = patches.Circle((5.25, 25), radius=.75, fill=False, lw=1)

    #midcourt circles
    parts['mid_circle']       = patches.Circle((47, 25), radius=8, fill=False, lw=1)
    parts['mid_small_circle'] = patches.Circle((47,25), radius=2, color="grey", lw=1)

    #3 point line
    break_angle = np.arccos( (25-35./12)/23.75 )      #angle between hoop->sideline and hoop->break
    break_angle_deg = break_angle / np.pi * 180
    break_length = 5.25 + 23.75*np.sin(break_angle)
    parts['arc'] = patches.Arc( (5.25, 25), 2*23.75, 2*23.75,
                                theta1 = -90+break_angle_deg, theta2 = 90-break_angle_deg, lw=1)
    parts['break0'] = patches.Rectangle((0, 35./12), break_length, 0, lw=1)
    parts['break1'] = patches.Rectangle((0, 50-35./12), break_length, 0, lw=1)

    #draw them
    for p in parts.values():
      p.set_alpha(alpha)
      ax.add_patch(p)
      if is_3d:
          art3d.pathpatch_2d_to_3d(p, z=0)

    #make sure court is drawn in proportion
    ax.set_xlim(-.1, 47.05)
    ax.set_ylim(-.1, 50.05)
    ax.set_aspect('equal')
    # ax.set_axis_off()
    # remove_plot_labels(ax)


def plot_state_trajectories(k, zs, datas, ids,
                            N_toplot=500,
                            ax=None, color='k',
                            filename=None):
    if ax is None:
        fig = plt.figure(figsize=(8,6))
        ax = fig.add_subplot(111)

    trajs = []
    traj_ids = []
    for z, data, id in zip(zs, datas, ids):
        zcps = np.concatenate(([0], np.where(np.diff(z))[0] + 1, [z.shape[0]]))
        for start, stop in zip(zcps[:-1], zcps[1:]):
            stop = min(stop+1, data.shape[0])
            if z[start] == k:
                if stop - start > 5:
                    trajs.append(data[start:stop])
                    traj_ids.append(id)

    # Plot a subset of traces
    if len(trajs) > N_toplot:
        toplot = npr.choice(len(trajs), N_toplot)
    else:
        toplot = np.arange(len(trajs))

    for i in toplot:
        ax.plot(trajs[i][:, 0],
                trajs[i][:, 1],
                lw=0.5, ls="-",
                color=colors[traj_ids[i]],
                alpha=0.25)

    if filename is not None:
        fig.savefig(os.path.join(results_dir, filename))


def plot_dynamics(A, b=None, ax=None, plot_center=False,
                  xlim=(-2, 2.), ylim=(-2, 2.), npts=20,
                  mu=0, sigma=1.0,
                  color='r',
                  filename=None):
    b = np.zeros((A.shape[0], 1)) if b is None else b
    x = np.linspace(*xlim, npts)
    y = np.linspace(*ylim, npts)
    X,Y = np.meshgrid(x,y)
    xy = np.column_stack((X.ravel(), Y.ravel()))

    # dydt_m = xy.dot(A.T) + b.T - xy
    xyp1 = xy.dot(A.T) + b.T
    uxy = unstandardize_data([xy], mu, sigma)[0]
    uxyp1 = unstandardize_data([xyp1], mu, sigma)[0]
    dxydt = uxyp1 - uxy

    if ax is None:
        fig = plt.figure(figsize=(6,6))
        ax = fig.add_subplot(111)


    # Plot the dynamics in "unstandardized" space
    ax.quiver(uxy[:, 0], uxy[:, 1],
              dxydt[:, 0], dxydt[:, 1],
              color=color, alpha=1.0,
              headwidth=5.)

    # Plot the stable point
    if plot_center:
        try:
            center = -np.linalg.solve(A-np.eye(D_latent), b)
            ax.plot(center[0], center[1], 'o', color=color, markersize=8)
        except:
            print("Dynamics are not invertible!")

    ax.set_xlabel('$x_1$', fontsize=12, labelpad=10)
    ax.set_ylabel('$x_2$', fontsize=12, labelpad=10)

    if filename is not None:
        fig.savefig(os.path.join(results_dir, filename))

    return ax


def plot_dynamics_by_probability(
        reg, k, A, b=None, ax=None, plot_center=False,
        xlim=(-2,2.), ylim=(-2,2.), npts=20,
        mu=0, sigma=1.0,
        color='r',
        filename=None):

    if ax is None:
        fig = plt.figure(figsize=(6,6))
        ax = fig.add_subplot(111)

    b = np.zeros((A.shape[0], 1)) if b is None else b
    x = np.linspace(*xlim, npts)
    y = np.linspace(*ylim, npts)
    X,Y = np.meshgrid(x,y)
    xy = np.column_stack((X.ravel(), Y.ravel()))

    # dydt_m = xy.dot(A.T) + b.T - xy
    xyp1 = xy.dot(A.T) + b.T
    uxy = unstandardize_data([xy], mu, sigma)[0]
    uxyp1 = unstandardize_data([xyp1], mu, sigma)[0]
    dxydt = uxyp1 - uxy

    # Compute the transition probabilities
    D_reg = reg.D_in
    inputs = np.hstack((np.zeros((npts ** 2, D_reg - 2)), xy))
    test_prs = reg.pi(inputs)

    # Convert color to RGBA
    assert isinstance(color, tuple) and len(color) == 3
    qcolor = np.zeros((xy.shape[0], 4))
    qcolor[:,:3] = color
    qcolor[:,-1] = 0.1 + 0.89 * test_prs[:,k] / test_prs[:,k].max()

    # Draw the court
    draw_halfcourt(ax)

    # Plot the dynamics in "unstandardized" space
    ax.quiver(uxy[:, 0], uxy[:, 1],
              dxydt[:, 0], dxydt[:, 1],
              color=qcolor,
              headwidth=5.,
              scale=5.0,
              )

    # ax.set_title("Max Pr: {0:.2f}".format(test_prs[:,k].max()))

    if filename is not None:
        plt.savefig(os.path.join(results_dir, filename))

    return ax

def plot_dynamics_and_trajectories(reg, k, zs, datas,
                                   A, usage, ids,
                                   b=None, mu=0, sigma=1,
                                   color='r', filename=None):
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111)
    draw_halfcourt(ax)
    plot_state_trajectories(k, zs, datas, ids, ax=ax, color=colors[0])
    plot_dynamics_by_probability(reg, k, A, b=b, mu=mu, sigma=sigma, ax=ax, color=(0,0,0))

    divider = make_axes_locatable(ax)
    bax = divider.append_axes("bottom", size="20%", pad=0.05)
    for n in range(N):
        bax.bar(n, usage[n], width=0.8, color=colors[n], label=players[n])
    bax.set_xlim(-0.2, N)
    bax.set_xticks(np.arange(N)+0.4)
    bax.set_xticklabels(players, rotation="vertical")
    bax.set_ylim(0, 0.1)

    if filename is not None:
        plt.savefig(os.path.join(results_dir, filename))


def plot_most_likely_dynamics(
        reg, dynamics_distns,
        xlim=(-4, 4), ylim=(-3, 3),  nxpts=20, nypts=10,
        alpha=0.8,
        ax=None, figsize=(3,3)):

    x = np.linspace(*xlim, nxpts)
    y = np.linspace(*ylim, nypts)
    X, Y = np.meshgrid(x, y)
    xy = np.column_stack((X.ravel(), Y.ravel()))

    # Get the probability of each state at each xy location
    inputs = np.hstack((np.zeros((nxpts * nypts, reg.D_in - 2)), xy))
    prs = reg.pi(inputs)
    z = np.argmax(prs, axis=1)


    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)

    for k in range(K):
        A = dynamics_distns[k].A[:, :D_latent]
        b = dynamics_distns[k].A[:, D_latent:]
        dydt_m = xy.dot(A.T) + b.T - xy

        zk = z == k
        ax.quiver(xy[zk, 0], xy[zk, 1],
                  dydt_m[zk, 0], dydt_m[zk, 1],
                  color=colors[k], alpha=alpha)

    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')

    plt.tight_layout()

    return ax


def plot_trans_probs(reg,
                     xlim=(-4,4), ylim=(-3,3), n_pts=50,
                     ax=None,
                     filename=None):
    XX,YY = np.meshgrid(np.linspace(*xlim,n_pts),
                        np.linspace(*ylim,n_pts))
    XY = np.column_stack((np.ravel(XX), np.ravel(YY)))

    D_reg = reg.D_in
    inputs = np.hstack((np.zeros((n_pts**2, D_reg-2)), XY))
    test_prs = reg.pi(inputs)

    if ax is None:
        fig = plt.figure(figsize=(10,6))
        ax = fig.add_subplot(111)

    for k in range(K):
        start = np.array([1., 1., 1., 0.])
        end = np.concatenate((colors[k], [0.5]))
        cmap = gradient_cmap([start, end])
        im1 = ax.imshow(test_prs[:,k].reshape(*XX.shape),
                         extent=xlim + tuple(reversed(ylim)),
                         vmin=0, vmax=1, cmap=cmap)

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        # ax.set_title("State {}".format(k+1))

    plt.tight_layout()
    return ax

def plot_trajectory(zhat, x, ax=None, ls="-", filename=None):
    zcps = np.concatenate(([0], np.where(np.diff(zhat))[0] + 1, [zhat.size]))
    if ax is None:
        fig = plt.figure(figsize=(4, 4))
        ax = fig.gca()
    for start, stop in zip(zcps[:-1], zcps[1:]):
        ax.plot(x[start:stop + 1, 0],
                x[start:stop + 1, 1],
                lw=1, ls=ls,
                color=colors[zhat[start]],
                alpha=1.0)

    # ax.set_xlabel('$x_1$', fontsize=12, labelpad=10)
    # ax.set_ylabel('$x_2$', fontsize=12, labelpad=10)
    if filename is not None:
        plt.savefig(filename)

    return ax

def plot_trajectory_and_probs(z, x,
                              ax=None,
                              trans_distn=None,
                              title=None,
                              filename=None,
                              **trargs):
    if ax is None:
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111)

    if trans_distn is not None:
        xlim = abs(x[:, 0]).max()
        xlim = (-xlim, xlim)
        ylim = abs(x[:, 0]).max()
        ylim = (-ylim, ylim)
        ax = plot_trans_probs(trans_distn, ax=ax,
                              xlim=xlim, ylim=ylim)
    plot_trajectory(z, x, ax=ax, **trargs)
    plt.tight_layout()
    plt.title(title)
    if filename is not None:
        plt.savefig(os.path.join(results_dir, filename))

    return ax


def plot_data(zhat, y, ax=None, ls="-", filename=None):
    zcps = np.concatenate(([0], np.where(np.diff(zhat))[0] + 1, [zhat.size]))
    if ax is None:
        fig = plt.figure(figsize=(4, 4))
        ax = fig.gca()
    for start, stop in zip(zcps[:-1], zcps[1:]):
        stop = min(y.shape[0], stop+1)
        ax.plot(np.arange(start, stop),
                y[start:stop ],
                lw=1, ls=ls,
                color=colors[zhat[start]],
                alpha=1.0)

    # ax.set_xlabel('$x_1$', fontsize=12, labelpad=10)
    # ax.set_ylabel('$x_2$', fontsize=12, labelpad=10)
    if filename is not None:
        plt.savefig(filename)

    return ax

def plot_separate_trans_probs(reg,
                              xlim=(-3,3), ylim=(-3,3), n_pts=100,
                              mu=0, sigma=1,
                              ax=None,
                              filename=None):
    XX,YY = np.meshgrid(np.linspace(*xlim,n_pts),
                        np.linspace(*ylim,n_pts))
    XY = np.column_stack((np.ravel(XX), np.ravel(YY)))

    D_reg = reg.D_in
    inputs = np.hstack((np.zeros((n_pts**2, D_reg-2)), XY))
    test_prs = reg.pi(inputs)

    if ax is None:
        fig = plt.figure(figsize=(12,3))

    bl = np.array([xlim[0], ylim[0]])
    ur = np.array([xlim[1], ylim[1]])
    ubl = unstandardize_data([bl], mu, sigma)[0]
    uur = unstandardize_data([ur], mu, sigma)[0]
    # import ipdb; ipdb.set_trace()

    for k in range(5):
        ax = fig.add_subplot(1,5,k+1)
        draw_halfcourt(ax)
        cmap = gradient_cmap([np.ones(3), colors[k]])
        im1 = ax.imshow(test_prs[:,k].reshape(*XX.shape),
                         # extent=xlim + tuple(reversed(ylim)),
                         extent=(ubl[0], uur[0], uur[1], ubl[1]),
                         vmin=0, vmax=1, cmap=cmap)

        # ax.set_xlim(ubl[0], uur[0])
        # ax.set_ylim(ubl[1], uur[1])

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im1, cax=cax, ax=ax)
        # ax.set_title("State {}".format(k+1))

    plt.tight_layout()
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

    im = ax.imshow(zs[:,slice(*plt_slice)], aspect='auto', vmin=0, vmax=K-1,
                     cmap=gradient_cmap(colors[:K]), interpolation="nearest",
                     extent=plt_slice + (N_iters, 0))
    # ax.autoscale(False)
    ax.set_xticks([])
    # ax.set_yticks([0, N_iters])
    ax.set_ylabel("Iteration")

    if zref is not None:
        divider = make_axes_locatable(ax)
        ax2 = divider.append_axes("bottom", size="10%", pad=0.05)

        zref = np.atleast_2d(zref)
        im = ax2.imshow(zref[:, slice(*plt_slice)], aspect='auto', vmin=0, vmax=K-1,
                         cmap=gradient_cmap(colors[:K]), interpolation="nearest")
        # ax2.autoscale(False)
        ax.set_xticks([])
        ax2.set_yticks([])
        ax2.set_ylabel("True $z$", rotation=0)
        ax2.yaxis.set_label_coords(-.15, -.5)
        ax2.set_xlabel("Time")

    if title is not None:
        ax.set_title(title)

    if filename is not None:
        plt.savefig(os.path.join(results_dir, filename))

def standardize_data(datas):
    # all_datas = np.vstack(datas)
    # m, s = all_datas.mean(0), all_datas.std(0)
    m = np.array([23.5, 25.])
    s = np.array([47./4, 50./4.])

    std_datas = [(d-m)/s for d in datas]
    return std_datas, m, s

def unstandardize_data(std_datas, m, s):
    datas = [d*s + m for d in std_datas]
    return datas

### Fit an AR model as a baseline
@cached("ar")
def fit_ar(xs, affine=True, N_iter=100):
    assert isinstance(xs, list)

    print("Fitting AR")
    dynamics_hypparams = \
        dict(nu_0=D_obs + N_lags + 1,
             S_0=np.eye(D_obs),
             M_0=np.hstack((np.zeros((D_obs, (N_lags-1)*D_obs)),
                            0.99*np.eye(D_obs),
                            np.zeros((D_obs, int(affine))))),
             K_0=np.eye(D_obs*N_lags + affine),
             affine=affine)
    dynamics_hypparams = get_empirical_ar_params(xs, dynamics_hypparams)

    dynamics_distns = [
        AutoRegression(
            A=np.column_stack((np.zeros((D_obs, (N_lags-1)*D_obs)),
                               0.99 * np.eye(D_obs),
                               np.zeros((D_obs, int(affine))))),
            sigma=np.eye(D_obs),
            **dynamics_hypparams)]

    init_distn = Gaussian(nu_0=D_obs + 2,
                          sigma_0=np.eye(D_obs),
                          mu_0=np.zeros(D_obs),
                          kappa_0=1.0)

    ar = ARHMM(
        init_state_distn='uniform',
        init_emission_distn=init_distn,
        obs_distns=dynamics_distns,
        alpha=3.0)

    for x in xs:
        ar.add_data(x)

    lps = []
    for _ in progprint_xrange(N_iter, perline=1):
        ar.resample_model()
        lps.append(ar.log_likelihood())

    z_init = ar.stateseqs

    return ar, z_init, lps

### Make an ARHMM for initialization
@cached("arhmm")
def fit_arhmm(xs, affine=True, N_iter=1000):
    assert isinstance(xs, list)

    print("Fitting Sticky ARHMM")
    dynamics_hypparams = \
        dict(nu_0=D_obs + N_lags + 1,
             S_0=np.eye(D_obs),
             M_0=np.hstack((np.zeros((D_obs, (N_lags-1)*D_obs)),
                            0.99*np.eye(D_obs),
                            np.zeros((D_obs, int(affine))))),
             K_0=np.eye(D_obs*N_lags + affine),
             affine=affine)
    dynamics_hypparams = get_empirical_ar_params(xs, dynamics_hypparams)

    dynamics_distns = [
        AutoRegression(
            A=np.column_stack((np.zeros((D_obs, (N_lags-1)*D_obs)),
                               0.99 * np.eye(D_obs),
                               np.zeros((D_obs, int(affine))))),
            sigma=np.eye(D_obs),
            **dynamics_hypparams)
        for _ in range(K)]

    init_distn = Gaussian(nu_0=D_obs + 2,
                          sigma_0=np.eye(D_obs),
                          mu_0=np.zeros(D_obs),
                          kappa_0=1.0)

    arhmm = ARWeakLimitStickyHDPHMM(
        init_state_distn='uniform',
        init_emission_distn=init_distn,
        obs_distns=dynamics_distns,
        alpha=3.0, kappa=100.0, gamma=3.0)

    for x in xs:
        arhmm.add_data(x)

    lps = []
    for _ in progprint_xrange(N_iter, perline=1):
        arhmm.resample_model()
        lps.append(arhmm.log_likelihood())

    z_init = arhmm.stateseqs

    return arhmm, z_init, lps

### Make an ARHMM for initialization
@cached("rarhmm")
def fit_rarhmm(xs, affine=True, N_iter=1000, arhmm=None, z_inits=None, P_init=None):
    assert isinstance(xs, list)

    print("Fitting Recurrent ARHMM")
    dynamics_hypparams = \
        dict(nu_0=D_obs + N_lags + 1,
             S_0=np.eye(D_obs),
             M_0=np.hstack((np.zeros((D_obs, (N_lags-1)*D_obs)),
                            0.99*np.eye(D_obs),
                            np.zeros((D_obs, int(affine))))),
             K_0=np.eye(D_obs*N_lags + affine),
             affine=affine)
    dynamics_hypparams = get_empirical_ar_params(xs, dynamics_hypparams)

    dynamics_distns = [
        AutoRegression(
            A=np.column_stack((np.zeros((D_obs, (N_lags-1)*D_obs)),
                               0.99 * np.eye(D_obs),
                               np.zeros((D_obs, int(affine))))),
            sigma=np.eye(D_obs),
            **dynamics_hypparams)
        for _ in range(K)]


    rarhmm = RecurrentARHMM(
        D_in=D_obs,
        init_state_distn='uniform',
        obs_distns=dynamics_distns,
        alpha=3.0,
        trans_params=dict(sigmasq_A=100., sigmasq_b=100.))

    if arhmm is not None:
        from pgmult.utils import pi_to_psi
        Psi_init = np.array([pi_to_psi(p) for p in arhmm.trans_distn.trans_matrix])
        # Transpose so that Psi_init is a K-1 x K matrix
        Psi_init = Psi_init.T

        # Subtract the mean
        b_init = Psi_init.mean(axis=1)
        Psi_init -= b_init[:, None]

        # Concatenate two more dimensions for the inputs
        Psi_init = np.column_stack((Psi_init, np.zeros((K-1, D_obs))))

        # Plug it in
        assert rarhmm.trans_distn.A.shape == Psi_init.shape
        rarhmm.trans_distn.A = Psi_init.copy()
        rarhmm.trans_distn.b = b_init.copy()


        # Set the observation distributions
        for i in range(K):
            rarhmm.obs_distns[i] = copy.deepcopy(arhmm.obs_distns[i])

        # Initialize with arhmm sequences
        for x, z in zip(xs, arhmm.stateseqs):
            rarhmm.add_data(x)
            rarhmm.states_list[-1].stateseq = z

    else:
        for x in xs:
            rarhmm.add_data(x)

    lps = []
    print("lp: ", rarhmm.log_likelihood())
    for _ in progprint_xrange(N_iter):
        rarhmm.resample_model()
        lps.append(rarhmm.log_likelihood())
        print("lp: ", lps[-1])

    z_init = rarhmm.stateseqs

    return rarhmm, z_init, lps


### Make an ARHMM for initialization
@cached("iorarhmm")
def fit_iorarhmm(xs, affine=True, N_iter=1000):
    assert isinstance(xs, list)

    print("Fitting Recurrent ARHMM")
    dynamics_hypparams = \
        dict(nu_0=D_obs + N_lags + 1,
             S_0=np.eye(D_obs),
             M_0=np.hstack((np.zeros((D_obs, (N_lags-1)*D_obs)),
                            0.99*np.eye(D_obs),
                            np.zeros((D_obs, int(affine))))),
             K_0=np.eye(D_obs*N_lags + affine),
             affine=affine)
    dynamics_hypparams = get_empirical_ar_params(xs, dynamics_hypparams)

    dynamics_distns = [
        AutoRegression(
            A=np.column_stack((np.zeros((D_obs, (N_lags-1)*D_obs)),
                               0.99 * np.eye(D_obs),
                               np.zeros((D_obs, int(affine))))),
            sigma=np.eye(D_obs),
            **dynamics_hypparams)
        for _ in range(K)]


    rarhmm = InputOnlyRecurrentARHMM(
        D_in=D_obs,
        init_state_distn='uniform',
        obs_distns=dynamics_distns,
        alpha=3.0)

    for x in xs:
        rarhmm.add_data(x)

    lps = []
    for _ in progprint_xrange(N_iter):
        rarhmm.resample_model()
        lps.append(rarhmm.log_likelihood())

    z_init = rarhmm.stateseqs

    return rarhmm, z_init, lps


### Make an ARHMM for initialization
@cached("siorarhmm")
def fit_sticky_iorarhmm(xs, affine=True, kappa=1.0, N_iter=1000, z_inits=None):
    assert isinstance(xs, list)

    print("Fitting Sticky Input Only Recurrent ARHMM")
    dynamics_hypparams = \
        dict(nu_0=D_obs + N_lags + 1,
             S_0=np.eye(D_obs),
             M_0=np.hstack((np.zeros((D_obs, (N_lags-1)*D_obs)),
                            0.99*np.eye(D_obs),
                            np.zeros((D_obs, int(affine))))),
             K_0=np.eye(D_obs*N_lags + affine),
             affine=affine)
    dynamics_hypparams = get_empirical_ar_params(xs, dynamics_hypparams)

    dynamics_distns = [
        AutoRegression(
            A=np.column_stack((np.zeros((D_obs, (N_lags-1)*D_obs)),
                               0.99 * np.eye(D_obs),
                               np.zeros((D_obs, int(affine))))),
            sigma=np.eye(D_obs),
            **dynamics_hypparams)
        for _ in range(K)]


    rarhmm = StickyInputOnlyRecurrentARHMM(
        D_in=D_obs,
        trans_params=dict(kappa=kappa),
        init_state_distn='uniform',
        obs_distns=dynamics_distns,
        alpha=3.0)

    if z_inits is None:
        for x in xs:
            rarhmm.add_data(x)
    else:
        for x, z in zip(xs, z_inits):
            rarhmm.add_data(x)
            rarhmm.states_list[-1].stateseq = z

    # Initialize the dynamics parameters
    for _ in progprint_xrange(100):
        rarhmm.resample_obs_distns()

    for _ in progprint_xrange(10):
        rarhmm.resample_trans_distn()

    lps = []
    print("lp: ", rarhmm.log_likelihood())
    for _ in progprint_xrange(N_iter, perline=1):
        rarhmm.resample_model()
        lps.append(rarhmm.log_likelihood())
        print("lp: ", lps[-1])

    z_init = rarhmm.stateseqs

    return rarhmm, z_init, lps


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

    emission_distns = [
        DiagonalRegression(D_obs, D_latent + 1,
                           A=C_init, sigmasq=np.ones(D_obs),
                           alpha_0=2.0, beta_0=2.0)]

    return init_dynamics_distns, dynamics_distns, emission_distns


@cached("slds")
def fit_slds(ys, inputss, masks,
             z_init=None, x_init=None, C_init=None,
             N_iters=1000):
    print("Fitting standard SLDS")
    init_dynamics_distns, dynamics_distns, emission_distns = \
        make_rslds_parameters(C_init)

    # slds = MixedEmissionHMMSLDS(
    slds = MixedEmissionWeakLimitStickyHDPHMMSLDS(
        init_state_distn='uniform',
        init_dynamics_distns=init_dynamics_distns,
        dynamics_distns=dynamics_distns,
        emission_distns=emission_distns,
        alpha=3., gamma=3., kappa=100.)

    for y, inputs, mask in zip(ys, inputss, masks):
        slds.add_data(y, inputs=inputs, mask=mask)

    # Initialize states
    if z_init is not None:
        slds.states_list[0].stateseq = z_init.copy().astype(np.int32)

    if x_init is not None:
        slds.states_list[0].gaussian_states = x_init.copy()

        # Initialize dynamics
        print("Initializing dynamics with Gibbs sampling")
        for _ in progprint_xrange(100):
            slds.resample_dynamics_distns()

    else:
        for states in slds.states_list:
            states.gaussian_states *= 0

    # Fit the model
    lps = []
    z_smpls = []
    for _ in progprint_xrange(N_iters):
        slds.resample_model()
        lps.append(slds.log_likelihood())
        z_smpls.append(slds.stateseqs[0].copy())

    x_test = slds.states_list[0].gaussian_states
    z_smpls = np.array(z_smpls)
    lps = np.array(lps)

    return slds, lps, z_smpls, x_test


if __name__ == "__main__":
    ## Load a player's trajectories
    datas = []
    player_ids = []
    for i,player in enumerate(players):
        print("Loading data for {}".format(player))
        with open(os.path.join(data_dir, player + ".pkl"), "rb") as f:
            player_trajectories = pickle.load(f)
            datas.extend(player_trajectories)
            player_ids = np.concatenate((player_ids, i * np.ones(len(player_trajectories))))
    player_ids = player_ids.astype(int)
    T_train = sum([d.shape[0] for d in datas])
    print("T_train: ", T_train)

    test_datas = []
    test_player_ids = []
    for i, player in enumerate(extra_players[3:4]):
        print("Loading data for {}".format(player))
        with open(os.path.join(data_dir, player + ".pkl"), "rb") as f:
            player_trajectories = pickle.load(f)
            test_datas.extend(player_trajectories)
            test_player_ids = np.concatenate((test_player_ids, i * np.ones(len(player_trajectories))))
    test_player_ids = test_player_ids.astype(int)
    T_test = np.sum([d.shape[0] for d in test_datas])
    print("T_test: ", T_test)


    # Standardize the data
    std_datas, mu, sigma = standardize_data(datas)
    std_test_datas = [(d-mu)/sigma for d in test_datas]

    # Fit an AR (1 state) as a baseline
    ar, _, ar_lps = fit_ar(std_datas, N_iter=200)

    ## Fit an ARHMM for initialization
    arhmm, z_init, arhmm_lps = fit_arhmm(std_datas, N_iter=200)
    rarhmm, _, rarhmm_lps = fit_rarhmm(std_datas, N_iter=200, arhmm=arhmm)
    # ioarhmm, _, ioarhmm_lps = fit_iorarhmm(std_datas, N_iter=200)
    # siorarhmm, _, siorarhmm_lps = fit_sticky_iorarhmm(std_datas, N_iter=200, z_inits=arhmm.stateseqs)

    # Compute test likelihoods
    ar_test_lp = np.sum([ar.log_likelihood(d) for d in std_test_datas])
    arhmm_test_lp = np.sum([arhmm.log_likelihood(d) for d in std_test_datas])
    rarhmm_test_lp = np.sum([rarhmm.log_likelihood(d) for d in std_test_datas])

    # Print likelihoods
    print("Training likelihoods:")
    print("ar lp:     ", ar_lps[-1] / T_train)
    print("arhmm lp:  ", arhmm_lps[-1] / T_train)
    print("rarhmm lp: ", rarhmm_lps[-1] / T_train)
    print("")
    print("Testing likelihoods:")
    print("ar lp:     ", ar_test_lp / T_test)
    print("arhmm lp:  ", arhmm_test_lp / T_test)
    print("rarhmm lp: ", rarhmm_test_lp / T_test)
    print("imprvmnt:  ", (rarhmm_test_lp - arhmm_test_lp) / (arhmm_test_lp - ar_test_lp))

    # Get state permutation
    perm = np.argsort(-rarhmm.state_usages)

    # Get per player usages
    player_usages = np.zeros((N, K))
    for i,player in enumerate(players):
        for j in range(len(datas)):
            if player_ids[j] == i:
                player_usages[i] += np.bincount(rarhmm.stateseqs[j], minlength=K)

    player_usages /= player_usages.sum(1)[:,None]
    # player_usages = player_usages[:, perm]

    # plt.figure(figsize=(8,6))
    # for n in range(N):
    #     plt.bar(np.arange(K)+n/(N+1.0), player_usages[n], width=1./(N+1), color=colors[n], label=players[n])
    # plt.legend()
    # plt.xticks(np.arange(K))
    # plt.savefig(os.path.join(results_dir, "usage_comparison.png"))

    # Get the top 5 states
    lagged_datas = [d[N_lags:] for d in datas]
    # for ind, k in enumerate(perm):
    #     if rarhmm.state_usages[k] < 1e-3:
    #         continue
    #
    #     od = rarhmm.obs_distns[k]
    #     A, b = od.A[:,:-1], od.A[:,-1]
    #     plot_dynamics_and_trajectories(rarhmm.trans_distn, k, rarhmm.stateseqs, lagged_datas,
    #                                    A, player_usages[:,k], player_ids, b=b, mu=mu, sigma=sigma,
    #                                    color=colors[k % len(colors)],
    #                                    filename="rarhmm_dynamics_{}.png".format(ind))
    #
    #     plt.close("all")
    #     # plot_dynamics_by_probability(arhmm.trans_distn, k,
    #     #                              A, b=b, mu=mu, sigma=sigma,
    #     #                              color=colors[k % len(colors)],
    #     #                              filename="arhmm_dynamics_prs_{}.png".format(k))


    k_to_plot = perm[np.array([15, 18, 23, 24, 25])]
    state_names = ["\"Run to Left Corner\"",
                   "\"Run to Right Corner\"",
                   "\"Cut to Basket\"",
                   "\"Cut Along 3pt Line\"",
                   "\"Baseline Drive\""]
    # make_figure(rarhmm, lagged_datas, player_ids,
    #             rarhmm.stateseqs,
    #             player_usages,
    #             k_to_plot,
    #             state_names=state_names,
    #             mu=mu, sigma=sigma)


    # Make the rest of the plots
    all_state_names = ["State {}".format(k+1) for k in range(K)]
    # for i,k in enumerate(k_to_plot):
    #     all_state_names[k] = state_names[i]

    # Hand label a few more
    # all_state_names[1] = "\"Run!\""
    # all_state_names[2] = "\"Inch Right\""
    # all_state_names[3] = "\"Work Left\""
    # all_state_names[5] = "\"Stand Still\""
    # all_state_names[6] = "\"Clear Left\""
    # all_state_names[9] = "\"Cross L. to R.\""
    # all_state_names[13] = "\"Inch Left\""
    # all_state_names[15] = "\"Clear Center\""
    # all_state_names[22] = "\"Stand Still\""
    # all_state_names[24] = "\"Run to Center\""
    # all_state_names[26] = "\"Cut to Basket\""



    # for start in range(0,K,5):
    #     make_figure(rarhmm, lagged_datas, player_ids,
    #                 rarhmm.stateseqs,
    #                 player_usages,
    #                 np.arange(start, start+5),
    #                 state_names=all_state_names[start:start+5],
    #                 mu=mu, sigma=sigma, p_lim=0.1,
    #                 filename="bball_extra_{0}-{1}".format(start,start+5))
    plt.show()