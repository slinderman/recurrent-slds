import os
import pickle

import numpy as np
import numpy.random as npr
npr.seed(1)

import matplotlib.pyplot as plt
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

from pybasicbayes.distributions import \
    Regression, Gaussian, DiagonalRegression

from rslds.util import compute_psi_cmoments
from rslds.models import TreeStructuredPGRecurrentSLDS
from rslds.dynamics import TreeStructuredHierarchicalDynamics
from pypolyagamma.binary_trees import balanced_binary_tree

### Global parameters
T, K, K_true, D_obs, D_latent = 200, 5, 5, 2, 2

results_dir = os.path.join("aux")

def plot_dynamics(Ab, k, ax=None, plot_center=True,
                  xlim=(-5,5), ylim=(-5,5), npts=10,
                  color='r'):
    A = Ab[:, :-1]
    b = Ab[:, -1]
    x = np.linspace(*xlim, npts)
    y = np.linspace(*ylim, npts)
    X,Y = np.meshgrid(x,y)
    xy = np.column_stack((X.ravel(), Y.ravel()))

    # dydt_m = xy.dot(A.T) + b.T - xy
    dydt_m = xy.dot(A.T) + b.T - xy

    if ax is None:
        fig = plt.figure(figsize=(6,6))
        ax = fig.add_subplot(111)

    ax.quiver(xy[:, 0], xy[:, 1],
              dydt_m[:, 0], dydt_m[:, 1],
              color=color, alpha=1.0,
              headwidth=5.)

    # Plot the stable point
    if plot_center:
        try:
            center = -np.linalg.solve(A-np.eye(D_latent), b)
            ax.plot(center[0], center[1], 'o', color=color, markersize=4)
        except:
            print("Dynamics are not invertible!")

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("$x_{t,1}$")
    ax.set_ylabel("$x_{t,2}$")
    ax.set_title("$A_{0}x_t + b_{0} - x_t$".format(k+1))

    return ax

def plot_single_trans_prob(trans_distn, k, ax=None,
                           xlim=(-5,5), ylim=(-5,5), n_pts=80):
    XX, YY = np.meshgrid(np.linspace(*xlim, n_pts),
                         np.linspace(*ylim, n_pts))
    XY = np.column_stack((np.ravel(XX), np.ravel(YY)))

    D_reg = trans_distn.D_in
    inputs = np.hstack((np.zeros((n_pts ** 2, D_reg - 2)), XY))
    test_prs = trans_distn.pi(inputs)

    if ax is None:
        fig = plt.figure(figsize=(12, 3))
        ax = fig.add_subplot(1, K, k + 1)

    cmap = gradient_cmap([np.ones(3), colors[k]])
    im1 = ax.imshow(test_prs[:, k].reshape(*XX.shape),
                    extent=xlim + tuple(reversed(ylim)),
                    vmin=0, vmax=1, cmap=cmap)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("$x_{t,1}$")
    ax.set_ylabel("$x_{t,2}$")
    ax.set_title("$\Pr(z_{{t+1}} = {0} \mid x_t)$".format(k + 1))


def plot_most_likely_dynamics(
        reg, dynamics_distns,
        xlim=(-5, 5), ylim=(-5, 5),  nxpts=20, nypts=20,
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
        if zk.sum() == 0:
            continue

        ax.quiver(xy[zk, 0], xy[zk, 1], dydt_m[zk, 0], dydt_m[zk, 1], color=colors[k],
                  headwidth=5.)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("$x_{t,1}$")
    ax.set_ylabel("$x_{t,2}$")
    ax.set_title("Superposition")

    return ax




if __name__ == "__main__":
    tree = balanced_binary_tree(4)

    # Make rSLDS parameters
    init_dynamics_distns = [
        Gaussian(
            mu=np.zeros(D_latent),
            sigma=np.eye(D_latent),
            nu_0=D_latent + 2, sigma_0=3. * np.eye(D_latent),
            mu_0=np.zeros(D_latent), kappa_0=1.0,
        )
        for _ in range(K)]

    # Create hierarchical dynamics model
    hierarchical_dynamics_distn = \
        TreeStructuredHierarchicalDynamics(tree, 0.1 * np.eye(D_latent), 3 * np.eye(D_latent + 1))

    emission_distns = \
        DiagonalRegression(D_obs, D_latent + 1,
                           alpha_0=2.0, beta_0=2.0)

    model = TreeStructuredPGRecurrentSLDS(
        trans_params=dict(sigmasq_A=10., sigmasq_b=0.01),
        init_state_distn='uniform',
        init_dynamics_distns=init_dynamics_distns,
        hierarchical_dynamics_distn=hierarchical_dynamics_distn,
        emission_distns=emission_distns)

    plt.figure(figsize=(12, 12))
    ax = plt.subplot(3, 4, 2)
    plot_dynamics(hierarchical_dynamics_distn.As[0], 0, ax=ax)
    ax = plt.subplot(3, 4, 4+1)
    plot_dynamics(hierarchical_dynamics_distn.As[1], 1, ax=ax)
    ax = plt.subplot(3, 4, 4 + 2)
    plot_dynamics(hierarchical_dynamics_distn.As[4], 4, ax=ax)
    ax = plt.subplot(3, 4, 8 + 1)
    plot_dynamics(hierarchical_dynamics_distn.As[2], 2, ax=ax)
    ax = plt.subplot(3, 4, 8 + 2)
    plot_dynamics(hierarchical_dynamics_distn.As[3], 3, ax=ax)
    ax = plt.subplot(3, 4, 8 + 3)
    plot_dynamics(hierarchical_dynamics_distn.As[5], 5, ax=ax)
    ax = plt.subplot(3, 4, 8 + 4)
    plot_dynamics(hierarchical_dynamics_distn.As[6], 6, ax=ax)
    plt.show()
