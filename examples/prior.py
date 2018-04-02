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
from rslds.models import PGRecurrentSLDS

### Global parameters
T, K, K_true, D_obs, D_latent = 200, 5, 5, 2, 2

results_dir = os.path.join("aux")

### Helper functions
def cached(results_name):
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

### Plotting code

def make_figure(true_model):
    """
    Show the following:
     - True dynamics distributions
     - True transition probabilities
    """
    fig = plt.figure(figsize=(6.5,2.75))
    gs = gridspec.GridSpec(2, K)

    fp = FontProperties()
    fp.set_weight("bold")

    # True dynamics
    for k in range(K):
        ax = fig.add_subplot(gs[0,k], aspect=1.0)
        plot_dynamics(true_model.dynamics_distns[k].A[:,:D_latent],
                      true_model.dynamics_distns[k].A[:,D_latent:],
                      k,
                      plot_center=True,
                      color=colors[k], ax=ax)

        ax = fig.add_subplot(gs[1, k], aspect=1.0)
        plot_single_trans_prob(true_model.trans_distn, k, ax=ax)

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "prior.png"), dpi=200)
    plt.savefig(os.path.join(results_dir, "prior.pdf"))

    plt.show()


def make_extended_figure(true_model, zs, xs):
    """
    Show the following:
     - True dynamics distributions
     - True transition probabilities
     - Superposition of most likely states
    """
    fig = plt.figure(figsize=(6.5,2.75))
    gs = gridspec.GridSpec(2, K+1)

    fp = FontProperties()
    fp.set_weight("bold")

    # True dynamics
    for k in range(K):
        ax = fig.add_subplot(gs[0,k], aspect=1.0)
        plot_dynamics(true_model.dynamics_distns[k].A[:,:D_latent],
                      true_model.dynamics_distns[k].A[:,D_latent:],
                      k,
                      plot_center=True,
                      color=colors[k], ax=ax)

        ax = fig.add_subplot(gs[1, k], aspect=1.0)
        plot_single_trans_prob(true_model.trans_distn, k, ax=ax)

    ax = fig.add_subplot(gs[0, -1], aspect=1.0)
    plot_most_likely_dynamics(true_model.trans_distn,
                              true_model.dynamics_distns,
                              ax=ax, nxpts=10, nypts=10)

    # Plot the trajectory
    ax = fig.add_subplot(gs[1, -1], aspect=1.0)
    for z, x in zip(zs, xs):
        plot_trajectory(z, x, ax=ax)

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "prior_with_sim.png"), dpi=200)
    plt.savefig(os.path.join(results_dir, "prior_with_sim.pdf"))

    plt.show()

def make_nonlinear_figure(true_model):
    """
    Show the following:
     - True dynamics distributions
     - True transition probabilities
    """
    fig = plt.figure(figsize=(6.5,2.75))
    gs = gridspec.GridSpec(2, K)

    fp = FontProperties()
    fp.set_weight("bold")

    # Override the transition distribution with a random neural network
    class DummyTransDistn:
        def __init__(self, K, D_in, n_rbfs=10):
            self.K, self.D_in = K, D_in

            self.cx, self.cy = np.meshgrid(np.linspace(-5, 5, n_rbfs),
                                           np.linspace(-5, 5, n_rbfs))
            self.cx, self.cy = np.ravel(self.cx), np.ravel(self.cy)

            # Compute distances between cells
            D = np.sqrt((self.cx[:, None] - self.cx[None, :])**2 +
                        (self.cy[:, None] - self.cy[None, :])**2)
            C = np.exp(-0.5 * D / 5.0) + 1e-3 * np.eye(D.shape[0])
            self.Ws = [npr.multivariate_normal(np.zeros(D.shape[0]), C, size=(K,))]
            self.Ws[0] -= self.Ws[0].mean(axis=1, keepdims=True)
            self.bs = [np.zeros(K)]
            # self.bs = [npr.randn(K)]

        def pi(self, xy):

            # Use RBFs for the hidden units
            activation = np.exp(-0.5 * ((xy[:,0][:,None] - self.cx[None,:])**2 +
                                        (xy[:,1][:,None] - self.cy[None,:])**2))

            activation = activation.dot(self.Ws[0].T) + self.bs[0]

            # Output is softmax scaled up to increase contrast
            from scipy.misc import logsumexp
            activation *= 3
            activation -= logsumexp(activation, axis=1, keepdims=True)
            pis = np.exp(activation)
            assert np.allclose(pis.sum(1), 1.0)
            return pis

    dummy_distn = DummyTransDistn(true_model.num_states, true_model.D_in)

    # True dynamics
    for k in range(K):
        ax = fig.add_subplot(gs[0,k], aspect=1.0)
        plot_dynamics(true_model.dynamics_distns[k].A[:,:D_latent],
                      true_model.dynamics_distns[k].A[:,D_latent:],
                      k,
                      plot_center=True,
                      color=colors[k], ax=ax)

        ax = fig.add_subplot(gs[1, k], aspect=1.0)
        plot_single_trans_prob(dummy_distn, k, ax=ax)

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "nn_prior.png"), dpi=200)
    plt.savefig(os.path.join(results_dir, "nn_prior.pdf"))

    plt.show()


def plot_dynamics(A, b, k, ax=None, plot_center=True,
                  xlim=(-5,5), ylim=(-5,5), npts=10,
                  color='r'):
    b = np.zeros((A.shape[0], 1)) if b is None else b
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


def plot_trajectory(z, x, ax=None, ls="-", xlim=(-5, 5), ylim=(-5, 5)):
    zcps = np.concatenate(([0], np.where(np.diff(z))[0] + 1, [z.size]))
    if ax is None:
        fig = plt.figure(figsize=(4, 4))
        ax = fig.gca()
    for start, stop in zip(zcps[:-1], zcps[1:]):
        ax.plot(x[start:stop + 1, 0],
                x[start:stop + 1, 1],
                lw=1, ls=ls,
                color=colors[z[start]],
                alpha=1.0)
        ax.plot(x[0,0], x[0,1], 'o', markersize=4, color=colors[z[0]])

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("$x_{t,1}$")
    ax.set_ylabel("$x_{t,2}$")
    ax.set_title("Sample Trajectories")

    return ax

### Manually construct a few interesting dynamical systems
def make_dynamics_distns():
    # Sample the fixed points on the unit disc
    # ths = 2 * np.pi * np.random.rand(K)
    ths = np.linspace(0, 2*np.pi, K+1)[:K]
    # ths = np.pi + np.array([np.pi/2., 0., 3*np.pi/2, 3*np.pi/2])
    # rs = 1 + np.random.rand(K)
    rs = 3 * np.ones(K)
    xs = rs * np.cos(ths)
    ys = rs * np.sin(ths)
    xstars = np.column_stack((xs, ys))

    # Sample stable dynamics matrices
    def _stable_dynamics():
        eigs = npr.rand(D_latent)
        X = np.random.randn(D_latent, D_latent)
        Q,_ = np.linalg.qr(X)
        A = Q.dot(np.diag(eigs)).dot(Q.T)
        return A

    As = [_stable_dynamics() for _ in range(K)]
    bs = [(A-np.eye(D_latent)).dot(x) for A, x in zip(As, xstars)]

    # Slow down the dynamics
    tau = 1. / 0.05
    As = [1. / tau * A + (1-1./tau) *np.eye(D_latent) for A in As]
    bs = [1. / tau * b for b in bs]
    # Abs = [0.01 * Ab for Ab in Abs]

    Abs = [np.column_stack((A,b)) for A, b in zip(As, bs)]

    # Make a recurrent SLDS with these params #
    dynamics_distns = [
        Regression(A=Ab, sigma=0.001 * np.eye(D_latent)) for Ab in Abs
        ]

    return dynamics_distns

### Make an example with 2D latent states and 4 discrete states
# @cached("simulated_data")
def simulate_prior():
    # Account for stick breaking asymmetry
    mu_b, _ = compute_psi_cmoments(np.ones(K_true))


    init_dynamics_distns = [
        Gaussian(
            mu=np.array([-0, -0]),
            sigma=5 * np.eye(D_latent))
        for _ in range(K)]

    dynamics_distns = make_dynamics_distns()

    emission_distns = \
        DiagonalRegression(D_obs, D_latent+1,
                           alpha_0=2.0, beta_0=2.0)

    model = PGRecurrentSLDS(
        trans_params=dict(sigmasq_A=10., sigmasq_b=0.01),
        init_state_distn='uniform',
        init_dynamics_distns=init_dynamics_distns,
        dynamics_distns=dynamics_distns,
        emission_distns=emission_distns,
        alpha=3.)


    # Print the true parameters
    np.set_printoptions(precision=2)
    print("True W_markov:\n{}".format(model.trans_distn.A[:,:K_true]))
    print("True W_input:\n{}".format(model.trans_distn.A[:,K_true:]))

    return model

def sample_model(model, x_init=None):

    #########################
    # Sample from the model #
    #########################
    inputs = np.ones((T, 1))
    if x_init is not None:
        for init_distn in model.init_dynamics_distns:
            init_distn.mu = x_init
            init_distn.sigma = 0.001 * np.eye(D_latent)

    ys, x, z = model.generate(T=T, inputs=inputs)
    y = ys[0]
    return inputs, z, x, y


if __name__ == "__main__":
    ## Simulate from the prior
    true_model = simulate_prior()

    N_samples = 10
    ths = np.linspace(0, 2*np.pi, N_samples+1)[:N_samples]
    zs = []
    xs = []
    for n in range(N_samples):
        x_init = np.array([4.5 * np.cos(ths[n]), 4.5 * np.sin(ths[n])])
        inputs, z, x, y = sample_model(true_model, x_init=x_init)
        zs.append(z[1:])
        xs.append(x[1:])

    make_figure(true_model)
    make_extended_figure(true_model, zs, xs)
    make_nonlinear_figure(true_model)
    plt.show()
