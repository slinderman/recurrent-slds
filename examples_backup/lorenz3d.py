import os
import pickle

import numpy as np
import numpy.random as npr
npr.seed(0)

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
from matplotlib.font_manager import FontProperties
from mpl_toolkits.mplot3d import Axes3D

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

from pybasicbayes.util.text import progprint_xrange
from pybasicbayes.models import FactorAnalysis
from pybasicbayes.distributions import \
    Regression, Gaussian, DiagonalRegression, AutoRegression

from pyslds.util import get_empirical_ar_params
from pyhsmm.util.general import relabel_by_permutation
from autoregressive.models import ARWeakLimitStickyHDPHMM

from pinkybrain.decision_list import DecisionList
from pinkybrain.distributions import MultinomialRegression
from pinkybrain.models import MixedEmissionHMMSLDS
from pinkybrain.inslds import InputSLDS, StickyInputSLDS, InputOnlySLDS, StickyInputOnlySLDS

### Global parameters
T, K, D_obs, D_latent = 10000, 2, 3, 3
etasq = 0.1**2
mask_start, mask_stop = 0, 0

runnum = 3
results_dir = os.path.join("experiments", "aistats", "lorenz3d", "run{:03d}".format(runnum))

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

def make_figure(y, x_true,
                rslds, zs_rslds, x_rslds,
                z_rslds_gen, x_rslds_gen, y_rslds_gen,
                slds, zs_slds, x_slds,
                z_slds_gen, x_slds_gen, y_slds_gen):
    """
    Show the following:
     - True data
    """
    fig = plt.figure(figsize=(6.5,3.5))
    gs = gridspec.GridSpec(6,3)

    fp = FontProperties()
    fp.set_weight("bold")

    # Observed data 3d
    ax1 = fig.add_subplot(gs[:3,0], projection="3d")
    plot_data(y[:2500], ax=ax1)
    ax1.set_title("Observed Data")
    plt.figtext(.025, 1-.075, '(a)', fontproperties=fp)

    # Plot the output vs time
    ax2 = fig.add_subplot(gs[3:, 0])
    ax2.plot(y[:2500, 0], color=0.25 * np.ones(3), lw=1, ls="-", label="$y_1$")
    ax2.plot([0,2500], [0,0], ':k')
    ax2.plot(-3 + y[:2500, 1], color=0.50 * np.ones(3), lw=1, ls="-", label="$y_1$")
    ax2.plot([0,2500], [-3,-3], ':k')
    ax2.plot(-6.0 + y[:2500, 2], color=0.75 * np.ones(3), lw=1, ls="-", label="$y_3$")
    ax2.plot([0,2500], [-6,-6], ':k')
    ax2.set_ylim(-7.5, 1.5)
    ax2.set_yticks([0, -3, -6])
    ax2.set_yticklabels(["$y_1$", "$y_2$", "$y_3$"])
    ax2.set_xlabel("Time")
    ax2.set_title("Observed Data")
    # plot_changepoints(y, zs_rslds, ax=ax2)
    plt.figtext(.025, .6 - .075, '(b)', fontproperties=fp)

    # Plot the inferred dynamics under the rSLDS
    ax3 = fig.add_subplot(gs[:3, 1], projection="3d")
    yinds = np.random.choice(T, size=400, replace=False)
    plot_most_likely_dynamics3d(rslds.trans_distn,
                                rslds.dynamics_distns,
                                x_true[yinds],
                                ax=ax3, alpha=0.5)

    # Overlay a partial trajectory
    ax3.set_title("Inferred Dynamics (rSLDS)")
    plt.figtext(.3 + .025, 1. - .075, '(c)', fontproperties=fp)

    # Plot something... z samples?
    ax4 = fig.add_subplot(gs[3:,1])
    plot_z_samples(zs_rslds, plt_slice=(0,2500), ax=ax4)
    ax4.set_title("Inferred States (rSLDS)")
    plt.figtext(.3 + .025, .6 - .075, '(d)', fontproperties=fp)

    # Plot simulated SLDS data
    ax5 = fig.add_subplot(gs[:3, 2], projection="3d")
    plot_data(y_slds_gen[1:1000], z=z_slds_gen[1:1000], ax=ax5)
    ax5.set_title("Generated Data (SLDS)")
    plt.figtext(.66 + .025, 1. - .075, '(e)', fontproperties=fp)

    # Plot simulated rSLDS data
    ax6 = fig.add_subplot(gs[3:, 2], projection="3d")
    plot_data(y_rslds_gen[1:], z=z_rslds_gen[1:], ax=ax6)
    ax6.set_title("Generated Data (rSLDS)")
    plt.figtext(.66 + .025, .6 - .075, '(f)', fontproperties=fp)

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "lorenz.png"), dpi=200)
    plt.savefig(os.path.join(results_dir, "lorenz.pdf"))
    plt.show()

def plot_most_likely_dynamics(
        reg, dynamics_distns,
        xyz=None,
        xlim=(-4, 4), ylim=(-3, 3),
        nxpts=20, nypts=10,
        alpha=0.8,
        ax=None, figsize=(3,3)):

    if xyz is None:
        x = np.linspace(*xlim, nxpts)
        y = np.linspace(*ylim, nypts)
        X, Y = np.meshgrid(x, y)
        xyz = np.column_stack((X.ravel(), Y.ravel(), np.zeros(nxpts*nypts)))

    # Get the probability of each state at each xy location
    inputs = np.hstack((np.zeros((xyz.shape[0], reg.D_in - D_latent)), xyz))
    prs = reg.pi(inputs)
    z = np.argmax(prs, axis=1)


    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)

    for k in range(K):
        A = dynamics_distns[k].A[:, :D_latent]
        b = dynamics_distns[k].A[:, D_latent:]
        dydt_m = xyz.dot(A.T) + b.T - xyz

        zk = z == k
        ax.quiver(xyz[zk, 0], xyz[zk, 1],
                  dydt_m[zk, 0], dydt_m[zk, 1],
                  color=colors[k], alpha=alpha)

    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')

    return ax

def plot_most_likely_dynamics3d(
        reg, dynamics_distns, xyz,
        alpha=0.5, length=0.5,
        ax=None, figsize=(3,3)):
    # Get the probability of each state at each xy location
    inputs = np.hstack((np.zeros((xyz.shape[0], reg.D_in - D_latent)), xyz))
    prs = reg.pi(inputs)
    z = np.argmax(prs, axis=1)


    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)

    for k in range(K):
        A = dynamics_distns[k].A[:, :D_latent]
        b = dynamics_distns[k].A[:, D_latent:]
        dydt_m = xyz.dot(A.T) + b.T - xyz

        zk = z == k
        ax.quiver(xyz[zk, 0], xyz[zk, 1], xyz[zk, 2],
                  dydt_m[zk, 0], dydt_m[zk, 1], dydt_m[zk, 2],
                  color=colors[k], alpha=alpha, length=length)

    ax.w_xaxis.set_pane_color(.95 * np.ones(3))
    ax.w_yaxis.set_pane_color(.95 * np.ones(3))
    ax.w_zaxis.set_pane_color(.95 * np.ones(3))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.set_xlabel('$y_1$', labelpad=-15)
    ax.set_ylabel('$y_2$', labelpad=-15)
    ax.set_zlabel('$y_3$', labelpad=-15)

    return ax

def plot_changepoints(y, z_smpls, ax=None):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)

    ax.plot(y[:2500, 0], color=0.25 * np.ones(3), lw=1, ls="-", label="$y_1$")
    ax.plot([0, 2500], [0, 0], ':k')
    ax.plot(-3 + y[:2500, 1], color=0.50 * np.ones(3), lw=1, ls="-", label="$y_1$")
    ax.plot([0, 2500], [-3, -3], ':k')
    ax.plot(-6.0 + y[:2500, 2], color=0.75 * np.ones(3), lw=1, ls="-", label="$y_3$")
    ax.plot([0, 2500], [-6, -6], ':k')
    ax.set_xticklabels([])
    ax.set_ylim(-7.5, 1.5)
    ax.set_yticks([0, -3, -6])
    ax.set_yticklabels(["$y_1$", "$y_2$", "$y_3$"])

    # Plot change point probability
    zcps = abs(np.diff(z_smpls, axis=1)) > 0
    divider = make_axes_locatable(ax)
    ax2 = divider.append_axes("bottom", size="10%", pad=0.05)

    pr_cp = np.mean(zcps[-500:], axis=0)
    # Smooth
    from scipy.ndimage.filters import gaussian_filter1d
    pr_cp_smooth = gaussian_filter1d(pr_cp, sigma=3)

    ax2.plot(np.arange(1, 2500), pr_cp_smooth[:2499], lw=1, color=colors[0])

    ax.set_xticks([])
    ax2.set_ylim(0,1)
    ax2.set_yticks([0,1])
    ax2.set_ylabel("$\Pr(\mathrm{cp})$", rotation=0)
    ax2.yaxis.set_label_coords(-.25, -.5)
    ax2.set_xlabel("Time")

def plot_data(y, z=None, ax=None, ls="-", filename=None):
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
    ax.set_xlabel('$y_1$', labelpad=-15)
    ax.set_ylabel('$y_2$', labelpad=-15)
    ax.set_zlabel('$y_3$', labelpad=-15)

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
        plt.savefig(os.path.join(results_dir, filename))

### Make an example with 2D latent states and 4 discrete states
@cached("simulated_lorenz")
def simulate_lorenz():
    from pinkybrain.util import lorenz
    x = lorenz(T)

    if D_latent == D_obs:
        C = np.eye(D_latent)
    else:
        C = np.random.randn(D_obs, D_latent)
    y = x.dot(C.T) + np.sqrt(etasq) * np.random.randn(T, D_obs)

    mask = np.ones((T, D_obs), dtype=bool)
    return x, y, C, mask

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
    model = PCA(n_components=D_latent, whiten=whiten)
    x_init = model.fit_transform(y)
    C_init = model.components_.T
    b_init = model.mean_[:,None]
    sigma = np.sqrt(model.explained_variance_)

    # inverse transform is given by
    # X.dot(sigma * C_init.T) + b_init.T
    if whiten:
        C_init = sigma * C_init

    return x_init, np.column_stack((C_init, b_init))

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

    emission_distns = [
        DiagonalRegression(D_obs, D_latent + 1,
                           A=C_init.copy(), sigmasq=np.ones(D_obs),
                           alpha_0=2.0, beta_0=2.0)]

    return init_dynamics_distns, dynamics_distns, emission_distns


@cached("slds")
def fit_slds(inputs, z_init, x_init, y, mask, C_init,
              N_iters=10000):
    print("Fitting standard SLDS")
    init_dynamics_distns, dynamics_distns, emission_distns = \
        make_rslds_parameters(C_init)

    slds = MixedEmissionHMMSLDS(
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

@cached("rslds")
def fit_rslds(inputs, z_init, x_init, y, mask, dl_reg, C_init,
              N_iters=10000):
    print("Fitting rSLDS")
    init_dynamics_distns, dynamics_distns, emission_distns = \
        make_rslds_parameters(C_init)

    rslds = InputSLDS(
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

    # Fit the model
    lps = []
    z_smpls = []
    for _ in progprint_xrange(N_iters):
        rslds.resample_model()
        lps.append(rslds.log_likelihood())
        z_smpls.append(rslds.stateseqs[0].copy())

    x_test = rslds.states_list[0].gaussian_states
    z_smpls = np.array(z_smpls)
    lps = np.array(lps)

    print("Inf W_markov:\n{}".format(rslds.trans_distn.A[:, :K]))
    print("Inf W_input:\n{}".format(rslds.trans_distn.A[:, K:]))

    return rslds, lps, z_smpls, x_test

@cached("inputonly_rslds")
def fit_inputonly_rslds(inputs, z_init, x_init, y, mask, dl_reg, C_init,
                        N_iters=10000):
    print("Fitting input only rSLDS")
    init_dynamics_distns, dynamics_distns, emission_distns = \
        make_rslds_parameters(C_init)

    rslds = InputOnlySLDS(
        D_in=D_latent,
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

    # Fit the model
    lps = []
    z_smpls = []
    for _ in progprint_xrange(N_iters):
        rslds.resample_model()
        lps.append(rslds.log_likelihood())
        z_smpls.append(rslds.stateseqs[0].copy())

    x_test = rslds.states_list[0].gaussian_states
    z_smpls = np.array(z_smpls)
    lps = np.array(lps)

    print("Inf W_markov:\n{}".format(rslds.trans_distn.A[:,:K]))
    print("Inf W_input:\n{}".format(rslds.trans_distn.A[:,K:]))
    return rslds, lps, z_smpls, x_test

if __name__ == "__main__":
    ## Simulate Lorenz data
    x_true, y, C, mask = simulate_lorenz()
    inputs = np.ones((T,1))

    x_init = y.copy()
    C_init = np.hstack((np.eye(D_latent), np.zeros((D_obs,1))))

    ## Fit an ARHMM for initialization
    arhmm, z_init = fit_arhmm(x_init)

    ## Fit a DecisionList to get a permutation of z_init
    z_perm, dl_reg = fit_decision_list(z_init, x_init)

    # Specifying the number of iterations of the Gibbs sampler
    N_iters = 1000

    ## Fit a standard SLDS
    slds, slds_lps, slds_z_smpls, slds_x = \
        fit_slds(inputs, z_perm, x_init, y, mask, C_init, N_iters=N_iters)

    ## Fit an input-only recurrent SLDS
    iorslds, iorslds_lps, iorslds_z_smpls, iorslds_x = \
        fit_inputonly_rslds(inputs, z_perm, x_init, y, mask, dl_reg, C_init, N_iters=N_iters)

    ## Generate from the model
    T_gen = 2000
    inputs = np.ones((T_gen, 1))
    for k in range(K):
        iorslds.init_dynamics_distns[k].mu = x_init[0].copy()
        iorslds.init_dynamics_distns[k].sigma = 1e-3 * np.eye(D_latent)

    (iorslds_ys_gen, iorslds_x_gen), iorslds_z_gen = iorslds.generate(T=T_gen, inputs=inputs, with_noise=True)
    iorslds_y_gen = iorslds_ys_gen[0]
    iorslds_y_gen = iorslds_x_gen.dot(iorslds.emission_distns[0].A[:, :D_latent].T) + iorslds.emission_distns[0].A[:, D_latent]

    for k in range(K):
        slds.init_dynamics_distns[k].mu = x_init[0].copy()
        slds.init_dynamics_distns[k].sigma = 1e-3 * np.eye(D_latent)

    (slds_ys_gen, slds_x_gen), slds_z_gen = slds.generate(T=T_gen, inputs=inputs)
    slds_y_gen = slds_ys_gen[0]
    slds_y_gen = slds_x_gen.dot(slds.emission_distns[0].A[:,:D_latent].T) + slds.emission_distns[0].A[:,D_latent]

    make_figure(y, x_true,
                iorslds, iorslds_z_smpls, iorslds_x,
                iorslds_z_gen, iorslds_x_gen, iorslds_y_gen,
                slds, slds_z_smpls, slds_x,
                slds_z_gen, slds_x_gen, slds_y_gen,
                )


    plt.show()
