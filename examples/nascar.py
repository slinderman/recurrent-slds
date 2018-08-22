import os
import pickle
import argparse
from tqdm import tqdm

import numpy as np
import numpy.random as npr

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.font_manager import FontProperties

from pybasicbayes.models import FactorAnalysis
from pybasicbayes.distributions import \
    Regression, Gaussian, DiagonalRegression, AutoRegression

from pyhsmm.util.general import relabel_by_permutation
from autoregressive.models import ARWeakLimitStickyHDPHMM
from pyslds.util import get_empirical_ar_params
from pyslds.models import HMMSLDS
from pypolyagamma.distributions import MultinomialRegression
from pypolyagamma.binary_trees import decision_list

from rslds.decision_list import DecisionList
from rslds.models import PGRecurrentSLDS, StickyPGRecurrentSLDS, \
    PGRecurrentOnlySLDS, StickyPGRecurrentOnlySLDS
from rslds.util import compute_psi_cmoments
import rslds.plotting as rplt


# Constants
K_true = 4
D_latent = 2

# Parse command line arguments
parser = argparse.ArgumentParser(description='Synthetic NASCAR Example')
parser.add_argument('--T', type=int, default=10000,
                    help='number of training time steps')
parser.add_argument('--T_sim', type=int, default=2000,
                    help='number of simulation time steps')
parser.add_argument('--K', type=int, default=4,
                    help='number of inferred states')
parser.add_argument('--D_obs', type=int, default=10,
                    help='number of observed dimensions')
parser.add_argument('--mask_start', type=int, default=0,
                    help='time index of start of mask')
parser.add_argument('--mask_stop', type=int, default=0,
                    help='time index of end of mask')
parser.add_argument('--N_samples', type=int, default=1000,
                    help='number of iterations to run the Gibbs sampler')
parser.add_argument('--seed', type=int, default=0,
                    help='random seed (default: 0)')
parser.add_argument('--cache', action='store_true', default=False,
                    help='whether or not to cache the results')
parser.add_argument('-o', '--output-dir', default='.',
                    help='where to store the results')
args = parser.parse_args()

print("Setting seed to ", args.seed)
npr.seed(args.seed)


# Cache results if requested
def cached(results_name):
    if args.cache:
        def _cache(func):
            def func_wrapper(*args, **kwargs):
                results_file = os.path.join(args.output_dir, results_name)
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


# Make an example with 2D latent states and 4 discrete states
def simulate_nascar():
    assert K_true == 4

    def random_rotation(n, theta):
        rot = np.array([[np.cos(theta), -np.sin(theta)],
                        [np.sin(theta), np.cos(theta)]])
        out = np.zeros((n,n))
        out[:2,:2] = rot
        q = np.linalg.qr(np.random.randn(n,n))[0]
        # q = np.eye(n)
        return q.dot(out).dot(q.T)

    As = [random_rotation(D_latent, np.pi/24.),
          random_rotation(D_latent, np.pi/48.)]

    # Set the center points for each system
    centers = [np.array([+2.0, 0.]),
               np.array([-2.0, 0.])]
    bs = [-(A - np.eye(D_latent)).dot(center) for A, center in zip(As, centers)]

    # Add a "right" state
    As.append(np.eye(D_latent))
    bs.append(np.array([+0.1, 0.]))

    # Add a "right" state
    As.append(np.eye(D_latent))
    bs.append(np.array([-0.35, 0.]))

    # Construct multinomial regression to divvy up the space #
    tree = decision_list(K_true)
    w1, b1 = np.array([+1.0, 0.0]), np.array([-2.0])   # x + b > 0 -> x > -b
    w2, b2 = np.array([-1.0, 0.0]), np.array([-2.0])   # -x + b > 0 -> x < b
    w3, b3 = np.array([0.0, +1.0]), np.array([0.0])    # y > 0

    reg_W = np.row_stack((w1, w2, w3))
    reg_b = np.row_stack((b1, b2, b3))

    # Scale the weights to make the transition boundary sharper
    reg_scale = 100.
    reg_b *= reg_scale
    reg_W *= reg_scale

    # Account for stick breaking asymmetry
    mu_b, _ = compute_psi_cmoments(np.ones(K_true))
    reg_b += mu_b[:,None]

    # Make a recurrent SLDS with these params #
    dynamics_distns = [
        Regression(
            A=np.column_stack((A,b)),
            sigma=1e-4 * np.eye(D_latent),
            nu_0=D_latent + 2,
            S_0=1e-4 * np.eye(D_latent),
            M_0=np.zeros((D_latent, D_latent + 1)),
            K_0=np.eye(D_latent + 1),
        )
        for A,b in zip(As, bs)]

    init_dynamics_distns = [
        Gaussian(
            mu=np.array([0.0, 1.0]),
            sigma=1e-3 * np.eye(D_latent))
        for _ in range(K_true)]

    C = np.hstack((npr.randn(args.D_obs, D_latent), np.zeros((args.D_obs, 1))))
    emission_distns = \
        DiagonalRegression(args.D_obs, D_latent+1,
                           A=C, sigmasq=1e-5 *np.ones(args.D_obs),
                           alpha_0=2.0, beta_0=2.0)

    model = PGRecurrentSLDS(
        trans_params=dict(A=np.hstack((np.zeros((K_true-1, K_true)), reg_W)), b=reg_b,
                          sigmasq_A=100., sigmasq_b=100., tree=tree),
        init_state_distn='uniform',
        init_dynamics_distns=init_dynamics_distns,
        dynamics_distns=dynamics_distns,
        emission_distns=emission_distns)

    # Sample from the model
    inputs = np.ones((args.T, 1))
    y, x, z = model.generate(T=args.T, inputs=inputs)

    # Maks off some data
    mask = np.ones((args.T, args.D_obs), dtype=bool)
    mask[args.mask_start:args.mask_stop] = False
    return model, inputs, z, x, y, mask


@cached("factor_analysis")
def fit_factor_analysis(y, mask=None, N_iters=100):
    print("Fitting Factor Analysis")
    model = FactorAnalysis(args.D_obs, D_latent)

    if mask is None:
        mask = np.ones_like(y, dtype=bool)

    # Center the data
    b = y.mean(0)
    data = model.add_data(y-b, mask=mask)
    for _ in tqdm(range(N_iters)):
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
        for _ in range(args.K)]

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
    for _ in tqdm(range(args.N_samples)):
        arhmm.resample_model()
        lps.append(arhmm.log_likelihood())

    z_init = arhmm.states_list[0].stateseq
    z_init = np.concatenate(([0], z_init))

    return arhmm, z_init


# Use a DecisionList to permute the discrete states
def fit_decision_list(z, y):
    print("Fitting Decision List")
    dlist = DecisionList(args.K, D_latent)
    dlist.fit(y[:-1], z[1:])

    dl_reg = MultinomialRegression(1, args.K, D_latent)
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
        for _ in range(args.K)]

    dynamics_distns = [
        Regression(
            nu_0=D_latent + 2,
            S_0=1e-4 * np.eye(D_latent),
            M_0=np.hstack((np.eye(D_latent), np.zeros((D_latent, 1)))),
            K_0=np.eye(D_latent + 1),
        )
        for _ in range(args.K)]

    emission_distns = \
        DiagonalRegression(args.D_obs, D_latent + 1,
                           A=C_init.copy(), sigmasq=np.ones(args.D_obs),
                           alpha_0=2.0, beta_0=2.0)

    return init_dynamics_distns, dynamics_distns, emission_distns


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
    for _ in tqdm(range(100)):
        slds.resample_dynamics_distns()

    # Fit the model
    lps = []
    z_smpls = []
    for itr in tqdm(range(args.N_samples)):
        slds.resample_model()
        lps.append(slds.log_likelihood())
        z_smpls.append(slds.stateseqs[0].copy())

    x_test = slds.states_list[0].gaussian_states
    z_smpls = np.array(z_smpls)
    lps = np.array(lps)

    return slds, lps, z_smpls, x_test


@cached("rslds")
def fit_rslds(inputs, z_init, x_init, y, mask, dl_reg, C_init):
    print("Fitting rSLDS")
    init_dynamics_distns, dynamics_distns, emission_distns = \
        make_rslds_parameters(C_init)

    rslds = PGRecurrentSLDS(
        trans_params=dict(sigmasq_A=10000., sigmasq_b=10000.,
                          A=np.hstack((np.zeros((args.K - 1, args.K)), dl_reg.A)),
                          b=dl_reg.b),
        init_state_distn='uniform',
        init_dynamics_distns=init_dynamics_distns,
        dynamics_distns=dynamics_distns,
        emission_distns=emission_distns,
        fixed_emission=False)

    rslds.add_data(y, inputs=inputs, mask=mask)

    # Initialize states
    rslds.states_list[0].stateseq = z_init.copy()
    rslds.states_list[0].gaussian_states = x_init.copy()

    # Initialize dynamics
    print("Initializing dynamics with Gibbs sampling")
    for itr in tqdm(range(100)):
        rslds.resample_dynamics_distns()

    # Fit the model
    lps = []
    z_smpls = []
    for itr in tqdm(range(args.N_samples)):
        rslds.resample_model()
        lps.append(rslds.log_likelihood())
        z_smpls.append(rslds.stateseqs[0].copy())

    x_test = rslds.states_list[0].gaussian_states
    z_smpls = np.array(z_smpls)
    lps = np.array(lps)
    return rslds, lps, z_smpls, x_test


@cached("sticky_rslds")
def fit_sticky_rslds(inputs, z_init, x_init, y, mask, dl_reg, C_init):
    print("Fitting Sticky rSLDS")
    init_dynamics_distns, dynamics_distns, emission_distns = \
        make_rslds_parameters(C_init)

    rslds = StickyPGRecurrentSLDS(
        D_in=D_latent,
        trans_params=dict(sigmasq_A=10000., sigmasq_b=10000., kappa=100.,
                          A=np.hstack((np.zeros((args.K - 1, args.K)), dl_reg.A)),
                          b=dl_reg.b),
        init_state_distn='uniform',
        init_dynamics_distns=init_dynamics_distns,
        dynamics_distns=dynamics_distns,
        emission_distns=emission_distns,
        fixed_emission=False)

    rslds.add_data(y, inputs=inputs, mask=mask)

    # Initialize states
    rslds.states_list[0].stateseq = z_init.copy()
    rslds.states_list[0].gaussian_states = x_init.copy()

    # Initialize dynamics
    print("Initializing dynamics with Gibbs sampling")
    for itr in tqdm(range(100)):
        rslds.resample_dynamics_distns()

    # Fit the model
    lps = []
    z_smpls = []
    for _ in tqdm(range(args.N_samples)):
        rslds.resample_model()
        lps.append(rslds.log_likelihood())
        z_smpls.append(rslds.stateseqs[0].copy())

    x_test = rslds.states_list[0].gaussian_states
    z_smpls = np.array(z_smpls)
    lps = np.array(lps)
    return rslds, lps, z_smpls, x_test


@cached("roslds")
def fit_roslds(inputs, z_init, x_init, y, mask, dl_reg, C_init):
    print("Fitting input only rSLDS")
    init_dynamics_distns, dynamics_distns, emission_distns = \
        make_rslds_parameters(C_init)

    rslds = PGRecurrentOnlySLDS(
        trans_params=dict(sigmasq_A=10000., sigmasq_b=10000.,
                          # A=np.hstack((np.zeros((args.K - 1, args.K)), dl_reg.A)),
                          # b=dl_reg.b
                          ),
        init_state_distn='uniform',
        init_dynamics_distns=init_dynamics_distns,
        dynamics_distns=dynamics_distns,
        emission_distns=emission_distns,
        fixed_emission=False)

    rslds.add_data(y, inputs=inputs, mask=mask, stateseq=z_init, gaussian_states=x_init)

    # Resample auxiliary variables
    for states in rslds.states_list:
        states.resample_transition_auxiliary_variables()

    # Initialize dynamics
    print("Initializing dynamics with Gibbs sampling")
    for itr in tqdm(range(100)):
        rslds.resample_dynamics_distns()

    print("Initializing transitions with Gibbs sampling")
    for itr in tqdm(range(100)):
        rslds.resample_trans_distn()

    # Fit the model
    lps = []
    z_smpls = []
    for itr in tqdm(range(args.N_samples)):
        rslds.resample_model()
        lps.append(rslds.log_likelihood())
        z_smpls.append(rslds.stateseqs[0].copy())

    x_smpl = rslds.states_list[0].gaussian_states
    z_smpls = np.array(z_smpls)
    lps = np.array(lps)
    return rslds, lps, z_smpls, x_smpl


@cached("sticky_roslds")
def fit_sticky_roslds(inputs, z_init, x_init, y, mask, dl_reg, C_init):
    print("Fitting sticky input only rSLDS")
    init_dynamics_distns, dynamics_distns, emission_distns = \
        make_rslds_parameters(C_init)

    rslds = StickyPGRecurrentOnlySLDS(
        trans_params=dict(sigmasq_A=10000., sigmasq_b=10000.,
                          kappa=1., sigmasq_kappa=1.0,
                          A=np.hstack((np.zeros((K-1, K)), dl_reg.A)),
                          b=dl_reg.b),
        init_state_distn='uniform',
        init_dynamics_distns=init_dynamics_distns,
        dynamics_distns=dynamics_distns,
        emission_distns=emission_distns,
        fixed_emission=False)

    rslds.add_data(y, inputs=inputs, mask=mask)

    # Initialize states
    rslds.states_list[0].stateseq = z_init.copy()
    rslds.states_list[0].gaussian_states = x_init.copy()

    # Initialize dynamics
    print("Initializing dynamics with Gibbs sampling")
    for itr in tqdm(range(100)):
        rslds.resample_dynamics_distns()

    # Fit the model
    lps = []
    z_smpls = []
    for itr in tqdm(range(args.N_samples)):
        rslds.resample_model()
        lps.append(rslds.log_likelihood())
        z_smpls.append(rslds.stateseqs[0].copy())

    x_test = rslds.states_list[0].gaussian_states
    z_smpls = np.array(z_smpls)
    lps = np.array(lps)
    return rslds, lps, z_smpls, x_test


# Plotting code
def make_figure(true_model, z_true, x_true, y,
                rslds, zs_rslds, x_rslds,
                z_rslds_gen, x_rslds_gen,
                z_slds_gen, x_slds_gen):
    fig = plt.figure(figsize=(6.5,3.5))
    gs = gridspec.GridSpec(2,3)

    fp = FontProperties()
    fp.set_weight("bold")

    # True dynamics
    ax1 = fig.add_subplot(gs[0,0])
    rplt.plot_most_likely_dynamics(true_model.trans_distn,
                                   true_model.dynamics_distns,
                                   xlim=(-3,3), ylim=(-2,2),
                                   ax=ax1)

    # Overlay a partial trajectory
    rplt.plot_trajectory(z_true[1:1000], x_true[1:1000], ax=ax1, ls="-")
    ax1.set_title("True Latent Dynamics")
    plt.figtext(.025, 1-.075, '(a)', fontproperties=fp)

    # Plot a few output dimensions
    ax2 = fig.add_subplot(gs[1, 0])
    for n in range(args.D_obs):
        rplt.plot_data(z_true[1:1000], y[1:1000, n], ax=ax2, ls="-")
    ax2.set_xlabel("Time")
    ax2.set_ylabel("$y$")
    ax2.set_title("Observed Data")
    plt.figtext(.025, .5 - .075, '(b)', fontproperties=fp)

    # Plot the inferred dynamics under the rSLDS
    ax3 = fig.add_subplot(gs[0, 1])
    rplt.plot_most_likely_dynamics(rslds.trans_distn,
                                   rslds.dynamics_distns,
                                   xlim=(-3, 3), ylim=(-2, 2),
                                   ax=ax3)

    # Overlay a partial trajectory
    rplt.plot_trajectory(zs_rslds[-1][1:1000], x_rslds[1:1000], ax=ax3, ls="-")
    ax3.set_title("Inferred Dynamics (rSLDS)")
    plt.figtext(.33 + .025, 1. - .075, '(c)', fontproperties=fp)

    # Plot samples of discrete state sequence
    ax4 = fig.add_subplot(gs[1,1])
    rplt.plot_z_samples(args.K, zs_rslds, zref=z_true, plt_slice=(0,1000), ax=ax4)
    ax4.set_title("Discrete State Samples")
    plt.figtext(.33 + .025, .5 - .075, '(d)', fontproperties=fp)

    # Plot simulated SLDS data
    ax5 = fig.add_subplot(gs[0, 2])
    rplt.plot_trajectory(z_slds_gen[-1000:], x_slds_gen[-1000:], ax=ax5, ls="-")
    plt.grid(True)
    ax5.set_title("Generated States (SLDS)")
    plt.figtext(.66 + .025, 1. - .075, '(e)', fontproperties=fp)

    # Plot simulated rSLDS data
    ax6 = fig.add_subplot(gs[1, 2])
    rplt.plot_trajectory(z_rslds_gen[-1000:], x_rslds_gen[-1000:], ax=ax6, ls="-")
    ax6.set_title("Generated States (rSLDS)")
    plt.grid(True)
    plt.figtext(.66 + .025, .5 - .075, '(f)', fontproperties=fp)

    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "nascar.png"), dpi=200)
    plt.savefig(os.path.join(args.output_dir, "nascar.pdf"))
    plt.show()


if __name__ == "__main__":
    # Simulate NASCAR data
    true_model, inputs, z_true, x_true, y, mask = simulate_nascar()

    # Run PCA to get 2D dynamics
    x_init, C_init = fit_pca(y)

    # Fit an ARHMM for initialization
    # Basically, we're only fitting on data that was observed
    good_inds = np.all(mask, axis=1)
    good_x_init = x_init[good_inds]
    arhmm, good_z_init = fit_arhmm(good_x_init)
    z_init = np.random.randint(0, args.K, size=args.T)
    z_init[good_inds] = good_z_init
    z_init[args.mask_start:args.mask_stop] = z_init[args.mask_start-1]
    x_init[~good_inds] = 0

    # Fit a DecisionList to get a permutation of z_init
    # z_perm, dl_reg = fit_decision_list(z_init, x_init)
    z_perm = z_init
    dl_reg = None

    # Fit a standard SLDS
    slds, slds_lps, slds_z_smpls, slds_x = \
        fit_slds(inputs, z_perm, x_init, y, mask, C_init)

    # Fit a recurrent SLDS
    # rslds, rslds_lps, rslds_z_smpls, rslds_x = \
    #     fit_rslds(inputs, z_perm, x_init, y, mask, dl_reg, C_init, N_iters=N_iters)

    # Fit an input-only recurrent SLDS
    roslds, roslds_lps, roslds_z_smpls, roslds_x = \
        fit_roslds(inputs, z_perm, x_init, y, mask, dl_reg, C_init)

    rplt.plot_trajectory_and_probs(
        roslds_z_smpls[-1][1:], roslds_x[1:],
        trans_distn=roslds.trans_distn,
        title="Recurrent SLDS")

    # Generate from the models
    roslds_y_gen, roslds_x_gen, roslds_z_gen = \
        roslds.generate(T=args.T_sim, inputs=np.ones((args.T_sim, 1)))
    slds_y_gen, slds_x_gen, slds_z_gen = \
        slds.generate(T=args.T_sim, inputs=np.ones((args.T_sim, 1)))

    make_figure(true_model, z_true, x_true, y,
                roslds, roslds_z_smpls, roslds_x,
                roslds_z_gen, roslds_x_gen,
                slds_z_gen, slds_x_gen
                )

    plt.show()
