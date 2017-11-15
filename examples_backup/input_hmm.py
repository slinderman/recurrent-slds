import numpy as np
np.random.seed(0)

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns
sns.set_style("white")
import seaborn as sns
color_names = ["red",
               "windows blue",
               "medium green",
               "orange",
               ]
colors = sns.xkcd_palette(color_names)
from hips.plotting.colormaps import gradient_cmap
cmap = gradient_cmap(colors)

from pybasicbayes.distributions import Gaussian
from rslds.models import InputHMM


#################################################
# create the "true" model and true covariates   #
#################################################
T = 1000
D_in  = 2
D_obs = 2
Nmax  = 2

tgrid = np.linspace(-5*np.pi, 5*np.pi, T)
covariate_seq = np.column_stack((np.sin(tgrid), np.cos(tgrid)))
covariate_seq += 0.001 * np.random.randn(T, D_in)

obs_hypparams = {'mu_0':np.zeros(D_obs),
                'sigma_0':np.eye(D_obs),
                'kappa_0':1.0,
                'nu_0': D_obs + 2}
true_model = \
    InputHMM(obs_distns=[Gaussian(**obs_hypparams) for state in range(Nmax)],
             init_state_concentration=1.0,
             D_in=D_in,
             trans_params=dict(sigmasq_A=4.0, sigmasq_b=0.001))

# Set the weights by hand such that they primarily
# depend on the input
true_model.trans_distn.A[0][Nmax:] = [ 5.,  5.]
# true_model.trans_distn.A[1][Nmax:] = [-2.,  2.]
# true_model.trans_distn.A[2][Nmax:] = [-2., -2.]

# generate fake data and plot
dataset = [true_model.generate(T, covariates=covariate_seq[:,:D_in]) for _ in range(5)]

########################################################
# Generate inference test model - initialized randomly #
########################################################
test_model = \
    InputHMM(obs_distns=[Gaussian(**obs_hypparams) for state in range(Nmax)],
             init_state_concentration=1.0,
             D_in=D_in,
             trans_params=dict(sigmasq_b=0.001))

for (obs, covs), _ in dataset:
    test_model.add_data(data=obs, covariates=covs[:,:D_in])

# run sampler
print("io hmm sampling")
num_iter = 200
A_smpls = np.zeros((num_iter,) + test_model.trans_distn.A.shape)
z_smpls = np.zeros((num_iter, T))
for itr in range(num_iter):
    test_model.resample_model(num_procs=0)
    if itr % 20 == 0:
        print(test_model.log_likelihood())
    A_smpls[itr, :, :] = test_model.trans_distn.A
    z_smpls[itr] = test_model.stateseqs[0]

# Get the inferred state sequence
true_Z = true_model.stateseqs[0]
true_X = dataset[0][0][0]
test_Z = test_model.stateseqs[0]


# Plot the true and inferred state sequences
plt_slice = (0, T)
fig = plt.figure(figsize=(10,5))
import matplotlib.gridspec as gridspec
gs = gridspec.GridSpec(5,1)
ax1 = fig.add_subplot(gs[:-1])

im = ax1.matshow(z_smpls, aspect='auto', cmap="RdBu", vmin=0, vmax=1)
ax1.autoscale(False)
ax1.set_xticks([])
ax1.set_yticks([0, num_iter])
ax1.set_ylabel("Iteration")
ax1.set_xlim(plt_slice)
ax1.set_xticks(plt_slice)

ax2 = fig.add_subplot(gs[-1])
im = ax2.matshow(true_Z[None,:], aspect='auto', cmap="RdBu", vmin=0, vmax=1)
ax2.autoscale(False)
ax2.set_xticks([])
ax2.set_xlim(plt_slice)
ax2.set_xticks(plt_slice)


## Plot the true data
fig = plt.figure(figsize=(10,5))
z_cps = np.concatenate(([0], np.where(np.diff(true_Z))[0] + 1, [true_Z.size]))
for start, stop in zip(z_cps[:-1], z_cps[1:]):
    stop = min(T, stop+1)
    plt.plot(np.arange(start, stop),
             true_X[start:stop,0],
             lw=1,
             color=colors[true_Z[start]])

#################################
# Plot the state probs vs space #
#################################
XX,YY = np.meshgrid(np.linspace(-1.5,1.5),
                    np.linspace(-1.5,1.5))

XY = np.column_stack((np.ravel(XX), np.ravel(YY)))
true_prs = true_model.trans_distn.pi(
    np.column_stack((np.zeros((XY.shape[0], Nmax)), XY)))

test_prs = test_model.trans_distn.pi(
    np.column_stack((np.zeros((XY.shape[0], Nmax)), XY)))

fig = plt.figure(figsize=(10,6))
for to_plot in range(Nmax):
    ax1 = fig.add_subplot(2, Nmax, to_plot + 1)
    im1 = ax1.imshow(
        true_prs[:,to_plot].reshape(*XX.shape),
        extent=(-1.5, 1.5, 1.5, -1.5),
        vmin=0, vmax=1)

    ax1.plot(covariate_seq[true_Z==to_plot, 0],
             covariate_seq[true_Z==to_plot, 1], 'o',
             markerfacecolor=colors[to_plot],
             markeredgecolor="none",
             markersize=6,
             alpha=0.5)
    ax1.set_xlim(-1.5, 1.5)
    ax1.set_ylim(-1.5, 1.5)
    ax1.set_title("True states, True divide")
    ax1.invert_yaxis()

    if to_plot == Nmax - 1:
        divider = make_axes_locatable(ax1)
        cbax = divider.new_horizontal(size="5%", pad=0.05)
        fig.add_axes(cbax)
        plt.colorbar(im1, cax=cbax)

    ax2 = fig.add_subplot(2, Nmax, Nmax + to_plot + 1)
    im2 = ax2.imshow(
        test_prs[:,to_plot].reshape(*XX.shape),
        extent=(-1.5, 1.5, 1.5, -1.5),
        vmin=0, vmax=1)
    ax2.plot(covariate_seq[test_Z==to_plot,0],
             covariate_seq[test_Z==to_plot,1], 'o',
             markerfacecolor=colors[to_plot],
             markeredgecolor="none",
             markersize=6,
             alpha=0.5)
    ax2.set_xlim(-1.5, 1.5)
    ax2.set_ylim(-1.5, 1.5)
    ax2.set_title("Inf states, Inf divide")
    ax2.invert_yaxis()

    if to_plot == Nmax - 1:
        divider = make_axes_locatable(ax2)
        cbax = divider.new_horizontal(size="5%", pad=0.05)
        fig.add_axes(cbax)
        plt.colorbar(im2, cax=cbax)

fig.suptitle("Input HMM with 2D Exogenous Inputs")
plt.tight_layout()
# plt.savefig("inhmm_partition.png")

# # Plot the true state sequence over time
# plt.figure(figsize=(8,4))
# plt.plot(true_Z)

plt.show()