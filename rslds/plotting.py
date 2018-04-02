import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LinearSegmentedColormap

import seaborn as sns

color_names = ["windows blue",
               "red",
               "amber",
               "faded green",
               "dusty purple",
               "orange",
               "clay",
               "pink",
               "greyish",
               "mint",
               "light cyan",
               "steel blue",
               "forest green",
               "pastel purple",
               "salmon",
               "dark brown"]

colors = sns.xkcd_palette(color_names)
sns.set_style("white")
sns.set_context("paper")


def gradient_cmap(gcolors, nsteps=256, bounds=None):
    """
    Make a colormap that interpolates between a set of colors
    """
    ncolors = len(gcolors)
    if bounds is None:
        bounds = np.linspace(0, 1, ncolors)

    reds = []
    greens = []
    blues = []
    alphas = []
    for b, c in zip(bounds, gcolors):
        reds.append((b, c[0], c[0]))
        greens.append((b, c[1], c[1]))
        blues.append((b, c[2], c[2]))
        alphas.append((b, c[3], c[3]) if len(c) == 4 else (b, 1., 1.))

    cdict = {'red': tuple(reds),
             'green': tuple(greens),
             'blue': tuple(blues),
             'alpha': tuple(alphas)}

    cmap = LinearSegmentedColormap('grad_colormap', cdict, nsteps)
    return cmap


def plot_dynamics(A, b=None, ax=None, plot_center=True,
                  xlim=(-4, 4), ylim=(-3, 3), npts=20,
                  color='r'):
    D_latent = A.shape[0]
    b = np.zeros((A.shape[0], 1)) if b is None else b
    x = np.linspace(*xlim, npts)
    y = np.linspace(*ylim, npts)
    X, Y = np.meshgrid(x, y)
    xy = np.column_stack((X.ravel(), Y.ravel()))

    # dydt_m = xy.dot(A.T) + b.T - xy
    dydt_m = xy.dot(A.T) + b.T - xy

    if ax is None:
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111)

    ax.quiver(xy[:, 0], xy[:, 1],
              dydt_m[:, 0], dydt_m[:, 1],
              color=color, alpha=1.0,
              headwidth=5.)

    # Plot the stable point
    if plot_center:
        try:
            center = -np.linalg.solve(A - np.eye(D_latent), b)
            ax.plot(center[0], center[1], 'o', color=color, markersize=8)
        except:
            print("Dynamics are not invertible!")

    ax.set_xlabel('$x_1$', fontsize=12, labelpad=10)
    ax.set_ylabel('$x_2$', fontsize=12, labelpad=10)

    return ax


def plot_all_dynamics(dynamics_distns):
    K = len(dynamics_distns)
    D_latent = dynamics_distns[0].D_out

    fig = plt.figure(figsize=(12, 3))
    for k in range(K):
        ax = fig.add_subplot(1, K, k + 1)
        plot_dynamics(dynamics_distns[k].A[:, :D_latent],
                      b=dynamics_distns[k].A[:, D_latent:],
                      plot_center=False,
                      color=colors[k], ax=ax)


def plot_most_likely_dynamics(
        reg, dynamics_distns,
        xlim=(-4, 4), ylim=(-3, 3), nxpts=20, nypts=10,
        alpha=0.8,
        ax=None, figsize=(3, 3)):
    K = len(dynamics_distns)
    D_latent = dynamics_distns[0].D_out
    x = np.linspace(*xlim, nxpts)
    y = np.linspace(*ylim, nypts)
    X, Y = np.meshgrid(x, y)
    xy = np.column_stack((X.ravel(), Y.ravel()))

    # Get the probability of each state at each xy location
    Ts = reg.get_trans_matrices(xy)
    prs = Ts[:, 0, :]
    z = np.argmax(prs, axis=1)

    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)

    for k in range(K):
        A = dynamics_distns[k].A[:, :D_latent]
        b = dynamics_distns[k].A[:, D_latent:]
        dydt_m = xy.dot(A.T) + b.T - xy

        zk = z == k
        if zk.sum(0) > 0:
            ax.quiver(xy[zk, 0], xy[zk, 1],
                      dydt_m[zk, 0], dydt_m[zk, 1],
                      color=colors[k], alpha=alpha)

    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')

    plt.tight_layout()

    return ax


def plot_trans_probs(reg, xlim=(-4, 4), ylim=(-3, 3), n_pts=50, ax=None):
    K = reg.D_out + 1

    XX, YY = np.meshgrid(np.linspace(*xlim, n_pts),
                         np.linspace(*ylim, n_pts))
    XY = np.column_stack((np.ravel(XX), np.ravel(YY)))
    test_prs = reg.get_trans_matrices(XY)[:, 0, :]

    if ax is None:
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111)

    for k in range(K):
        start = np.array([1., 1., 1., 0.])
        end = np.concatenate((colors[k % len(colors)], [0.5]))
        cmap = gradient_cmap([start, end])
        im1 = ax.imshow(test_prs[:, k].reshape(*XX.shape),
                        extent=xlim + tuple(reversed(ylim)),
                        vmin=0, vmax=1, cmap=cmap)

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

    plt.tight_layout()
    return ax


def plot_trajectory(zhat, x, ax=None, ls="-"):
    zcps = np.concatenate(([0], np.where(np.diff(zhat))[0] + 1, [zhat.size]))
    if ax is None:
        fig = plt.figure(figsize=(4, 4))
        ax = fig.gca()
    for start, stop in zip(zcps[:-1], zcps[1:]):
        ax.plot(x[start:stop + 1, 0],
                x[start:stop + 1, 1],
                lw=1, ls=ls,
                color=colors[zhat[start] % len(colors)],
                alpha=1.0)

    return ax


def plot_trajectory_and_probs(z, x,
                              ax=None,
                              trans_distn=None,
                              title=None,
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
    return ax


def plot_data(zhat, y, ax=None, ls="-"):
    zcps = np.concatenate(([0], np.where(np.diff(zhat))[0] + 1, [zhat.size]))
    if ax is None:
        fig = plt.figure(figsize=(4, 4))
        ax = fig.gca()
    for start, stop in zip(zcps[:-1], zcps[1:]):
        stop = min(y.shape[0], stop + 1)
        ax.plot(np.arange(start, stop),
                y[start:stop],
                lw=1, ls=ls,
                color=colors[zhat[start] % len(colors)],
                alpha=1.0)

    return ax


def plot_separate_trans_probs(reg, xlim=(-4, 4), ylim=(-3, 3), n_pts=100, ax=None):
    K = reg.D_out
    XX, YY = np.meshgrid(np.linspace(*xlim, n_pts),
                         np.linspace(*ylim, n_pts))
    XY = np.column_stack((np.ravel(XX), np.ravel(YY)))

    D_reg = reg.D_in
    inputs = np.hstack((np.zeros((n_pts ** 2, D_reg - 2)), XY))
    test_prs = reg.pi(inputs)

    if ax is None:
        fig = plt.figure(figsize=(12, 3))

    for k in range(K):
        ax = fig.add_subplot(1, K, k + 1)
        cmap = gradient_cmap([np.ones(3), colors[k % len(colors)]])
        im1 = ax.imshow(test_prs[:, k].reshape(*XX.shape),
                        extent=xlim + tuple(reversed(ylim)),
                        vmin=0, vmax=1, cmap=cmap)

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im1, cax=cax, ax=ax)

    plt.tight_layout()
    return ax


def plot_z_samples(K, zs, zref=None,
                   plt_slice=None,
                   N_iters=None,
                   title=None,
                   ax=None):
    if ax is None:
        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(111)

    zs = np.array(zs)
    if plt_slice is None:
        plt_slice = (0, zs.shape[1])
    if N_iters is None:
        N_iters = zs.shape[0]

    im = ax.imshow(zs[:, slice(*plt_slice)], aspect='auto', vmin=0, vmax=K - 1,
                   cmap=gradient_cmap(colors[:K]), interpolation="nearest",
                   extent=plt_slice + (N_iters, 0))

    ax.set_xticks([])
    ax.set_ylabel("Iteration")

    if zref is not None:
        divider = make_axes_locatable(ax)
        ax2 = divider.append_axes("bottom", size="10%", pad=0.05)

        zref = np.atleast_2d(zref)
        im = ax2.imshow(zref[:, slice(*plt_slice)], aspect='auto', vmin=0, vmax=K - 1,
                        cmap=gradient_cmap(colors[:K]), interpolation="nearest")
        ax.set_xticks([])
        ax2.set_yticks([])
        ax2.set_ylabel("True $z$", rotation=0)
        ax2.yaxis.set_label_coords(-.15, -.5)
        ax2.set_xlabel("Time")

    if title is not None:
        ax.set_title(title)
