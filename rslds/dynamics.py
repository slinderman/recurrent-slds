import numpy as np

from pybasicbayes.distributions import Regression
from pybasicbayes.util.stats import sample_gaussian
from pypolyagamma.binary_trees import ids, adjacency, depths

class TreeStructuredHierarchicalDynamics(object):
    """
    A hierarchical model for linking dynamics matrices according
    to a tree structure.  The observations hang off the leaves of
    the tree, but the parents of the leaves specify their prior
    mean.  The parents of the parents specify their means, and so on.

    Each observation is a tuple (x_t, y_t, z_t) where (x_t, y_t) are
    the input and output, respectively, and z_t \in {1, ..., L} specifies
    which of the L leaf nodes the observation is drawn from.
    Let P = dim(y) and D = dim(x).

    The parameters of the model are a set of regression matrices
    {A_1, ..., A_N} for each node in the tree.  The prior is,

       vec(A_n) ~ N(vec(A_m), sigma^2 I)

    where m = par(n) is the parent of node n.  If n is the root node,
    its parent A_m = I.

    The likelihood at the leaf node is

    log p(y_t | x_t, z_t = n, A_n) = log N(y_t | A_n x_t, Q_n^{-1})

        = -1/2 (y_t - A_n x_t)^T Q (y_t - A_n x_t) + const
        = -1/2 x_t^T A_n^T Q_n A_n x_t + x_t^T A_n^T Q_n y_t + const

    When Q is diagonal the log probability separates into a sum over
    the output P dimensions

        = -1/2 \sum_p (x_t^T A_{np:} Q_{npp} A_{np:}^T x_t)
          + \sum_p x_t^T A_{np:} Q_{npp} y_tp + const

        = \sum_p -1/2 Tr(A_{np:} A_{np:}^T Q_{npp} x_t x_t^T)
          + \sum_p Tr(A_{np:} Q_{npp} y_tp x_t^T ) + const

        = \sum_p -1/2  Tr(A_{np:} A_{np:}^T J_tp) + Tr(A_{np:} h_tp)

    where

        J_tp = Q_{npp} x_t x_t^T
        h_tp = Q_{npp} y_tp x_t^T

    These play nicely with the prior. Let m = par(n) and S = sigma^{-2} I

    log p(A_n | A_m) = \sum_p log N(A_{np:} | A_{mp:}, S^{-1})
        = \sum_p -1/2 [A_{np:}^T S A_{np:} -2 A_{np:}^T S A_{mp:} + A_{mp:}^T S A_{mp:}]
        = \sum_p -1/2   [  A_{np:}^T   S  A_{np:}
                         + A_{np:}^T (-S) A_{mp:}
                         + A_{mp:}^T (-S) A_{np:}
                         + A_{mp:}^T   S  A_{mp:}  ]

        = \sum_p -1/2 A_{:p:}^T \hat{S} A_{:p:}

    where \hat{S} \in R^{ND \times ND} with blocks S_{mn} \in R^{D \times D}
    such that
        S_{mn} = S_{nm}^T = -S    if m = par(n)
        S_{nn}            = dS    where d is the degree of node n

    Finally, the root node gets an extra S from its prior.
    """

    def __init__(self, tree, P, D, affine=True,
                 S=None, S_scale=1.5, alpha_0=1, beta_0=1):
        """
        :param P: output dimension
        :param D: input dimension (without affine term)
        :param affine: whether or not to include an affine term
        :param S: prior precision for tree-structured weights
        :param S_scale: scaling factor for precision as a function of depth
        :param alpha_0: shape of gamma prior on observation noise precision
        :param beta_0: rate of gamma prior on observation noise precision
        """
        self.tree = tree
        self.affine = affine

        # Initialize output precision parameters
        self.P = P                                              # output dimension
        self.alpha_0 = alpha_0                                  # shape of gamma prior on precision
        self.beta_0 = beta_0                                    # rate of gamma prior on precision

        # Initialize tree-structured prior on regression weights
        self.D = D
        self.S = S if S is not None else np.eye(D + affine)     # hierarchical prior precision
        assert self.S.shape == (self.D + affine, self.D + affine)

        # Get the precision matrix from the tree-structured prior
        self.adj = adjacency(tree)                              # adjacency matrix of tree
        self.N = self.adj.shape[0]                              # number of nodes in tree
        self.L = (self.N + 1) // 2                              # number of leaves

        # Get the indices of the leaves
        self._ids = ids(tree)                                   # id's of nodes in tree (in 0...N-1)
        self.depths = depths(tree)                              # height of nodes in tree
        self.leaves = np.array([self._ids[l] for l in range(self.L)])   # id's of leaf nodes

        # Make the prior precision for the regression matrices
        J_0 = np.zeros((self.N, self.N))
        J_0[0,0] = 1
        def _prior(node, depth):
            if np.isscalar(node):
                return
            elif isinstance(node, tuple) and len(node) == 2:
                J_0[self._ids[node], self._ids[node]] += S_scale**depth
                J_0[self._ids[node], self._ids[node[0]]] -= S_scale**depth
                J_0[self._ids[node[0]], self._ids[node]] -= S_scale**depth
                J_0[self._ids[node[0]], self._ids[node[0]]] += S_scale**depth
                _prior(node[0], depth + 1)

                J_0[self._ids[node], self._ids[node]] += S_scale**depth
                J_0[self._ids[node], self._ids[node[1]]] -= S_scale**depth
                J_0[self._ids[node[1]], self._ids[node]] -= S_scale**depth
                J_0[self._ids[node[1]], self._ids[node[1]]] += S_scale**depth
                _prior(node[1], depth+1)
        _prior(tree, 0)
        self.J_0 = np.kron(J_0, self.S)

        # Initialize dynamics matrices A and dynamics noise precisions Q
        self.As = np.zeros((self.N, self.P, self.D))
        self.Qs = np.tile(np.eye(self.P)[None, :, :], (self.L, 1, 1))
        self.regressions = [Regression(A=self.As[j], sigma=np.linalg.inv(self.Qs[l]), affine=self.affine)
                            for (l,j) in enumerate(self.leaves)]

        # Resample to populate with a draw from the prior
        self.resample()

    def resample(self, data=[]):
        if not isinstance(data, list):
            if isinstance(data, tuple) and len(data) == 3:
                data = [data]
            else:
                raise Exception("Expected list or length-3 tuple (x, y, z)!")

        self._resample_dynamics_matrices(data)
        self._resample_dynamics_precisions(data)

    def _resample_dynamics_matrices(self, data=[]):
        N, P, D, L, affine = self.N, self.P, self.D, self.L, self.affine

        # Prior
        big_J = np.tile(self.J_0[:,:,None], (1, 1, P))
        big_h = np.zeros((N * (D + affine), P))

        # Likelihood
        for (x, y, z) in data:
            T = x.shape[0]

            # pad x as necessary
            if self.affine:
                x = np.column_stack((x, np.ones(T)))

            assert x.shape == (T, D + affine)
            assert y.shape == (T, P)

            # make sure z is ints between 0 and L-1
            assert z.shape == (T,)
            assert z.dtype == int and z.min() >= 0 and z.max() < L

            # compute the likelihood for each leaf node
            for l in range(L):
                inds = z == l
                xi = x[inds]
                yi = y[inds]
                xxT = (xi[:, :, None] * xi[:, None, :]).sum(0)
                yxT = (yi[:,:,None] * xi[:, None, :]).sum(0)

                j = self.leaves[l]
                for p in range(P):
                    big_J[j*D:(j+1)*D, j*D:(j+1)*D, p] += self.Q[l,p,p] * xxT
                    big_h[j*D:(j+1)*D, p] += self.Q[l,p,p] * yxT[p]

        # Sample As from their Gaussian conditional
        self.As = np.zeros((N, P, D + affine))
        for p in range(P):
            self.As[:,p,:] = sample_gaussian(J=big_J[:,:,p], h=big_h[:,p]).reshape((N, D + affine))

        # Set the parameters of the regression object
        for l, regression in zip(self.leaves, self.regressions):
            regression.A = self.As[l]

    def _resample_dynamics_precisions(self, data=[]):
        """
        Sample the precision matrix Q for the dynamics noise
        """
        L, P = self.L, self.P

        # Prior (for each leaf node)
        alpha = self.alpha_0 * np.ones((L,))
        beta = self.beta_0 * np.ones((L,P))

        # Likelihood
        for (x, y, z) in data:
            T = x.shape[0]

            # pad x as necessary
            if self.affine:
                x = np.column_stack((x, np.ones(T)))

            assert x.shape == (T, D + affine)
            assert y.shape == (T, P)

            # make sure z is ints between 0 and L-1
            assert z.shape == (T,)
            assert z.dtype == int and z.min() >= 0 and z.max() < L

            # compute the likelihood for each leaf node
            for l in range(L):
                inds = z == l
                xi = x[inds]
                yi = y[inds]
                xxT = (xi[:, :, None] * xi[:, None, :]).sum(0)
                yxT = (yi[:,:,None] * xi[:, None, :]).sum(0)
                n = inds.sum()

                # Resample from the gamma posterior
                AAT = np.array([np.outer(a,a) for a in self.As[l]])
                alpha[l] += n / 2.0
                beta[l] += 0.5 * ysq
                beta[l] += -1.0 * np.sum(yxT * self.As[l], axis=1)
                beta[l] += 0.5 * np.sum(AAT * xxT, axis=(1,2))

        # Set the parameters of the regression object
        for l, regression in enumerate(self.regressions):
            self.Qs[l] = np.diag(np.random.gamma(alpha[l], 1 / beta[l]))
            regression.sigma = np.diag(1 / np.diag(self.Qs[l]))


if __name__ == "__main__":
    np.random.seed(0)

    from pypolyagamma.binary_trees import balanced_binary_tree
    tree = balanced_binary_tree(16)
    P = 1
    D = 2

    import matplotlib.pyplot as plt
    plt.figure(figsize=(8,8))
    lim = 5.25
    nn = 3
    for ii in range(nn):
        for jj in range(nn):
            ax = plt.subplot(nn, nn, ii * nn + jj + 1)
            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)
            for sp in ax.spines:
                ax.spines[sp].set_visible(False)

            dynamics = TreeStructuredHierarchicalDynamics(tree, P, D)
            D = max(dynamics.depths.values())
            for n in range(dynamics.N):
                d = dynamics.depths[n]
                plt.plot(dynamics.As[n,0,0], dynamics.As[n,0,1], 'ko', alpha=1-d/(D+1), markersize=8)

            for d in range(D):
                plt.plot(-2*lim, -2*lim, 'ko', alpha=1-d/(D+1), markersize=4, label="depth {}".format(d))

            for (i, j) in zip(*dynamics.adj.nonzero()):
                plt.plot([dynamics.As[i,0,0], dynamics.As[j,0,0]],
                         [dynamics.As[i,0,1], dynamics.As[j,0,1]],
                         '-k', alpha=0.25)

            plt.xlim(-lim, lim)
            plt.ylim(-lim, lim)

            if ii == 0 and jj == nn-1:
                plt.legend(loc="lower right")

            plt.title("Tree {}".format(nn*ii + jj + 1))

    plt.tight_layout()
    plt.show()

