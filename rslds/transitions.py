import numpy as np
from pypolyagamma import MultinomialRegression
from rslds.util import psi_to_pi, one_hot

class InputHMMTransitions(MultinomialRegression):
    """
    Model the transition probability as a multinomial
    regression whose inputs include the previous state
    as well as some covariates. For example, the covariates
    could be an external signal or even the latent states
    of a switching linear dynamical system.
    """
    def __init__(self, num_states, covariate_dim, **kwargs):
        super(InputHMMTransitions, self).\
            __init__(1, num_states, num_states+covariate_dim, **kwargs)
        self.num_states = num_states
        self.covariate_dim = covariate_dim

    def get_trans_matrices(self, X):
        """ return a stack of transition matrices, one for each input """
        mu, W = self.b, self.A
        W_markov = W[:,:self.num_states]
        W_covs = W[:,self.num_states:]

        # compute the contribution of the covariate to transmat
        psi_X = X.dot(W_covs.T)

        # compute the transmat stack without the covariate contributions
        psi_Z = W_markov.T

        # add the (K x K-1) and (T x K-1) matrices together such that they
        # broadcast into a [T x K x K-1] stack of trans matrices
        trans_psi = psi_X[:, None, :] + psi_Z

        # Add the (K-1) mean
        trans_psi += mu.reshape((self.D_out,))

        pi_stack = psi_to_pi(trans_psi, axis=2)
        pi_stack = np.ascontiguousarray(pi_stack)
        return pi_stack

    def resample(self, stateseqs=None, covseqs=None, omegas=None, **kwargs):
        """ conditioned on stateseqs and covseqs, stack up all of the data
        and use the PGMult class to resample """
        # assemble all of the discrete states into a dataset
        def align_lags(stateseq, covseq):
            prev_state = one_hot(stateseq[:-1], self.num_states)
            next_state = one_hot(stateseq[1:], self.num_states)
            return np.column_stack([prev_state, covseq]), next_state

        # Get the stacked previous states, covariates, and next states
        datas = [align_lags(z,x) for z, x in zip(stateseqs, covseqs)]

        # Clip the last data column since it is redundant
        # and not expected by the MultinomialRegression
        datas = [(x, y[:,:-1]) for x, y in datas]
        masks = [np.ones(y.shape, dtype=bool) for _,y in datas]
        super(InputHMMTransitions, self).\
            resample(datas, mask=masks, omega=omegas)


class StickyInputHMMTransitions(InputHMMTransitions):
    """
    Introduce a "stickiness" parameter to capture the tendency
    to stay in the same state. In the standard InputHMM model,

    psi_t = W_markov * I[z_{t-1}] + W_input * x_{t-1} + b.

    Now we want W_markov[k,k] ~ N(kappa, sigma^2) with kappa > 0,
    and W_markov[k,j] ~ N(0, sigma^2) for j \neq k.
    """
    def __init__(self, num_states, covariate_dim, kappa=1.0, **kwargs):
        assert "mu_A" not in kwargs, "StickyInputHMMTransitions overrides provided mu_A"
        mu_A = np.zeros((num_states-1, num_states+covariate_dim))
        mu_A[:,:num_states-1] = kappa * np.eye(num_states-1)
        kwargs["mu_A"] = mu_A

        super(StickyInputHMMTransitions, self).\
            __init__(num_states, covariate_dim, **kwargs)


class InputOnlyHMMTransitions(InputHMMTransitions):
    """
    Model the transition probability as a multinomial
    regression that depends only on the covariates.
    For example, the covariates
    could be an external signal or even the latent states
    of a switching linear dynamical system.
    """
    def __init__(self, num_states, covariate_dim, **kwargs):
        super(InputOnlyHMMTransitions, self).\
            __init__(num_states, covariate_dim, **kwargs)
        self.A[:,:self.num_states] = 0

    def resample(self, stateseqs=None, covseqs=None, omegas=None, **kwargs):
        """ conditioned on stateseqs and covseqs, stack up all of the data
        and use the PGMult class to resample """

        # Zero out the previous state in the regression
        def align_lags(stateseq, covseq):
            prev_state = np.zeros((stateseq.shape[0]-1, self.num_states))
            next_state = one_hot(stateseq[1:], self.num_states)
            return np.column_stack([prev_state, covseq]), next_state

        # Get the stacked previous states, covariates, and next states
        datas = [align_lags(z,x) for z, x in zip(stateseqs, covseqs)]

        # Clip the last data column since it is redundant
        # and not expected by the MultinomialRegression
        datas = [(x, y[:,:-1]) for x, y in datas]
        masks = [np.ones(y.shape, dtype=bool) for _,y in datas]
        super(InputHMMTransitions, self).\
            resample(datas, mask=masks, omega=omegas)

        # Zero out the weights on the previous state
        self.A[:, :self.num_states] = 0


class StickyInputOnlyHMMTransitions(InputHMMTransitions):
    """
    Hacky way to implement the sticky input only model in which

    psi_{t,k} | z_{t-1} =
        kappa_k + w_j \dot x_{t-1} + b_j     if z_{t-1} = k
        0       + w_j \dot x_{t-1} + b_j     otherwise

    We just set the prior such that the off-diagonal entries of
    W_{markov} are effectively zero by setting the variance of
    these entries to be super small.
    """
    def __init__(self, num_states, covariate_dim, kappa=1.0, sigmasq_kappa=1e-8, **kwargs):

        # Set the mean of A
        K, D = num_states, covariate_dim
        assert "mu_A" not in kwargs, "StickyInputHMMTransitions overrides provided mu_A"
        mu_A = np.zeros((K-1, K+D))
        mu_A[:,:K-1] = kappa * np.eye(K-1)
        kwargs["mu_A"] = mu_A

        # Set the covariance of A
        if "sigmasq_A" in kwargs:
            assert np.isscalar(kwargs["sigmasq_A"])
            sig0 = kwargs["sigmasq_A"]
        else:
            sig0 = 1.0

        sigmasq_A = np.zeros((K-1, K+D, K+D))
        for k in range(K-1):
            sigmasq_A[k, :K, :K] = 1e-8 * np.eye(K)
            sigmasq_A[k,  k,  k] = sigmasq_kappa
            sigmasq_A[k, K:, K:] = sig0 * np.eye(D)

        kwargs["sigmasq_A"] = sigmasq_A

        super(StickyInputOnlyHMMTransitions, self).\
            __init__(num_states, covariate_dim, **kwargs)

import autograd.numpy as anp
import autograd.scipy.misc as amisc
from autograd import grad

class SoftmaxInputHMMTransitions(object):
    """
    Like above but with a softmax transition model.

    log p(z_{t+1} | z_t, x_t) = z_t^T log pi z_{t+1} + x_t^T W z_{t+1} - Z

    where Z = log ( \sum_k exp { z_t^T log pi e_k + x_t^T W e_k} )

    CONVENTION: logpi[i,j] = Pr(z_{t+1} = j | z_t = i).

    TODO: We could include a redundant affine term b^T z_{t+1} as well.
          This would let us seamlessly handle the "input-only" model.
    """
    def __init__(self, num_states, covariate_dim,
                 mu_0=0.0, Sigma_0=1.0,
                 logpi=None, W=None):
        self.num_states = num_states
        self.covariate_dim = covariate_dim
        self.D_out = num_states
        self.D_in = num_states + covariate_dim

        if logpi is not None:
            assert logpi.shape == (num_states, num_states)
            self.logpi = logpi
        else:
            self.logpi = np.zeros((num_states, num_states))

        if W is not None:
            assert W.shape == (covariate_dim, num_states)
            self.W = W
        else:
            self.W = np.zeros((covariate_dim, num_states))

        # Assume diagonal prior on vec(W)
        # self.mu_W = mu_W * anp.ones((covariate_dim, num_states))
        # self.sigmasq_W = sigmasq_W * anp.ones((covariate_dim, num_states))

        mu_0 = np.zeros(self.D_in) if mu_0 is None else mu_0
        Sigma_0 = np.eye(self.D_in) if Sigma_0 is None else Sigma_0
        assert mu_0.shape == (self.D_in,)
        assert Sigma_0.shape == (self.D_in, self.D_in)
        self.h_0 = np.linalg.solve(Sigma_0, mu_0)
        self.J_0 = np.linalg.inv(Sigma_0)

        # HMC params
        self.step_sz = 0.01
        self.accept_rate = 0.9
        self.target_accept_rate = 0.9

        # Mean field natural parameters
        self.mf_J = np.array([self.J_0.copy() for _ in range(self.D_out)])
        self.mf_h = np.array([self.h_0.copy() for Jd in self.mf_J])
        self._mf_Sigma = self._mf_mu = self._mf_mumuT = None

    ### HMC
    def get_log_trans_matrices(self, X):
        """
        Get log transition matrices as a function of X

        :param X: inputs/covariates
        :return: stack of transition matrices log A[t] \in Kin x Kout
        """
        # compute the contribution of the covariate to transition matrix
        psi_X = np.dot(X, self.W)

        # add the (T x Kout) and (Kin x Kout) matrices together such that they
        # broadcast into a (T x Kin x Kout) stack of matrices
        psi = psi_X[:, None, :] + self.logpi

        # apply softmax and normalize over outputs
        log_trans_matrices = psi - amisc.logsumexp(psi, axis=2, keepdims=True)

        return log_trans_matrices

    def get_trans_matrices(self, X):
        """
        Get transition matrices as a function of X

        :param X: inputs/covariates
        :return: stack of transition matrices A[t] \in Kin x Kout
        """
        log_trans_matrices = self.get_log_trans_matrices(X)
        return np.exp(log_trans_matrices)

    def joint_log_probability(self, logpi, W, stateseqs, covseqs):
        K, D = self.num_states, self.covariate_dim

        # Compute the objective
        ll = 0
        for z, x in zip(stateseqs, covseqs):
            T = z.size
            assert x.ndim == 2 and x.shape[0] == T - 1
            z_prev = one_hot(z[:-1], K)
            z_next = one_hot(z[1:], K)

            # Numerator
            tmp = anp.dot(z_prev, logpi) + anp.dot(x, W)
            ll += anp.sum(tmp * z_next)

            # Denominator
            Z = amisc.logsumexp(tmp, axis=1)
            ll -= anp.sum(Z)

        return ll

    def resample(self, stateseqs=None, covseqs=None,
                 n_steps=10, **kwargs):
        K, D = self.num_states, self.covariate_dim

        # Run HMC
        from hips.inference.hmc import hmc
        def hmc_objective(params):
            # Unpack params
            K, D = self.num_states, self.covariate_dim
            logpi = params[:K ** 2].reshape((K, K))
            W = params[K ** 2:].reshape((D, K))
            return self.joint_log_probability(logpi, W, stateseqs, covseqs)

        grad_hmc_objective = grad(hmc_objective)
        x0 = np.concatenate((np.ravel(self.logpi), np.ravel(self.W)))
        xf, self.step_sz, self.accept_rate = \
            hmc(hmc_objective, grad_hmc_objective,
                step_sz=self.step_sz, n_steps=n_steps, q_curr=x0,
                negative_log_prob=False,
                adaptive_step_sz=True,
                avg_accept_rate=self.accept_rate)

        self.logpi = xf[:K**2].reshape((K, K))
        self.W = xf[K**2:].reshape((D, K))

    ### Mean field
    @property
    def expected_W(self):
        # _mf_mu = [E[logpi],  E[W]]
        return self._mf_mu[:, self.num_states:].T

    @property
    def expected_logpi(self):
        # _mf_mu = [E[logpi],  E[W]]
        return self._mf_mu[:, :self.num_states].T

    @property
    def exp_expected_logpi(self):
        P = np.exp(self.expected_logpi)
        P /= np.sum(P, axis=1, keepdims=True)
        return P

    @property
    def expected_WWT(self):
        return self._mf_mumuT[:,self.num_states:, self.num_states:]

    @property
    def expected_logpi_WT(self):
        return self._mf_mumuT[:, :self.num_states, self.num_states:]

    @property
    def expected_logpi_logpiT(self):
        return self._mf_mumuT[:, :self.num_states, :self.num_states]

    def meanfieldupdate(self, stats, prob=1.0, stepsize=1.0):
        """
        Update the expected transition matrix with a bunch of stats
        :param stats: E_zp1_uT, E_uuT, E_u, a, lambda_bs from the states model
        :param prob: minibatch probability
        :param stepsize: svi step size
        """
        E_u_zp1T, E_uuT, E_u, a, lambda_bs = stats

        update_param = lambda oldv, newv, stepsize: \
            oldv * (1 - stepsize) + newv * stepsize

        # Update statistics each row of A
        for k in range(self.D_out):
            # Jk = self.J_0 + 2 * lambda_bs[:,k][:,None,None] * E_uuT
            Jk = self.J_0 + 2 * np.einsum('t, tij -> ij', lambda_bs[:, k], E_uuT) / prob
            hk = self.h_0 + E_u_zp1T[:, :, k].sum(0) / prob
            hk -= np.einsum('t, ti -> i', (0.5 - 2 * lambda_bs[:, k] * a), E_u) / prob

            # Update the mean field natural parameters
            self.mf_J[k] = update_param(self.mf_J[k], Jk, stepsize)
            self.mf_h[k] = update_param(self.mf_h[k], hk, stepsize)

        self._set_standard_expectations()

        # Update log pi and W with meanfield expectations
        self.logpi = self.expected_logpi
        self.W = self.expected_W

    def _set_standard_expectations(self):
        # Compute expectations
        self._mf_Sigma = np.array([np.linalg.inv(Jk) for Jk in self.mf_J])
        self._mf_mu = np.array([np.dot(Sk, hk) for Sk, hk in zip(self._mf_Sigma, self.mf_h)])
        self._mf_mumuT = np.array([Sd + np.outer(md, md)
                                   for Sd, md in zip(self._mf_Sigma, self._mf_mu)])

    def get_vlb(self):
        # TODO
        return 0

    def _initialize_mean_field(self):
        self.mf_J = np.array([1e2 * self.J_0.copy() for _ in range(self.D_out)])
        # Initializing with mean zero is pathological. Break symmetry by starting with sampled A.
        # self.mf_h_A = np.array([self.h_0.copy() for _ in range(D_out)])
        A = np.hstack((self.logpi, self.W.T))
        self.mf_h = np.array([Jd.dot(Ad) for Jd, Ad in zip(self.mf_J, A)])
        self._set_standard_expectations()


# class SoftmaxInputOnlyHMMTransitions(SoftmaxInputHMMTransitions):
#     """
#     Like above but with logpi constant for all rows (prev states)
#     Assume a variational factor q(b) q(W)
#     """
#
#     def __init__(self, num_states, covariate_dim,
#                  mu_0=None, Sigma_0=None,
#                  b=None, W=None):
#         super(SoftmaxInputOnlyHMMTransitions, self).\
#             __init__(num_states, covariate_dim,
#                      mu_0=mu_0, Sigma_0=Sigma_0, W=W)
#
#         if b is not None:
#             assert b.shape == (num_states,)
#             self.b = b
#
#         delattr(self, "J_0")
#         delattr(self, "h_0")
#         delattr(self, "mf_J")
#         delattr(self, "mf_h")
#
#         self.Jb_0 = 0.0001 * np.eye(1)
#         self.hb_0 = np.zeros(1)
#         self.JW_0 = 0.0001 * np.eye(covariate_dim)
#         self.hW_0 = np.zeros(covariate_dim)
#
#         # Mean field natural parameters
#         self.mf_J_b = np.array([np.eye(1) for _ in range(self.D_out)])
#         self.mf_h_b = np.array([self.hb_0.copy() for _ in self.mf_J_b])
#
#         self.mf_J_W = np.array([self.JW_0.copy() for _ in range(self.D_out)])
#         self.mf_h_W = np.array([self.hW_0.copy() for _ in self.mf_J_W])
#
#         self._mf_Sigma_W = self._mf_mu_W = self._mf_mumu_WT = None
#         self._mf_Sigma_b = self._mf_mu_b = self._mf_mumu_bT = None
#
#
#     @property
#     def b(self):
#         return self.logpi[0]
#
#     @b.setter
#     def b(self, value):
#         assert value.shape == (self.num_states,)
#         self.logpi = np.tile(value[None, :], (self.num_states, 1))
#
#     def resample(self, stateseqs=None, covseqs=None,
#                  n_steps=10, step_sz=0.01, **kwargs):
#         K, D = self.num_states, self.covariate_dim
#
#         # Run HMC
#         from hips.inference.hmc import hmc
#
#         def hmc_objective(params):
#             # Unpack params
#             assert params.size == K + K * D
#             assert params.ndim == 1
#             b = params[:K]
#             logpi = anp.tile(b[None, :], (K, 1))
#             W = params[K:].reshape((D, K))
#             return self.joint_log_probability(logpi, W, stateseqs, covseqs)
#
#         # hmc_objective = lambda params: self.joint_log_probability(params, stateseqs, covseqs)
#         grad_hmc_objective = grad(hmc_objective)
#         x0 = np.concatenate((self.b, np.ravel(self.W)))
#         xf, self.step_sz, self.accept_rate = \
#             hmc(hmc_objective, grad_hmc_objective,
#                 step_sz=self.step_sz, n_steps=n_steps, q_curr=x0,
#                 negative_log_prob=False,
#                 adaptive_step_sz=True,
#                 avg_accept_rate=self.accept_rate)
#
#         self.b = xf[:K]
#         self.W = xf[K:].reshape((D, K))
#
#     ### Mean field
#     @property
#     def expected_W(self):
#         # _mf_mu = [E[logpi],  E[W]]
#         return self._mf_mu_W.T
#
#     @property
#     def expected_b(self):
#         return self._mf_mu_b
#
#     @property
#     def expected_logpi(self):
#         return np.tile(self.expected_b[None,:], (self.num_states, 1))
#
#     @property
#     def expected_WWT(self):
#         return self._mf_mumuT_W
#
#     @property
#     def expected_logpi_WT(self):
#         # return self._mf_mumuT[:, :self.num_states, self.num_states:]
#         # return self.expected_logpi.dot(self.expected_W.T)
#         return np.array([np.outer(self.expected_b[k] * np.ones(self.num_states),
#                                   self.expected_W[:,k])
#                          for k in range(self.D_out)])
#
#     @property
#     def expected_logpi_logpiT(self):
#         K = self.num_states
#         return np.array([bksq * np.ones((K,K)) for bksq in self._mf_musq_b])
#
#     def meanfieldupdate(self, stats, prob=1.0, stepsize=1.0):
#         """
#         Update the expected transition matrix with a bunch of stats
#         :param stats: E_zp1_uT, E_uuT, E_u, a, lambda_bs from the states model
#         :param prob: minibatch probability
#         :param stepsize: svi step size
#         """
#         num_states, num_covariates = self.num_states, self.covariate_dim
#         E_u_zp1T, E_uuT, E_u, a, lambda_bs = stats
#
#         # Split u into z and x
#         E_z_zp1T, E_x_zp1T = E_u_zp1T[:, :num_states, :], E_u_zp1T[:, num_states:, :]
#         E_zzT, E_zxT, E_xxT = E_uuT[:, :num_states, :num_states], \
#                               E_uuT[:, :num_states, num_states:], \
#                               E_uuT[:, num_states:, num_states:]
#
#         E_z, E_x = E_u[:, :num_states], E_u[:, num_states:]
#
#         update_param = lambda oldv, newv, stepsize: \
#             oldv * (1 - stepsize) + newv * stepsize
#
#         # Update statistics each row of A
#         # Since our variational family is factorized here, we need to
#         # update q(W_{k:}) holding q(b) fixed and vice versa.  This just
#         # means we need to split J into pieces that depend on W and b separately.
#         # for k in range(self.D_out):
#         #     # Jk = self.J_0 + 2 * lambda_bs[:,k][:,None,None] * E_uuT
#         #     Jk = self.JW_0 + 2 * np.einsum('t, tij -> ij', lambda_bs[:, k], E_xxT) / prob
#         #     hk = self.hW_0 + E_x_zp1T[:, :, k].sum(0) / prob
#         #     hk -= 2 * np.einsum('i, t,  tij -> j', self.expected_b, lambda_bs[:, k], E_zxT) / prob
#         #     hk -= np.einsum('t, ti -> i', (0.5 - 2 * lambda_bs[:, k] * a), E_x) / prob
#         #
#         #     # The only quadratic term in W is Jk_WWT
#         #     self.mf_J_W[k] = update_param(self.mf_J_W[k], Jk, stepsize)
#         #     self.mf_h_W[k] = update_param(self.mf_h_W[k], hk, stepsize)
#         # self._set_standard_expectations()
#
#         # Now update b with potentials from each output k
#         for k in range(self.D_out):
#             # Jk = self.J_0 + 2 * lambda_bs[:,k][:,None,None] * E_uuT
#             Jbk = self.Jb_0 + 2 * np.einsum('t, ti -> ', lambda_bs[:, k], E_z) / prob
#             hbk = self.hb_0 + E_z_zp1T[:, :, k].sum() / prob
#             hbk -= 2 * np.einsum('j, t,  tij -> ', self.expected_W[:,k], lambda_bs[:, k], E_zxT) / prob
#             hbk -= np.einsum('t, ti -> ', (0.5 - 2 * lambda_bs[:, k] * a), E_z) / prob
#
#             self.mf_J_b[k] = update_param(self.mf_J_b[k], Jbk, stepsize)
#             self.mf_h_b[k] = update_param(self.mf_h_b[k], hbk, stepsize)
#
#         self._set_standard_expectations()
#
#         # Update log pi and W with meanfield expectations
#         self.b = self.expected_b
#         self.W = self.expected_W
#
#     def _set_standard_expectations(self):
#         # Compute expectations
#         # self._mf_Sigma_b = np.linalg.inv(self.mf_J_b)
#         # self._mf_mu_b = np.dot(self._mf_Sigma_b, self.mf_h_b)
#         # self._mf_mumuT_b = self._mf_Sigma_b + np.outer(self._mf_mu_b, self._mf_mu_b)
#         self._mf_Sigma_b = np.array([1./Jk[0,0] for Jk in self.mf_J_b])
#         self._mf_mu_b = np.array([Sk * hk[0,0] for Sk, hk in zip(self._mf_Sigma_b, self.mf_h_b)])
#         self._mf_musq_b = np.array([Sk + mk**2 for Sk, mk in zip(self._mf_Sigma_b, self._mf_mu_b)])
#
#         self._mf_Sigma_W = np.array([np.linalg.inv(Jk) for Jk in self.mf_J_W])
#         self._mf_mu_W = np.array([np.dot(Sk, hk) for Sk, hk in zip(self._mf_Sigma_W, self.mf_h_W)])
#         self._mf_mumuT_W = np.array([Sd + np.outer(md, md)
#                                    for Sd, md in zip(self._mf_Sigma_W, self._mf_mu_W)])
#
#     def _initialize_mean_field(self):
#         self.mf_J_W = np.array([1e2 * self.JW_0.copy() for _ in range(self.D_out)])
#         self.mf_J_b = 1e2 * np.ones((self.num_states, 1, 1))
#
#         # Initializing with mean zero is pathological. Break symmetry by starting with sampled A.
#         # self.mf_h_A = np.array([self.h_0.copy() for _ in range(D_out)])
#         self.mf_h_W = np.array([Jd.dot(Wd) for Jd, Wd in zip(self.mf_J_W, self.W.T)])
#         self.mf_h_b = np.array([Jd.dot(bd) for Jd, bd in zip(self.mf_J_b, self.b)])
#         self._set_standard_expectations()



class SoftmaxInputOnlyHMMTransitions(SoftmaxInputHMMTransitions):
    """
    Like above but with logpi constant for all rows (prev states)
    Assume a variational factor q(b) q(W)
    """

    def __init__(self, num_states, covariate_dim,
                 mu_0=None, Sigma_0=None,
                 b=None, W=None):
        super(SoftmaxInputOnlyHMMTransitions, self).\
            __init__(num_states, covariate_dim,
                     mu_0=mu_0, Sigma_0=Sigma_0, W=W)

        if b is not None:
            assert b.shape == (num_states,)
            self.b = b

        self.D_in = 1 + self.covariate_dim

        mu_0 = np.zeros(self.D_in) if mu_0 is None else mu_0
        Sigma_0 = np.eye(self.D_in) if Sigma_0 is None else Sigma_0
        assert mu_0.shape == (self.D_in,)
        assert Sigma_0.shape == (self.D_in, self.D_in)
        self.h_0 = np.linalg.solve(Sigma_0, mu_0)
        self.J_0 = np.linalg.inv(Sigma_0)

        # Mean field natural parameters
        self.mf_J = np.array([self.J_0.copy() for _ in range(self.D_out)])
        self.mf_h = np.array([self.h_0.copy() for Jd in self.mf_J])
        self._mf_Sigma = self._mf_mu = self._mf_mumuT = None


    @property
    def b(self):
        return self.logpi[0]

    @b.setter
    def b(self, value):
        assert value.shape == (self.num_states,)
        self.logpi = np.tile(value[None, :], (self.num_states, 1))

    def resample(self, stateseqs=None, covseqs=None,
                 n_steps=10, step_sz=0.01, **kwargs):
        K, D = self.num_states, self.covariate_dim

        # Run HMC
        from hips.inference.hmc import hmc

        def hmc_objective(params):
            # Unpack params
            assert params.size == K + K * D
            assert params.ndim == 1
            b = params[:K]
            logpi = anp.tile(b[None, :], (K, 1))
            W = params[K:].reshape((D, K))
            return self.joint_log_probability(logpi, W, stateseqs, covseqs)

        # hmc_objective = lambda params: self.joint_log_probability(params, stateseqs, covseqs)
        grad_hmc_objective = grad(hmc_objective)
        x0 = np.concatenate((self.b, np.ravel(self.W)))
        xf, self.step_sz, self.accept_rate = \
            hmc(hmc_objective, grad_hmc_objective,
                step_sz=self.step_sz, n_steps=n_steps, q_curr=x0,
                negative_log_prob=False,
                adaptive_step_sz=True,
                avg_accept_rate=self.accept_rate)

        self.b = xf[:K]
        self.W = xf[K:].reshape((D, K))

    ### Mean field
    @property
    def expected_W(self):
        # Recall, _mf_mu = [E[b],  E[W.T]]  and is K x (1+D)
        return self._mf_mu[:, 1:].T

    @property
    def expected_b(self):
        return self._mf_mu[:, 0]

    @property
    def expected_logpi(self):
        return np.tile(self.expected_b[None,:], (self.num_states, 1))

    @property
    def expected_WWT(self):
        return self._mf_mumuT[:, 1:, 1:]

    @property
    def expected_bsq(self):
        return self._mf_mumuT[:, 0, 0]

    @property
    def expected_bWT(self):
        return self._mf_mumuT[:, 0, 1:]

    @property
    def expected_logpi_WT(self):
        return np.tile(self.expected_bWT[:,None,:], (1, self.num_states, 1))

    @property
    def expected_logpi_logpiT(self):
        return np.tile(self.expected_bsq[:,None,None], (1, self.num_states, self.num_states))

    def meanfieldupdate(self, stats, prob=1.0, stepsize=1.0):
        """
        Update the expected transition matrix with a bunch of stats
        :param stats: E_zp1_uT, E_uuT, E_u, a, lambda_bs from the states model
        :param prob: minibatch probability
        :param stepsize: svi step size
        """
        K, D = self.num_states, self.covariate_dim
        E_u_zp1T, E_uuT, E_u, a, lambda_bs = stats
        T = E_u.shape[0]

        # Remember u = [z, x]
        # Combine statistics across all preceding states, z,
        # to get statistics of shape (1+covariate_dim)
        E_u_zp1T_new = np.zeros((T, 1+D, K))
        E_u_zp1T_new[:, 0, :] = E_u_zp1T[:, :K, :].sum(axis=1)
        E_u_zp1T_new[:, 1:, :] = E_u_zp1T[:, K:, :]
        E_u_zp1T = E_u_zp1T_new

        E_uuT_new = np.zeros((T, 1+D, 1+D))
        E_uuT_new[:,0,0] = E_uuT[:, :K, :K].sum(axis=(1,2))
        E_uuT_new[:,0,1:] = E_uuT[:, :K, K:].sum(axis=1)
        E_uuT_new[:,1:,0] = E_uuT[:, K:, :K].sum(axis=2)
        E_uuT_new[:,1:,1:] = E_uuT[:, K:, K:]
        E_uuT = E_uuT_new

        E_u_new = np.zeros((T, 1+D))
        E_u_new[:,0] = E_u[:, :K].sum(axis=1)
        E_u_new[:,1:] = E_u[:, K:]
        E_u = E_u_new

        update_param = lambda oldv, newv, stepsize: \
            oldv * (1 - stepsize) + newv * stepsize

        # Update statistics each row of A
        for k in range(self.D_out):
            # Jk = self.J_0 + 2 * lambda_bs[:,k][:,None,None] * E_uuT
            Jk = self.J_0 + 2 * np.einsum('t, tij -> ij', lambda_bs[:, k], E_uuT) / prob
            hk = self.h_0 + E_u_zp1T[:, :, k].sum(0) / prob
            hk -= np.einsum('t, ti -> i', (0.5 - 2 * lambda_bs[:, k] * a), E_u) / prob

            # Update the mean field natural parameters
            self.mf_J[k] = update_param(self.mf_J[k], Jk, stepsize)
            self.mf_h[k] = update_param(self.mf_h[k], hk, stepsize)

        self._set_standard_expectations()

        # Update log pi and W with meanfield expectations
        self.logpi = self.expected_logpi
        self.W = self.expected_W

    def _set_standard_expectations(self):
        # Compute expectations
        self._mf_Sigma = np.array([np.linalg.inv(Jk) for Jk in self.mf_J])
        self._mf_mu = np.array([np.dot(Sk, hk) for Sk, hk in zip(self._mf_Sigma, self.mf_h)])
        self._mf_mumuT = np.array([Sd + np.outer(md, md)
                                   for Sd, md in zip(self._mf_Sigma, self._mf_mu)])

    def _initialize_mean_field(self):
        self.mf_J = np.array([1e2 * self.J_0.copy() for _ in range(self.D_out)])

        # Initializing with given b and W
        A = np.hstack((self.b[:,None], self.W.T))
        self.mf_h = np.array([Jk.dot(ak) for Jk, ak in zip(self.mf_J, A)])
        self._set_standard_expectations()
