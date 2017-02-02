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

    TODO: We could include a redundant affine term b^T z_{t+1} as well.
          This would let us seamlessly handle the "input-only" model.
    """
    def __init__(self, num_states, covariate_dim,
                 mu_W=0.0, sigmasq_W=1.0,
                 logpi=None, W=None):
        self.num_states = num_states
        self.covariate_dim = covariate_dim

        if logpi is not None:
            assert logpi.shape == (num_states, num_states)
            self.logpi = logpi
        else:
            self.logpi = anp.zeros((num_states, num_states))

        if W is not None:
            assert W.shape == (covariate_dim, num_states)
            self.W = W
        else:
            self.W = anp.zeros((covariate_dim, num_states))

        # Assume diagonal prior on vec(W)
        self.mu_W = mu_W * anp.ones((covariate_dim, num_states))
        self.sigmasq_W = sigmasq_W * anp.ones((covariate_dim, num_states))

        # HMC params
        self.step_sz = 0.01
        self.accept_rate = 0.9
        self.target_accept_rate = 0.9

    ### Mean field
    #   TODO: Implement variational factors for W and logpi
    @property
    def expected_W(self):
        return self.W

    @property
    def expected_logpi(self):
        return self.logpi

    @property
    def exp_expected_logpi(self):
        P = anp.exp(self.logpi)
        P /= anp.sum(P, axis=1, keepdims=True)
        return P

    def get_log_trans_matrices(self, X):
        """
        Get log transition matrices as a function of X

        :param X: inputs/covariates
        :return: stack of transition matrices log A[t] \in Kin x Kout
        """
        # compute the contribution of the covariate to transition matrix
        psi_X = anp.dot(X, self.W)

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
        return anp.exp(log_trans_matrices)

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
        x0 = anp.concatenate((anp.ravel(self.logpi), anp.ravel(self.W)))
        xf, self.step_sz, self.accept_rate = \
            hmc(hmc_objective, grad_hmc_objective,
                step_sz=self.step_sz, n_steps=n_steps, q_curr=x0,
                negative_log_prob=False,
                adaptive_step_sz=True,
                avg_accept_rate=self.accept_rate)

        self.logpi = xf[:K**2].reshape((K, K))
        self.W = xf[K**2:].reshape((D, K))

    def meanfieldupdate(self, E_x, E_z, E_zzp1T):
        """
        We're going to need access to the log joints...
        :return:
        """
        raise NotImplementedError

    def get_vlb(self):
        return 0


class SoftmaxInputOnlyHMMTransitions(SoftmaxInputHMMTransitions):
    """
    Like above but with logpi constant for all rows (prev states)
    """

    def __init__(self, num_states, covariate_dim,
                 mu_W=0.0, sigmasq_W=1.0,
                 b=None, W=None):
        super(SoftmaxInputOnlyHMMTransitions, self).\
            __init__(num_states, covariate_dim,
                     mu_W=mu_W, sigmasq_W=sigmasq_W, W=W)

        if b is not None:
            assert b.shape == (num_states,)
            self.b = b

    @property
    def b(self):
        return self.logpi[0]

    @b.setter
    def b(self, value):
        assert value.shape == (self.num_states,)
        self.logpi = anp.tile(value[None,:], (self.num_states, 1))

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
        x0 = anp.concatenate((self.b, anp.ravel(self.W)))
        xf, self.step_sz, self.accept_rate = \
            hmc(hmc_objective, grad_hmc_objective,
                step_sz=self.step_sz, n_steps=n_steps, q_curr=x0,
                negative_log_prob=False,
                adaptive_step_sz=True,
                avg_accept_rate=self.accept_rate)

        self.b = xf[:K]
        self.W = xf[K:].reshape((D, K))



