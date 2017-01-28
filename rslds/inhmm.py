import numpy as np

from pybasicbayes.util.stats import sample_discrete

from pyhsmm.models import _HMMGibbsSampling
from pyhsmm.internals.hmm_states import HMMStatesEigen
from pyhsmm.internals import initial_state

from pypolyagamma.distributions import MultinomialRegression

from rslds.util import one_hot, psi_to_pi



##############################################################################
# Outline for extending an HMM to IO-HMM with mix-ins                        #
#    1. create a transitions base class that creates stacks of transition    #
#       matrices given a matrix of inputs                                    #
#    2. create a Gibbs mix-in that resamples transition parameters given     #
#       realized discrete state sequences                                    #
#                                                                            #
##############################################################################

### Polya-gamma input model
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


class SoftmaxInputHMMTransitions:
    """
    Like above but with a softmax transition model.

    log p(z_{t+1} | z_t, x_t) = z_t^T log pi z_{t+1} + x_t^T W z_{t+1} - Z

    where Z = log ( \sum_k exp { z_t^T log pi e_k + x_t^T W e_k} )
    """
    # TODO: This needs to be implemented.  The updates for the
    #       transition parameters will no longer be conjugate though.
    pass


class InputHMMStates(HMMStatesEigen):

    def __init__(self, covariates, *args, **kwargs):
        self.covariates = covariates
        super(InputHMMStates, self).__init__(*args, **kwargs)

    @property
    def trans_matrix(self):
        return self.model.trans_distn.get_trans_matrices(self.covariates)

    @staticmethod
    def _sample_markov_hetero(T, trans_matrix_seq, init_state_distn):
        """utility function - just like sample_markov but with a stack of trans mats"""
        out = np.empty(T, dtype=np.int32)
        out[0] = sample_discrete(init_state_distn.ravel())
        for t in range(1, T):
            out[t] = sample_discrete(trans_matrix_seq[t - 1, out[t - 1], :].ravel())
        return out

    def generate_states(self):
        self.stateseq = InputHMMStates._sample_markov_hetero(
            T=self.T,
            trans_matrix_seq=self.trans_matrix,  # stack of matrices
            init_state_distn=np.ones(self.num_states) / float(self.num_states))

    ### forward messages and backward resampling
    def sample_backwards_normalized(self, alphan):
        Tmat = self.trans_matrix
        AT = np.ascontiguousarray(np.array([Tmat[t, :, :].T for t in range(Tmat.shape[0])]))
        self.stateseq = self._sample_backwards_normalized(alphan, AT)


### HMM Model with time varying covariates
class _InputHMMMixin(object):

    # Subclasses must specify the type of transition model
    _trans_class = None

    # custom init method, just so we call custom input trans class stuff
    def __init__(self,
                 obs_distns,
                 D_in=0,
                 trans_distn=None, trans_params={},
                 alpha=None, alpha_a_0=None, alpha_b_0=None, trans_matrix=None,
                 init_state_distn=None, init_state_concentration=None, pi_0=None,
                 ):
        self.obs_distns = obs_distns
        self.states_list = []
        self.D_in = D_in

        # our trans class
        if trans_distn is None:
            self.trans_distn = self._trans_class(num_states=len(obs_distns),
                                                 covariate_dim=D_in,
                                                 **trans_params)
        else:
            self.trans_distn = trans_distn

        if init_state_distn is not None:
            if init_state_distn == 'uniform':
                self.init_state_distn = initial_state.UniformInitialState(model=self)
            else:
                self.init_state_distn = init_state_distn
        else:
            self.init_state_distn = self._init_state_class(
                    model=self,
                    init_state_concentration=init_state_concentration,
                    pi_0=pi_0)

        self._clear_caches()

    # custom add_data - includes a covariates arg
    def add_data(self, data, covariates=None, **kwargs):
        # NOTE! Our convention is that covariates[t] drives the
        # NOTE! transition matrix going into time t. However, for
        # NOTE! implementation purposes, it is easier if these inputs
        # NOTE! are lagged so that covariates[t] drives the input to
        # NOTE! z_{t+1}. Then, we only have T-1 inputs for the T-1
        # NOTE! transition matrices in the heterogeneous model.

        # Offset the covariates by one so that
        # the inputs at time {t-1} determine the transition matrix
        # from z_{t-1} to z_{t}.
        offset_covariates = covariates[1:]
        self.states_list.append(
                self._states_class(
                    model=self, data=data,
                    covariates=offset_covariates, **kwargs))

    def generate(self, T=100, covariates=None, keep=True):
        if covariates is None:
            covariates = np.zeros((T, self.D_in))
        else:
            assert covariates.ndim == 2 and \
                   covariates.shape[0] == T
        s = self._states_class(model=self, covariates=covariates[1:], T=T, initialize_from_prior=True)
        data = self._generate_obs(s)
        if keep:
            self.states_list.append(s)
        return (data, covariates), s.stateseq

    def resample_trans_distn(self):
        # TODO: concatenate before sending these.
        # It should speed up gradient calcs by removing the for loop
        self.trans_distn.resample(
            stateseqs=[s.stateseq for s in self.states_list],
            covseqs=[s.covariates for s in self.states_list],
        )
        self._clear_caches()


class InputHMM(_InputHMMMixin, _HMMGibbsSampling):
    _trans_class = InputHMMTransitions
    _states_class = InputHMMStates


from autoregressive.models import _ARMixin
from autoregressive.util import AR_striding
class InputARHMM(_InputHMMMixin, _ARMixin, _HMMGibbsSampling):
    _trans_class = InputHMMTransitions
    _states_class = InputHMMStates

    def add_data(self, data, covariates=None, strided=False, **kwargs):
        if covariates is None:
            covariates = np.zeros((data.shape[0], 0))

        strided_data = AR_striding(data,self.nlags) if not strided else data
        lagged_covariates = covariates[self.nlags:]
        assert strided_data.shape[0] == lagged_covariates.shape[0]
        super(InputARHMM, self).add_data(data=strided_data,covariates=lagged_covariates,**kwargs)


class RecurrentARHMM(InputARHMM):
    """
    In the "recurrent" version, the data are the covariates.
    """
    # TODO: We should also allow for (data, inputs) to be covariates
    def add_data(self, data, **kwargs):
        assert "covariates" not in kwargs
        covariates = np.row_stack((np.zeros(self.D),
                                   data[:-1]))
        super(RecurrentARHMM, self).add_data(data, covariates=covariates, **kwargs)

    def generate(self, T=100, keep=True, init_data=None, covariates=None, with_noise=True):
        from pybasicbayes.util.stats import sample_discrete
        # Generate from the prior and raise exception if unstable
        K, n = self.num_states, self.D

        # Initialize discrete state sequence
        pi_0 = self.init_state_distn.pi_0
        dss = np.empty(T, dtype=np.int32)
        dss[0] = sample_discrete(pi_0.ravel())

        data = np.empty((T, n), dtype='double')
        if init_data is None:
            data[0] = np.random.randn(n)
        else:
            data[0] = init_data

        for t in range(1, T):
            # Sample discrete state given previous continuous state
            A = self.trans_distn.get_trans_matrices(data[t-1:t])[0]
            dss[t] = sample_discrete(A[dss[t-1], :])

            # Sample continuous state given current discrete state
            if with_noise:
                data[t] = self.obs_distns[dss[t]]. \
                    rvs(np.hstack((data[t-1][None, :])))
            else:
                data[t] = self.obs_distns[dss[t]]. \
                    predict(np.hstack((data[t - 1][None, :])))
            assert np.all(np.isfinite(data[t])), "RARHMM appears to be unstable!"

        # TODO:
        # if keep:
        #     ...

        return data, dss


class InputOnlyRecurrentARHMM(RecurrentARHMM):
    _trans_class = InputOnlyHMMTransitions


class StickyInputOnlyRecurrentARHMM(RecurrentARHMM):
    _trans_class = StickyInputOnlyHMMTransitions