import numpy as np
from pybasicbayes.util.stats import sample_discrete
from pyhsmm.internals import initial_state
from pyhsmm.internals.hmm_states import HMMStatesEigen
from pyhsmm.models import _HMMGibbsSampling, _HMMMeanField
from rslds.transitions import InputHMMTransitions, InputOnlyHMMTransitions, \
    StickyInputOnlyHMMTransitions, SoftmaxInputHMMTransitions, SoftmaxInputOnlyHMMTransitions


##############################################################################
# Outline for extending an HMM to IO-HMM with mix-ins                        #
#    1. create a transitions base class that creates stacks of transition    #
#       matrices given a matrix of inputs (see transitions.py)               #
#    2. create a Gibbs mix-in that resamples transition parameters given     #
#       realized discrete state sequences                                    #
#                                                                            #
##############################################################################


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
        self.trans_distn.resample(
            stateseqs=[s.stateseq for s in self.states_list],
            covseqs=[s.covariates for s in self.states_list],
        )
        self._clear_caches()

    ### Mean Field
    def meanfield_update_trans_distn(self):
        # self.trans_distn.meanfieldupdate(
        #     [s.expected_transcounts for s in self.states_list])
        # self._clear_caches()
        # TODO: Update to take in sufficient statistics of x and z
        pass

class InputHMM(_InputHMMMixin, _HMMGibbsSampling, _HMMMeanField):
    _trans_class = InputHMMTransitions
    _states_class = InputHMMStates

class InputOnlyHMM(InputHMM):
    _trans_class = InputOnlyHMMTransitions

class StickyInputOnlyHMM(InputHMM):
    _trans_class = StickyInputOnlyHMMTransitions

class SoftmaxInputHMM(InputHMM):
    _trans_class = SoftmaxInputHMMTransitions

class SoftmaxInputOnlyHMM(InputHMM):
    _trans_class = SoftmaxInputOnlyHMMTransitions

### ARHMM's
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

class InputOnlyARHMM(InputARHMM):
    _trans_class = InputOnlyHMMTransitions

class StickyInputOnlyARHMM(InputARHMM):
    _trans_class = StickyInputOnlyHMMTransitions

class SoftmaxInputARHMM(InputARHMM):
    _trans_class = SoftmaxInputHMMTransitions

class SoftmaxInputOnlyARHMM(InputARHMM):
    _trans_class = SoftmaxInputOnlyHMMTransitions

### Recurrent ARHMM's
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


class RecurrentOnlyARHMM(RecurrentARHMM):
    _trans_class = InputOnlyHMMTransitions

class StickyRecurrentOnlyARHMM(RecurrentARHMM):
    _trans_class = StickyInputOnlyHMMTransitions

class SoftmaxRecurrentARHMM(RecurrentARHMM):
    _trans_class = SoftmaxInputHMMTransitions

class SoftmaxRecurrentOnlyARHMM(RecurrentARHMM):
    _trans_class = SoftmaxInputOnlyHMMTransitions
