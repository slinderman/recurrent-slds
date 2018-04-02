import numpy as np

from pyhsmm.models import _HMMGibbsSampling, _HMMEM, _HMMMeanField
from pyhsmm.internals.initial_state import UniformInitialState

from autoregressive.models import _ARMixin
from autoregressive.util import AR_striding

from pyslds.models import _SLDSGibbsMixin, _SLDSVBEMMixin, _SLDSMeanFieldMixin

from rslds.states import InputHMMStates, PGRecurrentSLDSStates, SoftmaxRecurrentSLDSStates
import rslds.transitions as transitions

### Input-driven HMMs
class _InputHMMMixin(object):

    # Subclasses must specify the type of transition model
    _trans_class = None

    # custom init method, just so we call custom input trans class stuff
    def __init__(self,
                 obs_distns,
                 D_in=0,
                 trans_distn=None, trans_params={},
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
                self.init_state_distn = UniformInitialState(model=self)
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


class PGInputHMM(_InputHMMMixin, _HMMGibbsSampling):
    _trans_class = transitions.InputHMMTransitions
    _states_class = InputHMMStates


class PGInputOnlyHMM(PGInputHMM):
    _trans_class = transitions.InputOnlyHMMTransitions


class PGStickyInputOnlyHMM(PGInputHMM):
    _trans_class = transitions.StickyInputOnlyHMMTransitions


class SoftmaxInputHMM(_InputHMMMixin, _HMMGibbsSampling, _HMMEM):
    _trans_class = transitions.SoftmaxInputHMMTransitions
    _states_class = InputHMMStates

    ## EM
    def _M_step_trans_distn(self):
        zs = [s.expected_states.argmax(1).astype(np.int32) for s in self.states_list]
        xs = [s.covariates for s in self.states_list]
        xs = [np.row_stack([x, np.zeros(x.shape[1])]) for x in xs]
        self.trans_distn.initialize_with_logistic_regression(zs, xs)


class SoftmaxInputOnlyHMM(SoftmaxInputHMM):
    _trans_class = transitions.SoftmaxInputOnlyHMMTransitions


### ARHMM's
class _InputARHMMMixin(_InputHMMMixin, _ARMixin):

    def add_data(self, data, covariates=None, strided=False, **kwargs):
        if covariates is None:
            covariates = np.zeros((data.shape[0], 0))

        strided_data = AR_striding(data,self.nlags) if not strided else data
        lagged_covariates = covariates[self.nlags:]
        assert strided_data.shape[0] == lagged_covariates.shape[0]

        # Pass to InputHMM
        super(_InputARHMMMixin, self).add_data(data=strided_data,
                                               covariates=lagged_covariates,
                                               **kwargs)


class PGInputARHMM(_InputARHMMMixin, _HMMGibbsSampling):
    _trans_class = transitions.InputHMMTransitions
    _states_class = InputHMMStates


class PGInputOnlyARHMM(PGInputARHMM):
    _trans_class = transitions.InputOnlyHMMTransitions


class PGStickyInputOnlyARHMM(PGInputARHMM):
    _trans_class = transitions.StickyInputOnlyHMMTransitions


class SoftmaxInputARHMM(_InputARHMMMixin, _HMMGibbsSampling, _HMMEM):
    _trans_class = transitions.SoftmaxInputHMMTransitions
    _states_class = InputHMMStates


class SoftmaxInputOnlyARHMM(SoftmaxInputARHMM):
    _trans_class = transitions.SoftmaxInputOnlyHMMTransitions


### Recurrent ARHMM's
class _RecurrentARHMMMixin(_InputARHMMMixin):
    """
    In the "recurrent" version, the data also serve as covariates.
    """
    def add_data(self, data, covariates=None, strided=False, **kwargs):
        # Remember that the covariates[t] drives the transition probabilities p(z[t] | ...)
        # under our convention for add_data.
        T = data.shape[0]
        if covariates is None:
            covariates = np.zeros((T, 0))
        else:
            assert covariates.shape[0] == T

        # Combine the lagged data and the given covariates
        covariates = np.column_stack((
            np.row_stack((np.zeros(self.D), data[:-1])),
            covariates))

        super(_RecurrentARHMMMixin, self).add_data(data, covariates=covariates, **kwargs)


    def generate(self, T=100, keep=True, init_data=None, covariates=None, with_noise=True):
        from pybasicbayes.util.stats import sample_discrete
        # Generate from the prior and raise exception if unstable
        K, n = self.num_states, self.D

        # Prepare the covariates
        if covariates is None:
            covariates = np.zeros((T, 0))
        else:
            assert covariates.shape[0] == T

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
            # Sample discrete state given previous continuous state and covariates
            cov_t = np.column_stack((data[t-1:t], covariates[t]))
            A = self.trans_distn.get_trans_matrices(cov_t)[0]
            dss[t] = sample_discrete(A[dss[t-1], :])

            # Sample continuous state given current discrete state
            if with_noise:
                data[t] = self.obs_distns[dss[t]].rvs(cov_t, return_xy=False)
            else:
                data[t] = self.obs_distns[dss[t]].predict(cov_t)

            assert np.all(np.isfinite(data[t])), "RARHMM appears to be unstable!"

        # TODO:
        # if keep:
        #     ...

        return data, dss


class PGRecurrentARHMM(_RecurrentARHMMMixin, _HMMGibbsSampling):
    _trans_class = transitions.InputHMMTransitions
    _states_class = InputHMMStates


class PGRecurrentOnlyARHMM(PGRecurrentARHMM):
    _trans_class = transitions.InputOnlyHMMTransitions


class PGStickyRecurrentOnlyARHMM(PGRecurrentARHMM):
    _trans_class = transitions.StickyInputOnlyHMMTransitions


class SoftmaxRecurrentARHMM(_RecurrentARHMMMixin, _HMMGibbsSampling, _HMMEM):
    _trans_class = transitions.SoftmaxInputHMMTransitions
    _states_class = InputHMMStates


class SoftmaxRecurrentOnlyARHMM(SoftmaxRecurrentARHMM):
    _trans_class = transitions.SoftmaxInputOnlyHMMTransitions


### Stick-breaking transition models with Pólya-gamma augmentation
class _RecurrentSLDSBase(object):
    def __init__(self, dynamics_distns, emission_distns, init_dynamics_distns,
                 fixed_emission=False, **kwargs):

        self.fixed_emission = fixed_emission

        # This class must always be used in conjunction with an SLDS class
        super(_RecurrentSLDSBase, self).__init__(
            dynamics_distns, emission_distns, init_dynamics_distns,
            D_in=dynamics_distns[0].D_out, **kwargs)

    def add_data(self, data, **kwargs):
        self.states_list.append(
                self._states_class(model=self, data=data, **kwargs))


class PGRecurrentSLDS(_RecurrentSLDSBase, _SLDSGibbsMixin, PGInputHMM):

    _states_class = PGRecurrentSLDSStates
    _trans_class = transitions.InputHMMTransitions

    def resample_trans_distn(self):
        # Include the auxiliary variables used for state resampling
        self.trans_distn.resample(
            stateseqs=[s.stateseq for s in self.states_list],
            covseqs=[s.covariates for s in self.states_list],
            omegas=[s.trans_omegas for s in self.states_list]
        )
        self._clear_caches()

    def resample_emission_distns(self):
        if self.fixed_emission:
            return
        super(PGRecurrentSLDS, self).resample_emission_distns()


class StickyPGRecurrentSLDS(PGRecurrentSLDS):
    _trans_class = transitions.StickyInputHMMTransitions


class PGRecurrentOnlySLDS(PGRecurrentSLDS):
    _trans_class = transitions.InputOnlyHMMTransitions


class StickyPGRecurrentOnlySLDS(PGRecurrentSLDS):
    _trans_class = transitions.StickyInputOnlyHMMTransitions


### Softmax transition models with variational inference
class SoftmaxRecurrentSLDS(_RecurrentSLDSBase, _SLDSMeanFieldMixin, _SLDSVBEMMixin, SoftmaxInputHMM):
    _states_class = SoftmaxRecurrentSLDSStates
    _trans_class = transitions.SoftmaxInputHMMTransitions

    def _M_step_trans_distn(self):
        stack_tuples = lambda lst: list(map(lambda xs: np.concatenate(xs, axis=0), zip(*lst)))
        self.trans_distn.max_likelihood(
            stats=stack_tuples([s.E_trans_stats for s in self.states_list]))

    def meanfield_update_trans_distn(self):
        # Include the auxiliary variables of the lower bound
        stack_tuples = lambda lst: list(map(lambda xs: np.concatenate(xs, axis=0), zip(*lst)))
        self.trans_distn.meanfieldupdate(
            stats=stack_tuples([s.E_trans_stats for s in self.states_list]))

    def _init_mf_from_gibbs(self):
        self.trans_distn._initialize_mean_field()
        super(SoftmaxRecurrentSLDS, self)._init_mf_from_gibbs()

    def initialize_transitions_from_gibbs(self):
        self.trans_distn.initialize_with_logistic_regression(
            [s.stateseq for s in self.states_list],
            [s.gaussian_states for s in self.states_list])

    def meanfield_update_parameters(self):
        self.meanfield_update_init_dynamics_distns()
        self.meanfield_update_dynamics_distns()
        self.meanfield_update_emission_distns()
        super(SoftmaxRecurrentSLDS, self).meanfield_update_parameters()


class SoftmaxRecurrentOnlySLDS(SoftmaxRecurrentSLDS):
    _trans_class = transitions.SoftmaxInputOnlyHMMTransitions
