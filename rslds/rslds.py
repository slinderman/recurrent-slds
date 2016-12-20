import numpy as np

from pyslds.states import _SLDSStatesCountData, _SLDSStatesMaskedData
from pyslds.models import _SLDSGibbsMixin

import pypolyagamma as ppg

from rslds.inhmm import InputHMMTransitions, InputHMMStates, \
    InputHMM, StickyInputHMMTransitions, InputOnlyHMMTransitions, \
    StickyInputOnlyHMMTransitions
from rslds.util import one_hot

class InputSLDSStates(_SLDSStatesCountData, _SLDSStatesMaskedData, InputHMMStates):
    """
    Include the multinomial emissions from the discrete states
    as observations in the model.
    """
    def __init__(self, model, covariates=None, data=None, mask=None, fixed_stateseq=None, **kwargs):
        # By definition, the covariates are the latent gaussian states
        if covariates is not None:
            raise NotImplementedError("Not supporting exogenous inputs yet")

        super(InputSLDSStates, self).\
            __init__(model, data=data, mask=mask,
                     fixed_stateseq=fixed_stateseq, **kwargs)

        # Override the Gaussian states with random noise
        self.gaussian_states = np.random.randn(self.T, self.D_latent)

        # Set the covariates to be the gaussian states
        self.covariates = self.gaussian_states[:-1].copy()
        # warn("__init__: zero out covariates")
        # self.covariates *= 0

        # Initialize auxiliary variables for transitions
        self.trans_omegas = np.ones((self.T-1, self.num_states-1))

    @property
    def trans_distn(self):
        return self.model.trans_distn

    def generate_states(self, with_noise=True):
        """
        Jointly sample the discrete and continuous states
        """
        from pybasicbayes.util.stats import sample_discrete
        # Generate from the prior and raise exception if unstable
        T, K, n = self.T, self.num_states, self.D_latent

        # Initialize discrete state sequence
        init_state_distn = np.ones(self.num_states) / float(self.num_states)
        dss = np.empty(T, dtype=np.int32)
        dss[0] = sample_discrete(init_state_distn.ravel())

        gss = np.empty((T,n), dtype='double')
        gss[0] = self.init_dynamics_distns[dss[0]].rvs()

        for t in range(1,T):
            # Sample discrete state given previous continuous state
            A = self.trans_distn.get_trans_matrices(gss[t-1:t])[0]
            if with_noise:
                # Sample discrete state from recurrent transition matrix
                dss[t] = sample_discrete(A[dss[t-1], :])
                # Sample continuous state given current discrete state
                gss[t] = self.dynamics_distns[dss[t]].\
                    rvs(x=np.hstack((gss[t-1][None,:], self.inputs[t-1][None,:])),
                        return_xy=False)
            else:
                # Pick the most likely next discrete state and continuous state
                dss[t] = np.argmax(A[dss[t-1], :])
                gss[t] = self.dynamics_distns[dss[t]]. \
                    predict(np.hstack((gss[t-1][None,:], self.inputs[t-1][None,:])))
            assert np.all(np.isfinite(gss[t])), "SLDS appears to be unstable!"

        self.stateseq = dss
        self.gaussian_states = gss

    @property
    def info_emission_params(self):
        J_node, h_node, log_Z_node = super(InputSLDSStates, self).info_emission_params
        # warn("info emission params: no inputs")
        J_node_trans, h_node_trans = self.info_trans_params
        J_node[:-1] += J_node_trans
        h_node[:-1] += h_node_trans
        return J_node, h_node, log_Z_node

    @property
    def info_trans_params(self):
        # Add the potential from the transitions
        trans_distn, omega = self.trans_distn, self.trans_omegas

        prev_state = one_hot(self.stateseq[:-1], self.num_states)
        next_state = one_hot(self.stateseq[1:], self.num_states)

        A = trans_distn.A[:, :self.num_states]
        C = trans_distn.A[:, self.num_states:self.num_states+self.D_latent]
        # D = trans_distn.A[:, self.num_states+self.D_latent:]
        b = trans_distn.b

        CCT = np.array([np.outer(cp, cp) for cp in C]). \
            reshape((trans_distn.D_out, self.D_latent ** 2))
        J_node = np.dot(omega, CCT)

        kappa = trans_distn.kappa_func(next_state[:,:-1])
        h_node = kappa.dot(C)
        h_node -= (omega * b.T).dot(C)
        # warn("info_emission_params: Testing state dependence only")
        h_node -= (omega * prev_state.dot(A.T)).dot(C)
        # h_node[:-1] -= (omega * self.inputs.dot(D.T)).dot(C)

        # Restore J_node to its original shape
        J_node = J_node.reshape((self.T-1, self.D_latent, self.D_latent))
        return J_node, h_node

    def resample_gaussian_states(self):
        super(InputSLDSStates, self).resample_gaussian_states()
        self.covariates = self.gaussian_states[:-1].copy()
        # warn("resample_gaussian_states: zero out covariates")
        # self.covariates *= 0

    def resample_auxiliary_variables(self):
        super(InputSLDSStates, self).resample_auxiliary_variables()
        self.resample_transition_auxiliary_variables()

    def resample_transition_auxiliary_variables(self):
        # Resample the auxiliary variable for the transition matrix
        trans_distn = self.trans_distn
        prev_state = one_hot(self.stateseq[:-1], self.num_states)
        next_state = one_hot(self.stateseq[1:], self.num_states)

        A = trans_distn.A[:, :self.num_states]
        C = trans_distn.A[:, self.num_states:self.num_states + self.D_latent]
        # D = trans_distn.A[:, self.num_states+self.D_latent:]
        b = trans_distn.b

        # warn("resample_transition_aux_variables: Testing state dependence only")
        psi = prev_state.dot(A.T) \
              + self.covariates.dot(C.T) \
              + b.T \
              # + self.inputs.dot(D.T) \

        b_pg = trans_distn.b_func(next_state[:,:-1])
        omega0 = self.trans_omegas.copy()
        ppg.pgdrawvpar(self.ppgs, b_pg.ravel(), psi.ravel(), self.trans_omegas.ravel())
        assert not np.allclose(omega0, self.trans_omegas)


class InputSLDS(_SLDSGibbsMixin, InputHMM):

    _states_class = InputSLDSStates
    _trans_class = InputHMMTransitions

    def __init__(self, dynamics_distns, emission_distns, init_dynamics_distns,
                 fixed_emission=False, **kwargs):

        # self.init_dynamics_distns = init_dynamics_distns
        # self.dynamics_distns = dynamics_distns
        #
        # # Allow for a single, shared emission distribution
        # if not isinstance(emission_distns, list):
        #     self._single_emission = True
        #     self._emission_distn = emission_distns
        #     self.emission_distns = [emission_distns] * len(self.dynamics_distns)
        # else:
        #     assert isinstance(emission_distns, list) and \
        #            len(emission_distns) == len(dynamics_distns)
        #     self._single_emission = False
        #     self.emission_distns = emission_distns
        #
        # super(_SLDSMixin, self).__init__(
        #     obs_distns=self.dynamics_distns, **kwargs)
        #
        #
        # self.init_dynamics_distns = init_dynamics_distns
        # self.dynamics_distns = dynamics_distns
        # self.emission_distns = emission_distns
        self.fixed_emission = fixed_emission

        # TODO: Will this just work?
        kwargs["D_in"] = dynamics_distns[0].D_out

        super(InputSLDS, self).__init__(
            dynamics_distns, emission_distns, init_dynamics_distns, **kwargs)

    def resample_trans_distn(self):
        # Include the auxiliary variables used for state resampling
        self.trans_distn.resample(
            stateseqs=[s.stateseq for s in self.states_list],
            covseqs=[s.covariates for s in self.states_list],
            omegas=[s.trans_omegas for s in self.states_list]
        )
        self._clear_caches()

    def add_data(self, data, covariates=None, **kwargs):
        self.states_list.append(
                self._states_class(
                    model=self, data=data,
                    **kwargs))

    def generate(self, T=100, keep=True, with_noise=True, **kwargs):
        s = self._states_class(model=self, T=T, initialize_from_prior=True, **kwargs)
        s.generate_states(with_noise=with_noise)
        data = self._generate_obs(s, with_noise=with_noise)
        if keep:
            self.states_list.append(s)
        return data, s.stateseq

    def _generate_obs(self, s, with_noise=True):
        if s.data is None:
            s.data = s.generate_obs()
        else:
            # TODO: Handle missing data
            raise NotImplementedError

        return s.data, s.gaussian_states

    def resample_emission_distns(self):
        if self.fixed_emission:
            return
        super(InputSLDS, self).resample_emission_distns()

class StickyInputSLDS(InputSLDS):
    _trans_class = StickyInputHMMTransitions


class InputOnlySLDS(InputSLDS):
    _trans_class = InputOnlyHMMTransitions


class StickyInputOnlySLDS(InputSLDS):
    _trans_class = StickyInputOnlyHMMTransitions
