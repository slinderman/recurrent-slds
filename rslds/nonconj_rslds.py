from rslds.rslds import RecurrentSLDS, RecurrentSLDSStates
from rslds.transitions import SoftmaxInputHMMTransitions, SoftmaxInputOnlyHMMTransitions

import autograd.numpy as anp
from autograd import grad

class NonconjugateRecurrentSLDSStates(RecurrentSLDSStates):
    """
    When the transition model is not amenable to PG augmentation, we
    need an alternative approach.  Many methods could apply:
    - HMC for x | z;  HMM message passing for z | x.
    - SMC for x and z
    - Assumed density filtering for x and z

    This class will do HMC.
    """
    @property
    def info_trans_params(self):
        """
        The non conjugate model doesn't have simple potentials
        """
        return anp.zeros((self.D_latent, self.D_latent)), \
               anp.zeros((self.D_latent,))

    def joint_log_probability(self, x):
        # A differentiable function to compute the joint probability for a given
        # latent state sequence
        ll = 0

        # Initial likelihood
        J_init, h_init, _ = self.info_init_params
        ll += -0.5 * anp.einsum('i,ij,j->', x[0], J_init, x[0])
        ll += anp.sum(x[0] * h_init)

        # Continuous transition potentials
        J_pair_11, J_pair_21, J_pair_22, h_pair_1, h_pair_2, _ = self.info_dynamics_params
        x_prev, x_next = x[:-1], x[1:]
        ll += -0.5 * anp.einsum('ti,tij,tj->', x_prev, J_pair_11[:-1], x_prev)
        ll += -1.0 * anp.einsum('ti,tij,tj->', x_next, J_pair_21[:-1], x_prev)
        ll += -0.5 * anp.einsum('ti,tij,tj->', x_next, J_pair_22[:-1], x_next)
        ll += anp.sum(x_prev * h_pair_1[:-1])
        ll += anp.sum(x_next * h_pair_2[:-1])

        # Discrete transition probabilities
        trans_distn = self.trans_distn
        ll += trans_distn.joint_log_probability(
            trans_distn.logpi, trans_distn.W, [self.stateseq], [x[:-1]])

        # Observation likelihoods
        J_node, h_node, _ = self.info_emission_params
        ll += -0.5 * anp.einsum('ti,tij,tj->', x, J_node, x)
        ll += anp.sum(x * h_node)

        return ll

    def resample_gaussian_states(self, step_sz=0.01, n_steps=10):
        # Run HMC
        from hips.inference.hmc import hmc
        nll = lambda x: \
            -1 * self.joint_log_probability(
                anp.reshape(x, (self.T, self.D_latent))) / self.T

        dnll = grad(nll)
        x0 = self.gaussian_states.ravel()
        xf = hmc(nll, dnll, step_sz=step_sz, n_steps=n_steps, q_curr=x0)
        self.gaussian_states = xf.reshape((self.T, self.D_latent))

    def resample_transition_auxiliary_variables(self):
        pass


class SoftmaxRecurrentSLDS(RecurrentSLDS):
    _states_class = NonconjugateRecurrentSLDSStates
    _trans_class = SoftmaxInputHMMTransitions

    def resample_trans_distn(self):
        # Include the auxiliary variables used for state resampling
        self.trans_distn.resample(
            stateseqs=[s.stateseq for s in self.states_list],
            covseqs=[s.covariates for s in self.states_list],
        )
        self._clear_caches()


class SoftmaxRecurrentOnlySLDS(SoftmaxRecurrentSLDS):
    _trans_class = SoftmaxInputOnlyHMMTransitions