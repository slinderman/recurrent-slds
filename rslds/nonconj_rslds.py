from rslds.rslds import RecurrentSLDS, RecurrentSLDSStates
from rslds.transitions import SoftmaxInputHMMTransitions, SoftmaxInputOnlyHMMTransitions
from rslds.util import logistic

import autograd.numpy as anp
from autograd import grad

class _NonconjugateRecurrentSLDSStatesGibbs(RecurrentSLDSStates):
    """
    When the transition model is not amenable to PG augmentation, we
    need an alternative approach.  Many methods could apply:
    - HMC for x | z;  HMM message passing for z | x.
    - SMC for x and z
    - Assumed density filtering for x and z

    This class will do HMC.
    """

    def __init__(self, model, **kwargs):
        super(_NonconjugateRecurrentSLDSStatesGibbs, self).__init__(model, **kwargs)

        # HMC params
        self.step_sz = 1e-5
        self.accept_rate = 0.90

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

    def resample_gaussian_states(self, step_sz=0.001, n_steps=10):
        # Run HMC
        from hips.inference.hmc import hmc
        hmc_objective = lambda x: \
            self.joint_log_probability(anp.reshape(x, (self.T, self.D_latent)))

        grad_hmc_objective = grad(hmc_objective)
        x0 = self.gaussian_states.ravel()
        xf, self.step_sz, self.accept_rate = \
            hmc(hmc_objective, grad_hmc_objective,
                step_sz=self.step_sz, n_steps=n_steps, q_curr=x0,
                negative_log_prob=False,
                adaptive_step_sz=True,
                avg_accept_rate=self.accept_rate,
                min_step_sz=1e-5)

        self.gaussian_states = xf.reshape((self.T, self.D_latent))
        self.covariates = self.gaussian_states[:-1].copy()

    def resample_transition_auxiliary_variables(self):
        pass


class _NonconjugateRecurrentSLDSStatesMeanField(RecurrentSLDSStates):
    """
    Implement meanfield variational inference with the JJ96 lower bound.
    """
    def __init__(self, model, **kwargs):
        super(_NonconjugateRecurrentSLDSStatesMeanField, self).__init__(model, **kwargs)
        self.a = anp.zeros((self.T-1,))
        self.bs = anp.ones((self.T-1, self.num_states))

    @property
    def lambda_bs(self):
        return 0.5 / self.bs * (logistic(self.bs) - 0.5)

    ### Updates for q(x)
    @property
    def expected_info_rec_params(self):
        """
        Compute J_rec and h_rec
        """
        E_z = self.expected_states
        E_W = self.trans_distn.expected_W
        E_logpi = self.trans_distn.expected_logpi

        # Eq (24) 2 * E[ W diag(lambda(b_t)) W^\trans ]
        J_rec = anp.zeros((self.T, self.D_latent, self.D_latent))
        anp.einsum('ik, tk, kj -> tij', 2 * E_W, self.lambda_bs, E_W.T, out=J_rec[:-1])

        # Eq (25)
        h_rec = anp.zeros((self.T, self.D_latent))
        h_rec[:-1] += E_z[1:].dot(E_W.T)
        h_rec[:-1] += -1 * (0.5 - 2 * self.a[:,None] * self.lambda_bs).dot(E_W.T)
        h_rec[:-1] += -2 * (E_z[:-1].dot(E_logpi) * self.lambda_bs).dot(E_W.T)
        h_rec[:-1] += -(0.5 - 2 * self.a[:,None] * self.lambda_bs).dot(E_W.T)

        return J_rec, h_rec

    @property
    def expected_info_emission_params(self):
        """
        Fold in the recurrent potentials
        """
        J_node, h_node, log_Z_node = \
            super(_NonconjugateRecurrentSLDSStatesMeanField, self).\
                expected_info_emission_params

        J_rec, h_rec = self.expected_info_rec_params
        return J_node + J_rec, h_node + h_rec, log_Z_node

    ### Updates for q(z)
    @property
    def mf_trans_matrix(self):
        """
        Eq (31).  need exp { E[ \log \pi(\theta) ]}
        :return:
        """
        return self.trans_distn.exp_expected_logpi

    @property
    def mf_aBl(self):
        aBl = super(_NonconjugateRecurrentSLDSStatesMeanField, self).mf_aBl

        # Add in node potentials from transitions
        aBl += self._mf_aBl_rec
        return aBl

    @property
    def _mf_aBl_rec(self):
        # Compute the extra node *log* potentials from the transition model
        aBl = anp.zeros((self.T, self.num_states))

        # Eq (34): \psi_{t+1}^{rec} = E [ x_t^\trans W(\theta) ]
        E_x = self.smoothed_mus
        E_W = self.trans_distn.expected_W
        aBl[1:] += E_x[:-1].dot(E_W)

        # Eq (36) transpose:
        #   -2 E[ x_t W \diag(\lambda(b_t) \log \pi^\trans ]
        E_logpi = self.trans_distn.expected_logpi
        a, bs = self.a, self.bs
        aBl[:-1] += -2 * (E_x[:-1].dot(E_W) * self.lambda_bs).dot(E_logpi.T)

        # Eq (37) transpose:
        #   (1/2 - 2 a_t \lambda(b_t)^\trans E[ \log \pi^\trans]
        aBl[:-1] += -1 * (0.5 - 2*a[:,None] * self.lambda_bs).dot(E_logpi.T)

        # Eq (38)
        import warnings
        warnings.warn("Eq (38) is not implemented correctly.  "
                      "Needs E[log pi log pi^T] and traces.  "
                      "It will still work with point estimates of log pi")

        # TODO: Double check the last transpose!
        aBl[:-1] += -1 * anp.einsum('ik, tk, ki -> ti', E_logpi, self.lambda_bs, E_logpi.T)

        return aBl

    ### Updates for auxiliary variables of JJ96 bound (a and bs)
    def meanfield_update_auxiliary_vars(self, n_iter=10):
        """
        Update a and bs via block coordinate updates
        """
        K = self.num_states
        E_z = self.expected_states
        E_x = self.smoothed_mus
        E_xxT = self.smoothed_sigmas + E_x[:,:,None] * E_x[:,None,:]
        E_logpi = self.trans_distn.expected_logpi
        E_W = self.trans_distn.expected_W

        # Compute m_{tk} = E[v_{tk}]
        m = E_z[:-1].dot(E_logpi) + E_x[:-1].dot(E_W)

        # Compute s_{tk} = E[v_{tk}^2]
        import warnings
        warnings.warn("Eq (43) is not implemented correctly.  "
                      "Needs second moments of W.  "
                      "It will still work with point estimates W")
        E_logpi_sq = E_logpi ** 2
        E_WWT = anp.array([anp.outer(E_W[:,k], E_W[:,k]) for k in range(K)])

        # E[v_{tk}^2] = e_k^T E[\psi_1 + \psi_2 + \psi_3] e_k  where
        # e_k^T \psi_1 e_k =
        #        = Tr(E[z_t z_t^T p_k p_k^T])               with p_k = P[:,k]  (kth col of trans matrix)
        #        = Tr(diag(E[z_t]) \dot E[p_k p_k^T] )
        #        = Tr(A^T \dot B)                           with A = A^T = diag(E[z_t]), B = E[p_k p_k^T]
        #        = \sum_{ij} A_{ij} * B_{ij}
        #        = \sum_{i} E[z_{t,i}] * E[p_{ik}^2]
        psi_1 = E_z[:-1].dot(E_logpi_sq)

        # e_k^T \psi_2 e_k =
        #            = 2e_k^T E[W^T x_t z_t^T log pi] e_k
        # \psi_2     = 2 diag*(E[W^T x_t z_t^T log pi])
        #            = 2 E[(x_t^T W) * (z_t^T log pi)]
        psi_2 = 2 * E_x[:-1].dot(E_W) * E_z[:-1].dot(E_logpi)

        # e_k^T \psi_3 e_k =
        #        =Tr(E[x_t x_t^T w_k w_k^T])               with w_k = W[:,k]  (kth col of weight matrix)
        #        = Tr(E[x_t x_t^T] \dot E[w_k w_k^T])
        #        = Tr(A^T \dot B)                           with A = A^T = E[x_t x_t^T]), B = E[w_k w_k^T]
        #        = \sum_{ij} A_{ij} * B_{ij}
        psi_3 = anp.einsum('tij, kij -> tk', E_xxT[:-1], E_WWT)

        # s_{tk} = E[v_{tk}^2]
        s = psi_1 + psi_2 + psi_3
        assert anp.all(s > 0)

        for itr in range(n_iter):
            lambda_bs = self.lambda_bs

            # Eq (42)
            self.a = 2 * (m * lambda_bs).sum(axis=1) + K / 2.0 - 1.0
            self.a /= 2 * lambda_bs.sum(axis=1)

            # Eq (43)
            self.bs = anp.sqrt(s - 2 * m * self.a[:,None] + self.a[:,None]**2)

    def meanfieldupdate(self, niter=1):
        niter = self.niter if hasattr(self, 'niter') else niter
        for itr in range(niter):
            self.meanfield_update_discrete_states()
            self.meanfield_update_gaussian_states()
            self.meanfield_update_auxiliary_vars()

        self._mf_aBl = None


class NonconjugateRecurrentSLDSStates(_NonconjugateRecurrentSLDSStatesGibbs,
                                      _NonconjugateRecurrentSLDSStatesMeanField):
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