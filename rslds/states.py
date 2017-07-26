import numpy as np
from scipy.misc import logsumexp

from pybasicbayes.util.stats import sample_discrete
from pyhsmm.internals.hmm_states import HMMStatesEigen

from pyslds.states import _SLDSStatesCountData, _SLDSStatesMaskedData, _SLDSStatesVBEM

from rslds.util import one_hot, logistic, inhmm_entropy

### Recurrent models build on an input-driven HMM

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


###
# The recurrent SLDS is basically a combo of the Input HMM and SLDS
# However, we need a few enhancements. The latent states need to how to
# update the continuous values given the discrete states.
#
class _RecurrentSLDSStatesBase(object):
    """
    Effectively, the multinomial emissions from the discrete states
    as observations for the continuous states.
    """
    def __init__(self, model, covariates=None, data=None, **kwargs):

        # By definition, the covariates are the latent gaussian states
        if covariates is not None:
            raise NotImplementedError("Not supporting exogenous inputs yet")

        super(_RecurrentSLDSStatesBase, self).\
            __init__(model, data=data, **kwargs)

        # Set the covariates to be the gaussian states
        self.covariates = self.gaussian_states[:-1]

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


class PGRecurrentSLDSStates(_RecurrentSLDSStatesBase,
                            _SLDSStatesCountData,
                            InputHMMStates):
    """
    Use Pólya-gamma augmentation to perform Gibbs sampling with conjugate updates.
    """
    def __init__(self, model, covariates=None, data=None, mask=None, **kwargs):

        super(PGRecurrentSLDSStates, self).\
            __init__(model, covariates=covariates, data=data, mask=mask, **kwargs)

        # Initialize auxiliary variables for transitions
        self.trans_omegas = np.ones((self.T-1, self.num_states-1))

    @property
    def info_emission_params(self):
        J_node, h_node, log_Z_node = super(PGRecurrentSLDSStates, self).info_emission_params
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
        h_node -= (omega * prev_state.dot(A.T)).dot(C)
        # h_node[:-1] -= (omega * self.inputs.dot(D.T)).dot(C)

        # Restore J_node to its original shape
        J_node = J_node.reshape((self.T-1, self.D_latent, self.D_latent))
        return J_node, h_node

    def resample_gaussian_states(self):
        super(PGRecurrentSLDSStates, self).resample_gaussian_states()
        self.covariates = self.gaussian_states[:-1].copy()

    def resample_auxiliary_variables(self):
        super(PGRecurrentSLDSStates, self).resample_auxiliary_variables()
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

        psi = prev_state.dot(A.T) \
              + self.covariates.dot(C.T) \
              + b.T \
              # + self.inputs.dot(D.T) \

        b_pg = trans_distn.b_func(next_state[:,:-1])
        omega0 = self.trans_omegas.copy()

        import pypolyagamma as ppg
        ppg.pgdrawvpar(self.ppgs, b_pg.ravel(), psi.ravel(), self.trans_omegas.ravel())
        assert not np.allclose(omega0, self.trans_omegas)


##
# Recurrent SLDS with softmax transition model.
# Use a variational lower bound suggested by Knowles and Minka, 2009
# in order to perform conjugate updates of q(z) and q(x)
##
class _SoftmaxRecurrentSLDSStatesBase(_RecurrentSLDSStatesBase,
                                      _SLDSStatesMaskedData,
                                      InputHMMStates):
    """
    Implement variational EM with the JJ96 lower bound to update q(z) and q(x)
    """
    def __init__(self, model, **kwargs):
        super(_SoftmaxRecurrentSLDSStatesBase, self).__init__(model, **kwargs)
        self.a = np.zeros((self.T - 1,))
        self.bs = np.ones((self.T - 1, self.num_states))

    @property
    def lambda_bs(self):
        return 0.5 / self.bs * (logistic(self.bs) - 0.5)

    def _set_expected_trans_stats(self):
        """
        Compute the expected stats for updating the transition distn
        stats = E_u_zp1T, E_uuT, E_u, a, lambda_bs
        """
        T, D, K = self.T, self.D_latent, self.num_states

        E_z = self.expected_states
        E_z_zp1T = self.expected_joints
        E_x = self.smoothed_mus
        E_x_xT = self.smoothed_sigmas + E_x[:, :, None] * E_x[:, None, :]

        # Combine to get trans stats
        # E_u = [E[z], E[x]]
        E_u = np.concatenate((E_z[:-1], E_x[:-1]), axis=1)

        # E_u_zp1T = [ E[z zp1^T],  E[x, zp1^T] ]
        E_x_zp1T = E_x[:-1, :, None] * E_z[1:, None, :]
        E_u_zp1T = np.concatenate((E_z_zp1T, E_x_zp1T), axis=1)

        # E_uuT = [[ diag(E[z]),  E[z]E[x^T] ]
        #          [ E[x]E[z^T],  E[xxT]     ]]
        E_u_uT = np.zeros((T - 1, K + D, K + D))
        E_u_uT[:, np.arange(K), np.arange(K)] = E_z[:-1]
        E_u_uT[:, :K, K:] = E_z[:-1, :, None] * E_x[:-1, None, :]
        E_u_uT[:, K:, :K] = E_x[:-1, :, None] * E_z[:-1, None, :]
        E_u_uT[:, K:, K:] = E_x_xT[:-1]

        self.E_trans_stats = (E_u_zp1T, E_u_uT, E_u, self.a, self.lambda_bs)


class _SoftmaxRecurrentSLDSStatesMeanField(_SoftmaxRecurrentSLDSStatesBase):

    @property
    def expected_info_rec_params(self):
        """
        Compute J_rec and h_rec
        """
        E_z = self.expected_states
        E_W = self.trans_distn.expected_W
        E_WWT = self.trans_distn.expected_WWT
        E_logpi_WT = self.trans_distn.expected_logpi_WT

        # Eq (24) 2 * E[ W diag(lambda(b_t)) W^\trans ]
        J_rec = np.zeros((self.T, self.D_latent, self.D_latent))
        np.einsum('tk, kij -> tij', 2 * self.lambda_bs, E_WWT, out=J_rec[:-1])

        # Eq (25)
        h_rec = np.zeros((self.T, self.D_latent))
        h_rec[:-1] += E_z[1:].dot(E_W.T)
        h_rec[:-1] += -1 * (0.5 - 2 * self.a[:,None] * self.lambda_bs).dot(E_W.T)
        h_rec[:-1] += -2 * np.einsum('ti, tj, jid -> td', E_z[:-1], self.lambda_bs, E_logpi_WT)

        return J_rec, h_rec

    @property
    def expected_info_emission_params(self):
        """
        Fold in the recurrent potentials
        """
        J_node, h_node, log_Z_node = \
            super(_SoftmaxRecurrentSLDSStatesMeanField, self).\
                expected_info_emission_params

        J_rec, h_rec = self.expected_info_rec_params
        return J_node + J_rec, h_node + h_rec, log_Z_node

    ### Updates for q(z)
    # @property
    # def mf_trans_matrix(self):
    #     """
    #     Eq (31).  need exp { E[ \log \pi(\theta) ]}
    #     :return:
    #     """
    #     return self.trans_distn.exp_expected_logpi

    @property
    def mf_aBl(self):
        aBl = super(_SoftmaxRecurrentSLDSStatesMeanField, self).mf_aBl

        # Add in node potentials from transitions
        aBl += self._mf_aBl_rec
        return aBl

    @property
    def _mf_aBl_rec(self):
        # Compute the extra node *log* potentials from the transition model
        aBl = np.zeros((self.T, self.num_states))

        # Eq (34): \psi_{t+1}^{rec} = E [ x_t^\trans W(\theta) ]
        E_x = self.smoothed_mus
        E_W = self.trans_distn.expected_W
        E_logpi = self.trans_distn.expected_logpi
        E_logpi_WT = self.trans_distn.expected_logpi_WT
        E_logpi_logpiT = self.trans_distn.expected_logpi_logpiT
        E_logpisq = np.array([np.diag(Pk) for Pk in E_logpi_logpiT]).T

        aBl[1:] += E_x[:-1].dot(E_W)

        # Eq (36) transpose:
        #   -2 E[ x_t W \diag(\lambda(b_t) \log \pi^\trans ]
        aBl[:-1] += -2 * np.einsum('td, kid, tk -> ti', E_x[:-1], E_logpi_WT, self.lambda_bs)

        # Eq (37) transpose:
        #   (1/2 - 2 a_t \lambda(b_t)^\trans E[ \log \pi^\trans]
        a, bs = self.a, self.bs
        aBl[:-1] += -1 * (0.5 - 2*a[:,None] * self.lambda_bs).dot(E_logpi.T)

        # Eq (38)
        aBl[:-1] += -1 * self.lambda_bs.dot(E_logpisq.T)

        return aBl

    ### Updates for auxiliary variables of JJ96 bound (a and bs)
    def meanfield_update_auxiliary_vars(self, n_iter=10):
        """
        Update a and bs via block coordinate updates
        """
        K = self.num_states
        E_z = self.expected_states
        E_z /= E_z.sum(1, keepdims=True)
        E_x = self.smoothed_mus
        E_xxT = self.smoothed_sigmas + E_x[:,:,None] * E_x[:,None,:]
        E_logpi = self.trans_distn.expected_logpi
        E_W = self.trans_distn.expected_W
        E_WWT = self.trans_distn.expected_WWT
        E_logpi_WT = self.trans_distn.expected_logpi_WT
        E_logpi_logpiT = self.trans_distn.expected_logpi_logpiT
        E_logpi_sq = np.array([np.diag(Pk) for Pk in E_logpi_logpiT]).T

        # Compute m_{tk} = E[v_{tk}]
        m = E_z[:-1].dot(E_logpi) + E_x[:-1].dot(E_W)

        # Compute s_{tk} = E[v_{tk}^2]
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
        # psi_2 = 2 * E_x[:-1].dot(E_W) * E_z[:-1].dot(E_logpi)
        psi_2 = 2 * np.einsum('td, ti, kid -> tk', E_x[:-1], E_z[:-1], E_logpi_WT)

        # e_k^T \psi_3 e_k =
        #        =Tr(E[x_t x_t^T w_k w_k^T])               with w_k = W[:,k]  (kth col of weight matrix)
        #        = Tr(E[x_t x_t^T] \dot E[w_k w_k^T])
        #        = Tr(A^T \dot B)                           with A = A^T = E[x_t x_t^T]), B = E[w_k w_k^T]
        #        = \sum_{ij} A_{ij} * B_{ij}
        psi_3 = np.einsum('tij, kij -> tk', E_xxT[:-1], E_WWT)

        # s_{tk} = E[v_{tk}^2]
        s = psi_1 + psi_2 + psi_3
        assert np.all(s > 0)
        for itr in range(n_iter):
            lambda_bs = self.lambda_bs

            # Eq (42)
            self.a = 2 * (m * lambda_bs).sum(axis=1) + K / 2.0 - 1.0
            self.a /= 2 * lambda_bs.sum(axis=1)

            # Eq (43)
            self.bs = np.sqrt(s - 2 * m * self.a[:, None] + self.a[:, None] ** 2)

    def meanfield_update_discrete_states(self):
        """
        Override the discrete state updates in pyhsmm to keep the necessary suff stats.
        """
        self.clear_caches()

        # Run the message passing algorithm
        trans_potential = self.trans_distn.exp_expected_logpi
        init_potential = self.mf_pi_0
        likelihood_potential = self.mf_aBl
        alphal = self._messages_forwards_log(trans_potential, init_potential, likelihood_potential)
        betal = self._messages_backwards_log(trans_potential, likelihood_potential)

        # Convert messages into expectations
        expected_states = alphal + betal
        expected_states -= expected_states.max(1)[:, None]
        np.exp(expected_states, out=expected_states)
        expected_states /= expected_states.sum(1)[:, None]

        Al = np.log(trans_potential)
        log_joints = alphal[:-1, :, None] + betal[1:, None, :] \
                     + likelihood_potential[1:, None, :] \
                     + Al[None, ...]
        log_joints -= log_joints.max(axis=(1, 2), keepdims=True)
        joints = np.exp(log_joints)
        joints /= joints.sum(axis=(1, 2), keepdims=True)

        # Compute the log normalizer log p(x_{1:T} | \theta, a, b)
        normalizer = logsumexp(alphal[0] + betal[0])

        # Save expected statistics
        self.expected_states = expected_states
        self.expected_joints = joints
        self.expected_transcounts = joints.sum(0)
        self._normalizer = normalizer

        # Update the "stateseq" variable too
        self.stateseq = self.expected_states.argmax(1).astype('int32')

        # And then there's this snapshot thing... yikes mattjj!
        self._mf_param_snapshot = \
            (np.log(trans_potential), np.log(init_potential),
             likelihood_potential, normalizer)

        # Compute the variational entropy
        from pyslds.util import hmm_entropy
        params = (np.log(trans_potential), np.log(init_potential), likelihood_potential, normalizer)
        stats = (expected_states, self.expected_transcounts, normalizer)
        return hmm_entropy(params, stats)

    def meanfieldupdate(self, niter=1):
        super(_SoftmaxRecurrentSLDSStatesMeanField, self).meanfieldupdate()
        self.meanfield_update_auxiliary_vars()
        self._set_expected_trans_stats()

    def get_vlb(self, most_recently_updated=False):
        # E_{q(z)}[log p(z)]
        from pyslds.util import expected_hmm_logprob

        vlb = expected_hmm_logprob(
            self.mf_pi_0, self.trans_distn.exp_expected_logpi,
            (self.expected_states, self.expected_transcounts, self._normalizer))

        # E_{q(x)}[log p(y, x | z)]  is given by aBl
        # To get E_{q(x)}[ aBl ] we multiply and sum
        vlb += np.sum(self.expected_states * self.mf_aBl)

        # Add the variational entropy
        vlb += self._variational_entropy

        # test: compare to old code
        # vlb2 = super(_SLDSStatesMeanField, self).get_vlb(
        #         most_recently_updated=False) \
        #        + self._lds_normalizer
        # print(vlb - vlb2)
        return vlb

    def _init_mf_from_gibbs(self):
        super(_SoftmaxRecurrentSLDSStatesBase, self)._init_mf_from_gibbs()
        self.meanfield_update_auxiliary_vars()
        self.expected_joints = self.expected_states[:-1, :, None] * self.expected_states[1:, None, :]
        self._mf_param_snapshot = \
            (self.trans_distn.expected_logpi, np.log(self.mf_pi_0),
             self.mf_aBl, self._normalizer)
        self._set_expected_trans_stats()


class _SoftmaxRecurrentSLDSStatesVBEM(_SoftmaxRecurrentSLDSStatesBase):
    def vb_E_step(self):
        H_z = self.vb_E_step_discrete_states()
        H_x = self.vb_E_step_gaussian_states()
        self.vbem_update_auxiliary_vars()
        self._set_expected_trans_stats()
        self._variational_entropy = H_z + H_x

    ### Updates for q(x)
    @property
    def vbem_info_rec_params(self):
        """
        Compute J_rec and h_rec
        """
        E_z = self.expected_states
        W = self.trans_distn.W
        WWT = np.array([np.outer(wk, wk) for wk in W.T])
        logpi = self.trans_distn.logpi
        logpi_WT = np.array([np.outer(lpk, wk) for lpk, wk in zip(logpi.T, W.T)])

        # Eq (24) 2 * E[ W diag(lambda(b_t)) W^\trans ]
        J_rec = np.zeros((self.T, self.D_latent, self.D_latent))
        np.einsum('tk, kij -> tij', 2 * self.lambda_bs, WWT, out=J_rec[:-1])

        # Eq (25)
        h_rec = np.zeros((self.T, self.D_latent))
        h_rec[:-1] += E_z[1:].dot(W.T)
        h_rec[:-1] += -1 * (0.5 - 2 * self.a[:,None] * self.lambda_bs).dot(W.T)
        h_rec[:-1] += -2 * np.einsum('ti, tj, jid -> td', E_z[:-1], self.lambda_bs, logpi_WT)

        return J_rec, h_rec

    @property
    def vbem_info_emission_params(self):
        """
        Fold in the recurrent potentials
        """
        J_node, h_node, log_Z_node = \
            super(_SoftmaxRecurrentSLDSStatesVBEM, self). \
                vbem_info_emission_params

        J_rec, h_rec = self.vbem_info_rec_params
        return J_node + J_rec, h_node + h_rec, log_Z_node

    @property
    def vbem_aBl(self):
        aBl = super(_SoftmaxRecurrentSLDSStatesVBEM, self).vbem_aBl

        # Add in node potentials from transitions
        aBl += self._vbem_aBl_rec
        return aBl

    @property
    def _vbem_aBl_rec(self):
        # Compute the extra node *log* potentials from the transition model
        aBl = np.zeros((self.T, self.num_states))

        # Eq (34): \psi_{t+1}^{rec} = E [ x_t^\trans W(\theta) ]
        E_x = self.smoothed_mus
        W = self.trans_distn.W
        logpi = self.trans_distn.logpi
        logpi_WT = np.array([np.outer(lpk, wk) for lpk, wk in zip(logpi.T, W.T)])
        logpisq = logpi**2

        aBl[1:] += E_x[:-1].dot(W)

        # Eq (36) transpose:
        #   -2 E[ x_t W \diag(\lambda(b_t) \log \pi^\trans ]
        aBl[:-1] += -2 * np.einsum('td, kid, tk -> ti', E_x[:-1], logpi_WT, self.lambda_bs)

        # Eq (37) transpose:
        #   (1/2 - 2 a_t \lambda(b_t)^\trans E[ \log \pi^\trans]
        a, bs = self.a, self.bs
        aBl[:-1] += -1 * (0.5 - 2*a[:,None] * self.lambda_bs).dot(logpi.T)

        # Eq (38)
        aBl[:-1] += -1 * self.lambda_bs.dot(logpisq.T)

        return aBl

    ### Updates for auxiliary variables of JJ96 bound (a and bs)
    def vbem_update_auxiliary_vars(self, n_iter=10):
        """
        Update a and bs via block coordinate updates
        """
        K = self.num_states
        E_z = self.expected_states
        E_z /= E_z.sum(1, keepdims=True)
        E_x = self.smoothed_mus
        E_xxT = self.smoothed_sigmas + E_x[:,:,None] * E_x[:,None,:]
        logpi = self.trans_distn.logpi
        W = self.trans_distn.W
        WWT = np.array([np.outer(wk, wk) for wk in W.T])
        logpi_WT = np.array([np.outer(lpk, wk) for lpk, wk in zip(logpi.T, W.T)])
        logpi_sq = logpi**2

        # Compute m_{tk} = E[v_{tk}]
        m = E_z[:-1].dot(logpi) + E_x[:-1].dot(W)

        # Compute s_{tk} = E[v_{tk}^2]
        # E[v_{tk}^2] = e_k^T E[\psi_1 + \psi_2 + \psi_3] e_k  where
        # e_k^T \psi_1 e_k =
        #        = Tr(E[z_t z_t^T p_k p_k^T])               with p_k = P[:,k]  (kth col of trans matrix)
        #        = Tr(diag(E[z_t]) \dot E[p_k p_k^T] )
        #        = Tr(A^T \dot B)                           with A = A^T = diag(E[z_t]), B = E[p_k p_k^T]
        #        = \sum_{ij} A_{ij} * B_{ij}
        #        = \sum_{i} E[z_{t,i}] * E[p_{ik}^2]
        psi_1 = E_z[:-1].dot(logpi_sq)

        # e_k^T \psi_2 e_k =
        #            = 2e_k^T E[W^T x_t z_t^T log pi] e_k
        # \psi_2     = 2 diag*(E[W^T x_t z_t^T log pi])
        #            = 2 E[(x_t^T W) * (z_t^T log pi)]
        # psi_2 = 2 * E_x[:-1].dot(E_W) * E_z[:-1].dot(E_logpi)
        psi_2 = 2 * np.einsum('td, ti, kid -> tk', E_x[:-1], E_z[:-1], logpi_WT)

        # e_k^T \psi_3 e_k =
        #        =Tr(E[x_t x_t^T w_k w_k^T])               with w_k = W[:,k]  (kth col of weight matrix)
        #        = Tr(E[x_t x_t^T] \dot E[w_k w_k^T])
        #        = Tr(A^T \dot B)                           with A = A^T = E[x_t x_t^T]), B = E[w_k w_k^T]
        #        = \sum_{ij} A_{ij} * B_{ij}
        psi_3 = np.einsum('tij, kij -> tk', E_xxT[:-1], WWT)

        # s_{tk} = E[v_{tk}^2]
        s = psi_1 + psi_2 + psi_3
        assert np.all(s >= 0)
        for itr in range(n_iter):
            lambda_bs = self.lambda_bs

            # Eq (42)
            self.a = 2 * (m * lambda_bs).sum(axis=1) + K / 2.0 - 1.0
            self.a /= 2 * lambda_bs.sum(axis=1)

            # Eq (43)
            self.bs = np.sqrt(s - 2 * m * self.a[:, None] + self.a[:, None] ** 2)

    def vb_E_step_discrete_states(self):
        """
        Override the discrete state updates in pyhsmm to keep the necessary suff stats.
        """
        self.clear_caches()

        # Run the message passing algorithm
        trans_potential = np.exp(self.trans_distn.logpi)
        init_potential = self.pi_0
        likelihood_potential = self.vbem_aBl
        alphal = self._messages_forwards_log(trans_potential, init_potential, likelihood_potential)
        betal = self._messages_backwards_log(trans_potential, likelihood_potential)

        # Convert messages into expectations
        expected_states = alphal + betal
        expected_states -= expected_states.max(1)[:, None]
        np.exp(expected_states, out=expected_states)
        expected_states /= expected_states.sum(1)[:, None]

        Al = np.log(trans_potential)
        log_joints = alphal[:-1, :, None] + betal[1:, None, :] \
            + likelihood_potential[1:, None, :] + Al[None, :, :]
        log_joints -= log_joints.max(axis=(1, 2), keepdims=True)
        joints = np.exp(log_joints)
        joints /= joints.sum(axis=(1, 2), keepdims=True)

        # Compute the log normalizer log p(x_{1:T} | \theta, a, b)
        normalizer = logsumexp(alphal[0] + betal[0])

        # Save expected statistics
        self.expected_states = expected_states
        self.expected_joints = joints
        self.expected_transcounts = joints.sum(0)
        self._normalizer = normalizer

        # Update the "stateseq" variable too
        self.stateseq = self.expected_states.argmax(1).astype('int32')

        # Compute the entropy
        from pyslds.util import hmm_entropy
        params = (np.log(trans_potential), np.log(init_potential), likelihood_potential, normalizer)
        stats = (expected_states, self.expected_transcounts, normalizer)
        return hmm_entropy(params, stats)

    def expected_log_joint_probability(self):
        """
        Compute E_{q(z) q(x)} [log p(z) + log p(x | z) + log p(y | x, z)]
        """
        # E_{q(z)}[log p(z)]
        # todo: fix this to computed expected VLB instead
        elp = np.dot(self.expected_states[0], np.log(self.pi_0))
        elp += np.sum(self.expected_joints * np.log(self.trans_matrix + 1e-16))

        # E_{q(x)}[log p(y, x | z)]  is given by aBl
        # To get E_{q(x)}[ aBl ] we multiply and sum
        elp += np.sum(self.expected_states * self.vbem_aBl)
        return elp


class SoftmaxRecurrentSLDSStates(_SoftmaxRecurrentSLDSStatesVBEM,
                                 _SoftmaxRecurrentSLDSStatesMeanField):
    pass

