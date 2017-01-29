from pyslds.states import _SLDSStatesMaskedData

import autograd.numpy as anp
from autograd import grad

class NonconjugateRecurrentSLDSStates(_SLDSStatesMaskedData):
    """
    When the transition model is not amenable to PG augmentation, we
    need an alternative approach.  Many methods could apply:
    - HMC for x | z;  HMM message passing for z | x.
    - SMC for x and z
    - Assumed density filtering for x and z

    This class will do HMC.
    """
    def joint_log_probability(self, x):
        # A differentiable function to compute the joint probability for a given
        # latent state sequence
        # TODO: Vectorize?
        z, y, T = self.stateseq, self.data, self.T
        ll = 0

        # Initial likelihood
        mu_init, sigma_init = self.mu_init, self.sigma_init
        ll += -0.5 * anp.dot(x[0] - mu_init, anp.linalg.solve(sigma_init, x[0] - mu_init))

        # Continuous transition potentials
        As, Bs, Qs = self.As, self.Bs, self.sigma_statess
        for t in range(self.T-1):
            xpred = anp.dot(x[t], As[t].T) + anp.dot(self.inputs[t], Bs[t].T)
            ll += -0.5 * anp.dot(x[t+1] - xpred, anp.linalg.solve(Qs[t], x[t+1] - xpred))

        # Discrete transition probabilities
        log_pis = self.model.trans_distn.get_log_trans_matrices(x)
        ll += anp.sum(log_pis[:, z[:-1], z[1:]])

        # Observation likelihoods
        Cs, Ds, Rs = self.Cs, self.Ds, self.sigma_obss
        for t in range(self.T):
            ypred = anp.dot(x[t], Cs[t].T) + anp.dot(self.inputs[t], Ds[t].T)
            ll += -0.5 * anp.dot(y[t] - ypred, anp.linalg.solve(Rs[t], y[t] - ypred))

        return ll / T

    def resample_gaussian_states(self, step_sz=0.01, n_steps=10):
        # Run HMC
        from hips.inference.hmc import hmc
        nll = lambda x: -1 * self.joint_log_probability(anp.reshape(x, (self.T, self.D_latent)))
        dnll = grad(nll)
        x0 = self.gaussian_states.ravel()
        xf = hmc(nll, dnll, step_sz=step_sz, n_steps=n_steps, q_curr=x0)
        self.gaussian_states = xf.reshape((self.T, self.D_latent))
