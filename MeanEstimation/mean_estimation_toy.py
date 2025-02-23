import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from generalized_certificate import GeneralizedCertificate
from MeanEstimation.utils import *
from typing import Any, List


class MeanEstimationCertificate(GeneralizedCertificate):
    def __init__(self, poisoning_probability, learning_rate):
        super().__init__()
        self.poisoning_probability = poisoning_probability
        self.learning_rate = learning_rate

    def compute_certificate(self, mu, Sigma, S, delta=0.0) -> Any:
        """Compute the certificate value for mean estimation."""
        return inf_1(
            mu,
            Sigma,
            S,
            self.learning_rate,
            self.poisoning_probability,
            kappa=0.0,
            delta=delta,
        )

    def optimize_meta_learning(self, means, covariances, kappa, delta=0.0) -> Any:
        """Optimize the hyperparameters of the learning algorithm using the meta learning setup."""
        return alt_min_mean_estimation_bound_meta(
            means,
            covariances,
            self.learning_rate,
            self.poisoning_probability,
            kappa=kappa,
            delta=0.0,
        )


# Example Use Case
epsilon = 0.15
delta = 0.0
eta = 1e-2

# Example parameters for the NIW prior
d = 2  # Dimensionality of the multivariate Gaussian
mean_prior = np.zeros(d)  # Prior mean vector
cov_prior = 0.1 * np.eye(d)  # Prior covariance matrix
df_prior = d + 2  # Prior degrees of freedom
scale_prior = np.eye(d)  # Prior scale matrix

# Combine parameters into a tuple
niw_params = (mean_prior, cov_prior, df_prior, scale_prior)

# Number of samples to generate
K = 10

# Generate samples from the NIW prior
means, covariances = generate_niw_prior_samples(d, niw_params, K)
kappa = 1.0

meancertificate = MeanEstimationCertificate(epsilon, eta)
S = meancertificate.optimize_meta_learning(means, covariances, kappa)
