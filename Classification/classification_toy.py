import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from generalized_certificate import GeneralizedCertificate
from Classification.hinge_loss import *
from typing import Any, List


class ClassificationCertificate(GeneralizedCertificate):
    def __init__(self, poisoning_probability):
        self.poisoning_probability = poisoning_probability
        return

    def certificate(self, Z, sigma, eta) -> Any:
        """Compute the certificate value for binary classification."""
        return hinge_certificate(Z, sigma, eta, self.poisoning_probability)

    def optimize_certificate(
        self, Z_list, benign_loss_list, sigma_eta_pairs, kappa
    ) -> Any:
        """Optimize the hyperparameters of the learning algorithm using the meta learning setup."""
        return hinge_certificate_meta(
            Z_list, benign_loss_list, sigma_eta_pairs, self.poisoning_probability, kappa
        )
