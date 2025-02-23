from abc import ABC, abstractmethod
import logging
from typing import Any, List


class GeneralizedCertificate(ABC):
    def __init__(self, *args):
        pass

    @abstractmethod
    def compute_certificate(self, **kwargs) -> Any:
        """Compute the certificate value for a fixed learning algorithm."""
        pass

    @abstractmethod
    def optimize_meta_learning(self, **kwargs) -> Any:
        """Optimize the hyperparameters of the learning algorithm using the meta learning setup."""
        pass
