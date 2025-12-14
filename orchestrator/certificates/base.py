# orchestrator/certificates/base.py

from __future__ import annotations
from abc import ABC, abstractmethod


class FeasibilityCertificate(ABC):
    """
    Protocol for feasibility certificates on Bernoulli trials.

    A certificate provides a lower confidence bound (LCB) on the true
    success probability given observed successes and trials.
    """

    @abstractmethod
    def lower_confidence_bound(self, successes: int, trials: int) -> float:
        """
        Return a lower confidence bound on p = P(success).

        Must satisfy:
            P(LCB <= p) >= 1 - alpha
        """
        raise NotImplementedError
