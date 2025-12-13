from __future__ import annotations
import math
from .base import FeasibilityCertificate


class AllSuccessCertificate(FeasibilityCertificate):
    """
    Failure-intolerant certificate (S = n).
    Exact Clopper–Pearson lower bound for the zero-failure case.
    """

    def __init__(self, alpha: float):
        self.alpha = alpha

    def lower_confidence_bound(self, successes: int, trials: int) -> float:
        if trials == 0:
            return 0.0
        if successes == trials:
            return self.alpha ** (1.0 / trials)
        return 0.0


class ClopperPearsonCertificate(FeasibilityCertificate):
    """
    Exact one-sided Clopper–Pearson lower confidence bound.
    Failure-tolerant.
    """

    def __init__(self, alpha: float):
        self.alpha = alpha

    def lower_confidence_bound(self, successes: int, trials: int) -> float:
        if trials == 0 or successes == 0:
            return 0.0

        if successes == trials:
            return self.alpha ** (1.0 / trials)

        a = successes
        b = trials - successes + 1

        try:
            from scipy.stats import beta
            return float(beta.ppf(self.alpha, a, b))
        except Exception:
            import mpmath as mp

            target = mp.mpf(self.alpha)

            def I(x):
                return mp.betainc(a, b, 0, x, regularized=True)

            lo, hi = mp.mpf("0.0"), mp.mpf("1.0")
            for _ in range(80):
                mid = (lo + hi) / 2
                if I(mid) < target:
                    lo = mid
                else:
                    hi = mid
            return float((lo + hi) / 2)


class HoeffdingCertificate(FeasibilityCertificate):
    """
    Hoeffding lower confidence bound.
    Simple, conservative, dependency-free.
    """

    def __init__(self, alpha: float):
        self.alpha = alpha

    def lower_confidence_bound(self, successes: int, trials: int) -> float:
        if trials == 0:
            return 0.0

        phat = successes / trials
        eps = math.sqrt(math.log(1.0 / self.alpha) / (2.0 * trials))
        return max(0.0, phat - eps)
