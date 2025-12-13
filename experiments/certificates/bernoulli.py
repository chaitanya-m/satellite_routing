from __future__ import annotations
import math
from .base import FeasibilityCertificate


class AllSuccessCertificate(FeasibilityCertificate):
    """
    Failure-intolerant certificate (S = n).
    Exact Clopper–Pearson lower bound for the zero-failure case.

    This tests that all trials were successful. If any failures
    are observed, the lower confidence bound is zero.

    The minimum number of trials to certify a non-zero bound is:
        n >= log(alpha) / log(p_target) 
    where p_target is the desired lower bound on the success probability.

    For example, to certify p >= 0.9 with 95% confidence,
    we need at least n = log(0.05) / log(0.9) = 28 successful trials.

    To certify p >= 0.95 with 95% confidence, we need at least
    n = log(0.05) / log(0.95 ) = 58 successful trials.

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

    Given S observed successes out of n Bernoulli trials, this computes
    the exact lower confidence bound p_lower such that:

        P(X >= S | X ~ Binomial(n, p_lower)) = alpha

    Equivalently:
        p_lower = Beta.ppf(alpha, S, n - S + 1)

    This means that if the true success probability were any smaller than
    p_lower, then observing S or more successes would occur with probability
    less than alpha, and would be statistically implausible.

    Example:
        S = 18, n = 20, alpha = 0.05
        p_lower = Beta.ppf(0.05, 18, 3) ≈ 0.716

    Interpretation:
        With 95% confidence, the true success probability is at least 0.716.

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
    Failure-tolerant and distribution-free.

    Given S successes out of n Bernoulli trials, let p̂ = S / n.
    Hoeffding's inequality implies:

        P(p ≤ p̂ − ε) ≤ exp(−2 n ε²)

    Setting:
        ε = sqrt( ln(1/alpha) / (2 n) )

    yields a one-sided (1 − alpha) lower confidence bound:
        p_lower = p̂ − sqrt( ln(1/alpha) / (2 n) )

    Interpretation:
        With confidence 1 − alpha, the true success probability is at least p_lower.

    Example:
        S = 18, n = 20, alpha = 0.05
        p̂ = 0.9
        p_lower ≈ 0.626

    Compared to Clopper–Pearson, this bound is simpler and distribution-free,
    but more conservative, especially for small n.

    """

    def __init__(self, alpha: float):
        self.alpha = alpha

    def lower_confidence_bound(self, successes: int, trials: int) -> float:
        if trials == 0:
            return 0.0

        phat = successes / trials
        eps = math.sqrt(math.log(1.0 / self.alpha) / (2.0 * trials))
        return max(0.0, phat - eps)
