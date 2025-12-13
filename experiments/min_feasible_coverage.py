# experimental/min_feasible_coverage.py

from __future__ import annotations
from typing import Any, Dict, Literal
import math


CertType = Literal[
    "all_success",        # failure-intolerant (S = n)
    "clopper_pearson",    # exact, failure-tolerant
    "hoeffding",          # simple, failure-tolerant
]


class MinLambdaForCoverage:
    """
    Find the minimum design value whose probability of achieving
    coverage >= target_coverage is at least (1 - delta),
    with confidence (1 - alpha).

    Supports multiple certification objectives.
    """

    def __init__(
        self,
        target_coverage: float,
        delta: float,
        alpha: float,
        cert: CertType = "all_success",
    ):
        self.target_coverage = target_coverage
        self.delta = delta
        self.alpha = alpha
        self.cert = cert

        self._trials: Dict[Any, int] = {}
        self._successes: Dict[Any, int] = {}
        self._last_success_metrics: Dict[Any, dict[str, float]] = {}

    # ------------------------------------------------------------------
    # Optimisation objective (kept deliberately simple for now)
    # ------------------------------------------------------------------

    def objective(self, design: Any, metrics: dict[str, float]) -> float:
        # Smooth signal to guide optimiser
        return float(metrics["coverage"])

    # ------------------------------------------------------------------
    # Data collection
    # ------------------------------------------------------------------

    def on_evaluation(self, design: Any, metrics: dict[str, float]) -> None:
        # Ignore trivial worlds
        if metrics["n_ground"] == 0:
            return

        self._trials[design] = self._trials.get(design, 0) + 1

        if metrics["coverage"] >= self.target_coverage:
            self._successes[design] = self._successes.get(design, 0) + 1
            self._last_success_metrics[design] = metrics

    # ------------------------------------------------------------------
    # Certification: lower confidence bounds
    # ------------------------------------------------------------------

    def _lcb_all_success(self, successes: int, trials: int) -> float:
        """
        Failure-intolerant certificate.
        Exact Clopper–Pearson lower bound for S = n.
        """
        if trials == 0:
            return 0.0

        if successes == trials:
            return self.alpha ** (1.0 / trials)

        return 0.0

    def _lcb_clopper_pearson(self, successes: int, trials: int) -> float:
        """
        Exact one-sided Clopper–Pearson lower bound.
        Failure-tolerant.
        """
        if trials == 0 or successes == 0:
            return 0.0

        if successes == trials:
            return self.alpha ** (1.0 / trials)

        # Invert regularized incomplete beta via bisection
        a = successes
        b = trials - successes + 1

        try:
            from scipy.stats import beta
            return float(beta.ppf(self.alpha, a, b))
        except Exception:
            # Dependency-free fallback
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

    def _lcb_hoeffding(self, successes: int, trials: int) -> float:
        """
        Hoeffding lower confidence bound.
        Simple and conservative failure-tolerant certificate.
        """
        if trials == 0:
            return 0.0

        phat = successes / trials
        eps = math.sqrt(math.log(1.0 / self.alpha) / (2.0 * trials))
        return max(0.0, phat - eps)

    # ------------------------------------------------------------------
    # Feasibility
    # ------------------------------------------------------------------

    def _lower_confidence_bound(self, successes: int, trials: int) -> float:
        if self.cert == "all_success":
            return self._lcb_all_success(successes, trials)
        elif self.cert == "clopper_pearson":
            return self._lcb_clopper_pearson(successes, trials)
        elif self.cert == "hoeffding":
            return self._lcb_hoeffding(successes, trials)
        else:
            raise ValueError(f"Unknown certification type: {self.cert}")

    def is_feasible(self, design: Any) -> bool:
        trials = self._trials.get(design, 0)
        successes = self._successes.get(design, 0)

        lcb = self._lower_confidence_bound(successes, trials)
        return lcb >= 1.0 - self.delta

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------

    def select_min(self) -> tuple[Any, dict[str, float]]:
        feasible = [
            d for d in self._trials
            if self.is_feasible(d)
        ]

        if not feasible:
            raise AssertionError("No design certified feasible")

        best = min(feasible)
        return best, self._last_success_metrics[best]
