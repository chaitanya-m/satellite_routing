from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any
import random

from optimise_design import BayesianBanditOptimiser, run_optimisation
from optimise_design.interface import DesignProblem


@dataclass
class CircleAlignmentProblem(DesignProblem):
    """Toy design problem on a unit circle.

    - **Design space**: any angle in radians on the unit circle.
    - **Objective**: choose an angle that is as close as possible to a fixed
      ``target_angle``.
    - **Scoring**: convert angular distance into a score in [0, 1], where 1.0
      is a perfect match and 0.0 is the diametrically opposite point.
    - **Arm bucketing**: ``design_key`` rounds angles to two decimals so nearby
      proposals map to the same bandit arm.
    """

    target_angle: float = 0.0  # radians
    rng: random.Random = field(default_factory=random.Random)

    def sample_one_design(self) -> float:
        """Return a random angle uniformly distributed on [-pi, pi]."""

        return self.rng.uniform(-math.pi, math.pi)

    def evaluate(self, design: Any) -> float:
        """Compute a similarity score to the target angle.

        Angular distance is normalised by pi so that the farthest point (opposite
        side of the circle) scores 0.0 and the target itself scores 1.0.
        """
        angle = float(design)
        diff = abs((angle - self.target_angle + math.pi) % (2 * math.pi) - math.pi)
        return max(0.0, 1.0 - diff / math.pi)

    def design_key(self, design: Any) -> Any:
        """Bucket angles to 2 decimal places to ensure consistent arm keys."""

        return round(float(design), 2)


def test_bayesian_bandit_finds_target_angle():
    """Bandit should home in on the target angle on the unit circle.

    This test runs a modest budget of evaluations and expects the bandit to
    identify an angle near the target (0.5 radians) with a high similarity
    score.
    """
    problem = CircleAlignmentProblem(target_angle=0.5, rng=random.Random(42))
    optimiser = BayesianBanditOptimiser(rng=random.Random(123))

    best = run_optimisation(problem, optimiser, budget=400)
    assert best is not None
    design, score = best

    # The best design should be close to the target angle, and score should be high.
    assert abs(float(design) - problem.target_angle) < 0.2
    assert score > 0.8


def _poisson(lam: float, rng: random.Random) -> int:
    """Sample from a Poisson(lam) distribution using Knuth's algorithm.

    This is used to draw PPP counts for inner/outer circles. It returns an
    integer count consistent with the given intensity ``lam``.
    """

    L = math.exp(-lam)
    k = 0
    p = 1.0
    while p > L:
        k += 1
        p *= rng.random()
    return max(0, k - 1)


class SatellitePPPDimensioningProblem(DesignProblem):
    """PPP coverage toy with concentric circles.

    - Inner PPP: ground points on the unit circle with mean ``inner_lambda``.
    - Outer PPP: satellites on a slightly larger circle with mean given by the
      design (lambda) chosen from ``lambda_candidates``.
    - Score: fraction of inner points that have at least one outer point within
      ``coverage_distance``. This produces a score in [0, 1] usable by the
      Beta–Bernoulli bandit.
    - The test is designed so that higher outer lambdas yield better coverage.

    Parameters are fully specified by the caller to avoid hidden defaults in
    tests.
    """

    def __init__(
        self,
        lambda_candidates: tuple[float, ...],
        inner_lambda: float,
        inner_radius: float,
        outer_radius: float,
        coverage_distance: float,
        rng: random.Random | None = None,
    ) -> None:
        self.lambda_candidates = lambda_candidates
        self.inner_lambda = inner_lambda
        self.inner_radius = inner_radius
        self.outer_radius = outer_radius
        self.coverage_distance = coverage_distance
        self.rng = rng or random.Random()

    def sample_one_design(self) -> float:
        """Propose a lambda from the discrete candidate set."""

        return self.rng.choice(self.lambda_candidates)

    def evaluate(self, design: Any) -> float:
        """Sample PPP realisations and compute coverage fraction.

        Draw counts for inner/outer PPPs, sample angles uniformly, and compute
        the fraction of inner points within ``coverage_distance`` of any outer
        point. If no inner points are sampled, return full coverage (1.0).
        """

        poisson_lambda = float(design)
        # Sample inner and outer PPP counts using the shared RNG.
        n_inner = _poisson(self.inner_lambda, self.rng)
        n_outer = _poisson(poisson_lambda, self.rng)
        if n_inner == 0:
            # If no inner points are present in this draw, treat coverage as complete.
            return 1.0
        # Sample angles for each inner (ground) point and outer (satellite) point.
        inner_angles = [self.rng.uniform(-math.pi, math.pi) for _ in range(n_inner)]
        outer_angles = [self.rng.uniform(-math.pi, math.pi) for _ in range(n_outer)]

        covered = 0
        for theta in inner_angles:
            for phi in outer_angles:
                # Compute Euclidean distance between points on two concentric circles
                # given their angles (law of cosines). If within the coverage radius,
                # count this inner point as covered and move to the next inner point.
                dist = math.sqrt(
                    self.inner_radius**2
                    + self.outer_radius**2
                    - 2 * self.inner_radius * self.outer_radius * math.cos(theta - phi)
                )
                if dist <= self.coverage_distance:
                    covered += 1
                    break
        # Return the fraction of inner points that were covered in this realisation.
        return covered / n_inner

    def design_key(self, design: Any) -> Any:
        """Bucket lambda values to 3 decimal places for consistent arm keys."""

        return round(float(design), 3)


def _estimate_coverage(problem: SatellitePPPDimensioningProblem, lam: float, trials: int = 200) -> float:
    """Monte Carlo estimate of coverage for a fixed lambda.

    We build a fresh ``SatellitePPPDimensioningProblem`` with the same
    parameters as the original but with its own RNG. This keeps the estimate
    deterministic and ensures we don't mutate the optimiser's problem instance.
    For each trial, we evaluate a PPP realisation using the provided ``lam``
    (outer intensity) and accumulate the coverage fraction. The return value is
    the average coverage over ``trials`` samples, giving a crude estimate of
    how well that lambda performs without affecting the optimiser's internal
    state.
    """
    # Use a dedicated RNG to keep this estimate deterministic and separate from
    # the optimiser's RNG. Because we reuse the same ``tmp_problem`` instance
    # below, each call to ``evaluate`` consumes from this RNG in a contiguous
    # sequence across trials.
    rng = random.Random(0) # Fixed seed for reproducibility
    tmp_problem = SatellitePPPDimensioningProblem(
        lambda_candidates=problem.lambda_candidates,
        inner_lambda=problem.inner_lambda,
        inner_radius=problem.inner_radius,
        outer_radius=problem.outer_radius,
        coverage_distance=problem.coverage_distance,
        rng=rng,
    )
    total = 0.0
    for _ in range(trials):
        # Evaluate coverage for this realisation at the given lambda using the
        # temporary problem so we don't disturb the main optimiser/problem state.
        # Each call advances the shared RNG, so the trials form one contiguous
        # deterministic sequence of draws from the seeded generator.
        total += tmp_problem.evaluate(lam)
    # Return the average coverage across all trials.
    return total / float(trials)


def test_bayesian_bandit_satellite_ppp_selects_high_lambda():
    """Bandit should prefer higher lambdas that improve coverage in the PPP toy.

    After running the bandit, we estimate coverage for the selected lambda and
    compare it to the lowest candidate. The chosen lambda should deliver better
    coverage, clear a target coverage threshold, and coincide with the empirical
    best candidate.
    """
    # Supply candidate lambdas directly in the test (class requires them). The
    # highest candidate (50) should be needed to clear the coverage threshold.
    candidates = (0.0, 2.0, 10.0, 50.0)
    problem = SatellitePPPDimensioningProblem(
        lambda_candidates=candidates,
        inner_lambda=5.0,
        inner_radius=1.0,
        outer_radius=1.1,
        coverage_distance=0.35,
        rng=random.Random(123),
    )
    optimiser = BayesianBanditOptimiser(rng=random.Random(456))

    best = run_optimisation(problem, optimiser, budget=800)
    assert best is not None
    design, _score = best

    # Require that the chosen design achieves strong coverage; this prevents the
    # test from passing when only low-intensity candidates are provided.
    cov_best = _estimate_coverage(problem, design, trials=200)
    cov_low = _estimate_coverage(problem, min(problem.lambda_candidates), trials=200)
    # A mid-range lambda (20) should still fail the 0.9 target so the optimiser
    # cannot succeed without selecting the highest candidate.
    cov_twenty = _estimate_coverage(problem, 20.0, trials=200)
    assert cov_twenty < 0.99
    assert cov_best >= 0.99

    # Empirically evaluate all candidates to find the brute-force best.
    coverage_map = {lam: _estimate_coverage(problem, lam, trials=200) for lam in candidates}
    best_lambda_from_map = max(coverage_map.items(), key=lambda item: item[1])[0]

    # Best lambda should produce better coverage than the lowest candidate and match the empirical best.
    assert cov_best > cov_low + 0.1
    assert math.isclose(float(design), best_lambda_from_map, rel_tol=1e-6, abs_tol=1e-6)
