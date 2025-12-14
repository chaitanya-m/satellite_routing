# tests/test_dimensioning_2d.py

import random
import torch
import warnings

from botorch.exceptions import InputDataWarning

from experiments.satellites.min_feasible_coverage import MinLambdaForCoverage
from sim.dimensioning_2d import Dimensioning_2D
from optim.discrete_bandit import DiscreteBanditOptimiser

from orchestrator.certificates.base import FeasibilityCertificate

from orchestrator.certificates.bernoulli import (
    AllSuccessCertificate,
    ClopperPearsonCertificate,
    HoeffdingCertificate
)

TARGET_COVERAGE = 0.7
DELTA = 0.05   # allow 5% failure probability
ALPHA = 0.05   # 95% confidence


# ---------------------------------------------------------------------------
# Shared test harness
# ---------------------------------------------------------------------------

def run_experiment(certificate: FeasibilityCertificate) -> MinLambdaForCoverage:
    """
    Run the optimisation + evaluation loop once, returning the experiment.
    The only difference across tests is the certification objective.
    """

    torch.manual_seed(0)

    candidates = torch.tensor(
        [[0.0], [2.0], [10.0], [20.0], [30.0], [40.0], [500.0]],
        dtype=torch.double,
    )

    num_designs = 100
    evals_per_design = 100

    # NOTE:
    # For the all-success (failure-intolerant) certificate, the theoretical
    # minimum number of trials required is:
    #
    #   n >= ln(alpha) / ln(1 - delta)
    #     ≈ ln(0.05) / ln(0.95)
    #     ≈ 58.4
    #
    # so evals_per_design = 100 comfortably exceeds this bound.
    #
    # The same eval budget is reused for the tolerant certificates for
    # comparability, even though they require fewer trials in theory.

    seed = 0

    optimiser = DiscreteBanditOptimiser(candidates=candidates)
    experiment = MinLambdaForCoverage(
        target_coverage=TARGET_COVERAGE,
        delta=DELTA,
        certificate=certificate,
    )


    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=InputDataWarning)

        for _ in range(num_designs):
            lambda_outer = optimiser.ask()
            coverages = []

            for _ in range(evals_per_design):
                rng = random.Random(seed) # on every eval, use a different seed
                seed += 1

                sim = Dimensioning_2D(
                    inner_lambda=5.0,
                    inner_radius=1.0,
                    outer_radius=1.1,
                    coverage_distance=0.2,
                    rng=rng,
                )

                metrics = sim.evaluate(lambda_outer)
                coverages.append(metrics["coverage"])

                experiment.on_evaluation(lambda_outer, metrics)

            mean_coverage = sum(coverages) / len(coverages)
            optimiser.tell(lambda_outer, mean_coverage)

    return experiment


# ---------------------------------------------------------------------------
# Certificate-driven feasibility (orchestration-owned)
# ---------------------------------------------------------------------------

def is_feasible(
    *,
    experiment: MinLambdaForCoverage,
    certificate: FeasibilityCertificate,
    delta: float,
    design: float,
) -> bool:
    trials = experiment._trials.get(design, 0)
    successes = experiment._successes.get(design, 0)
    lcb = certificate.lower_confidence_bound(successes, trials)
    return lcb >= 1.0 - delta


def select_min_feasible(
    *,
    experiment: MinLambdaForCoverage,
    certificate: FeasibilityCertificate,
    delta: float,
) -> tuple[float, dict[str, float]]:
    feasible = [
        d
        for d in experiment._trials
        if is_feasible(experiment=experiment, certificate=certificate, delta=delta, design=d)
    ]
    if not feasible:
        raise AssertionError("No design certified feasible")

    best = min(feasible)
    return best, experiment._last_success_metrics[best]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_all_success_certificate_is_strict():
    """
    Failure-intolerant certificate (S = n).

    Expected behaviour:
    - Extremely strict
    - Likely certifies only very large satellite intensities
    - But if a design is certified, it must be the minimum such design
    """

    certificate = AllSuccessCertificate(alpha=ALPHA)
    experiment = run_experiment(certificate=certificate)

    feasible = [
        d for d in experiment._trials
        if is_feasible(experiment=experiment, certificate=certificate, delta=DELTA, design=d)
    ]

    assert feasible, "All-success certificate should certify at least one design"

    min_lambda, _ = select_min_feasible(experiment=experiment, certificate=certificate, delta=DELTA)

    assert is_feasible(experiment=experiment, certificate=certificate, delta=DELTA, design=min_lambda)
    assert min_lambda == min(feasible)


def test_clopper_pearson_certificate_is_failure_tolerant():
    """
    Exact failure-tolerant certificate (Clopper–Pearson).

    Expected behaviour:
    - Allows a small number of failures
    - Should certify designs that all-success rejects
    - Still enforces exact (1 - alpha) coverage guarantee

    Notes:

    For the Clopper–Pearson (exact, failure-tolerant) certificate, there is
    no hard minimum trial count: the lower confidence bound is valid for any n.

    Increasing n tightens the bound and allows certification with observed
    failures, but small n may simply fail to certify any design.

    Compared to all-success, this certificate should never be stricter
    in theory; tests later may assert relative ordering if desired.

    """

    certificate = ClopperPearsonCertificate(alpha=ALPHA)
    experiment = run_experiment(certificate=certificate)

    feasible = [
        d for d in experiment._trials
        if is_feasible(experiment=experiment, certificate=certificate, delta=DELTA, design=d)
    ]

    assert feasible, "Clopper–Pearson certificate should certify at least one design"

    min_lambda, _ = select_min_feasible(experiment=experiment, certificate=certificate, delta=DELTA)

    assert is_feasible(experiment=experiment, certificate=certificate, delta=DELTA, design=min_lambda)
    assert min_lambda == min(feasible)





def test_hoeffding_certificate_is_conservative_but_simple():
    """
    Failure-tolerant certificate using Hoeffding's inequality.

    Expected behaviour:
    - Simple, dependency-free
    - More conservative than Clopper–Pearson
    - Still tolerates failures

    Notes:
    For the Hoeffding certificate, the lower confidence bound is:

      LCB = p̂ - sqrt( ln(1/alpha) / (2n) )

    so the required number of trials to certify p >= 1 - delta satisfies:

      n >= ln(1/alpha) / (2 * (p̂ - (1 - delta))^2)

    There is no hard minimum n, but certification becomes impossible unless
    the empirical success rate p̂ is sufficiently above 1 - delta.

    Hoeffding may require more trials than Clopper–Pearson for the same
    guarantee, but should never certify a design that violates the target.

    """

    certificate = HoeffdingCertificate(alpha=ALPHA)
    experiment = run_experiment(certificate=certificate)

    feasible = [
        d for d in experiment._trials
        if is_feasible(experiment=experiment, certificate=certificate, delta=DELTA, design=d)
    ]

    assert feasible, "Hoeffding certificate should certify at least one design"

    min_lambda, _ = select_min_feasible(experiment=experiment, certificate=certificate, delta=DELTA)

    assert is_feasible(experiment=experiment, certificate=certificate, delta=DELTA, design=min_lambda)
    assert min_lambda == min(feasible)
