# tests/test_order_invariance_time_aware.py

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class LinearPolicy:
    gain: float

    def act(self, observation: float) -> float:
        return self.gain * observation


def simulate_episode(policy: LinearPolicy, rng: random.Random, t: int, horizon: int = 10) -> float:
    observation = 0.0
    total = 0.0
    for i in range(horizon):
        noise = (rng.random() - 0.5) + 0.05 * float(t)
        action = policy.act(observation)
        observation = 0.7 * observation + action + noise + 0.01 * float(i)
        total += observation
    return total


def trial_value(x: Any, rng: random.Random, t: int) -> float:
    if isinstance(x, (int, float)):
        return float(x) + 0.05 * float(t) + 0.01 * (rng.random() - 0.5)
    if isinstance(x, LinearPolicy):
        return simulate_episode(x, rng=rng, t=t)
    raise TypeError(f"Unsupported candidate type: {type(x)!r}")


def aggregate_vector(values: list[float]) -> tuple[float, float]:
    mean = sum(values) / len(values)
    mean_square = sum(v * v for v in values) / len(values)
    return (mean, mean_square)


def evaluate_design_context(
    *,
    candidate_index: int,
    x: Any,
    context_index: int,
    t: int,
    n_trials: int,
    master_seed: int,
) -> tuple[float, float]:
    values: list[float] = []
    for k in range(n_trials):
        seed = master_seed + candidate_index * 1_000_000 + context_index * 10_000 + k
        rng = random.Random(seed)
        values.append(trial_value(x, rng=rng, t=t))
    return aggregate_vector(values)


def test_time_aware_order_invariance_of_aggregated_feedback():
    """
    Framework correctness: aggregated feedback must be invariant to exploration order.

    We evaluate the same candidate set under the same context schedule, using design- and
    context-local RNG sequences (no global seed advancement). The aggregated feedback
    vectors returned to the "optimiser" must match exactly across different ask orders.
    """

    candidates: list[Any] = [
        1.0,
        2.0,
        LinearPolicy(gain=0.3),
        LinearPolicy(gain=0.7),
    ]
    contexts: list[int] = [0, 1, 2]

    n_trials = 25
    master_seed = 123_456

    def run(order: list[int]) -> dict[tuple[int, int], tuple[float, float]]:
        summaries: dict[tuple[int, int], tuple[float, float]] = {}
        for candidate_index in order:
            x = candidates[candidate_index]
            for context_index, t in enumerate(contexts):
                summaries[(candidate_index, t)] = evaluate_design_context(
                    candidate_index=candidate_index,
                    x=x,
                    context_index=context_index,
                    t=t,
                    n_trials=n_trials,
                    master_seed=master_seed,
                )
        return summaries

    order_1 = [0, 1, 2, 3]
    order_2 = [3, 1, 0, 2]

    s1 = run(order_1)
    s2 = run(order_2)

    assert set(s1.keys()) == set(s2.keys())
    for key in s1:
        assert s1[key] == s2[key]

