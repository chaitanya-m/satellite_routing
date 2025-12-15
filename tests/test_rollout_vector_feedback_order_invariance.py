# tests/test_rollout_vector_feedback_order_invariance.py

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Protocol


class Policy(Protocol):
    def act(self, observation: float, *, rng: random.Random, t: int, i: int) -> float: ...


@dataclass(frozen=True)
class LinearPolicy:
    gain: float

    def act(self, observation: float, *, rng: random.Random, t: int, i: int) -> float:
        return self.gain * observation


@dataclass(frozen=True)
class NoisyPolicy:
    gain: float
    noise_scale: float

    def act(self, observation: float, *, rng: random.Random, t: int, i: int) -> float:
        # Example: policy-internal randomness (e.g. a stochastic controller, a sampling policy,
        # or action noise from an RL agent).
        return self.gain * observation + self.noise_scale * (rng.random() - 0.5)


@dataclass(frozen=True)
class Trajectory:
    """
    A trajectory-valued per-trial object Z(π, ω, t).

    This is intentionally domain-neutral, but examples include:
    - `states[i]`: a sequence of network/link states, queue states, or system states.
    - `exogenous[i]`: a sequence of exogenous inputs such as demand, fades, failures.
    - `actions[i]`: a sequence of policy actions such as routing/scheduling decisions.
    - `outcomes[i]`: a sequence of observed outcomes such as delivered service, delay, loss.
    """

    states: tuple[float, ...]
    exogenous: tuple[float, ...]
    actions: tuple[float, ...]
    outcomes: tuple[float, ...]


def rollout(pi: Policy, *, rng: random.Random, t: int, horizon: int = 25) -> Trajectory:
    states: list[float] = [0.0]
    exogenous: list[float] = []
    actions: list[float] = []
    outcomes: list[float] = []

    backlog = 0.0
    observation = 0.0

    for i in range(horizon):
        exo = (rng.random() - 0.5) + 0.1 * float(t)
        action = pi.act(observation, rng=rng, t=t, i=i)

        capacity = 0.5 + 0.05 * float(t) + 0.1 * (rng.random() - 0.5)
        demand = 0.6 + 0.1 * (rng.random() - 0.5)

        served = max(0.0, min(demand + backlog, capacity + max(0.0, action)))
        backlog = max(0.0, backlog + demand - served)

        observation = 0.7 * observation + exo + 0.3 * action

        states.append(backlog)
        exogenous.append(exo)
        actions.append(action)
        outcomes.append(served)

    return Trajectory(
        states=tuple(states),
        exogenous=tuple(exogenous),
        actions=tuple(actions),
        outcomes=tuple(outcomes),
    )


def trial_summary(z: Trajectory) -> tuple[float, int, float]:
    """
    Per-trial metric vector derived from a trajectory.

    - component 1: average "shortfall" proxy (mean backlog)
    - component 2: any-violation indicator (1 if backlog crosses a threshold at any time)
    - component 3: average "latency" proxy (mean backlog after burn-in)
    """

    backlog = z.states[1:]
    mean_backlog = sum(backlog) / len(backlog)

    violation = 1 if any(b > 0.75 for b in backlog) else 0

    burn_in = max(0, len(backlog) // 5)
    tail = backlog[burn_in:] if burn_in < len(backlog) else backlog
    mean_tail_backlog = sum(tail) / len(tail)

    return (mean_backlog, violation, mean_tail_backlog)


def aggregate(trials: list[tuple[float, int, float]]) -> tuple[float, float, float]:
    mean_shortfall = sum(s for s, _, _ in trials) / len(trials)
    prob_violation = sum(v for _, v, _ in trials) / len(trials)
    mean_latency = sum(l for _, _, l in trials) / len(trials)
    return (mean_shortfall, prob_violation, mean_latency)


def evaluate(
    *,
    candidate_index: int,
    pi: Policy,
    t: int,
    n_trials: int,
    master_seed: int,
) -> tuple[float, float, float]:
    """
    Evaluate s(π,t) from n(π,t)=n_trials trajectory trials.

    Determinism contract: the k-th trial for a fixed (π,t) always uses the same RNG seed,
    regardless of exploration order.
    """

    trials: list[tuple[float, int, float]] = []
    for k in range(n_trials):
        seed = master_seed + candidate_index * 1_000_000 + int(t) * 10_000 + k
        rng = random.Random(seed)
        z = rollout(pi, rng=rng, t=t)
        trials.append(trial_summary(z))
    return aggregate(trials)


def test_rollout_vector_feedback_is_invariant_to_exploration_order():
    """
    Framework correctness test:
    for fixed policies π, fixed contexts t, fixed n(π,t), and design/context-local RNG,
    aggregated feedback s(π,t) must not depend on the optimiser ask order.
    """

    candidates: list[Policy] = [
        LinearPolicy(gain=0.0),
        LinearPolicy(gain=0.3),
        NoisyPolicy(gain=0.2, noise_scale=0.1),
        NoisyPolicy(gain=0.6, noise_scale=0.2),
    ]
    contexts: list[int] = [0, 1, 2]

    n_trials = 50
    master_seed = 123_456

    def run(order: list[int]) -> dict[tuple[int, int], tuple[float, float, float]]:
        summaries: dict[tuple[int, int], tuple[float, float, float]] = {}
        for candidate_index in order:
            pi = candidates[candidate_index]
            for t in contexts:
                summaries[(candidate_index, t)] = evaluate(
                    candidate_index=candidate_index,
                    pi=pi,
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


def test_context_changes_can_legitimately_change_feedback():
    """
    Sanity check for concept drift:
    changing the scheduled context t is allowed to change s(π,t).
    """

    pi: Policy = NoisyPolicy(gain=0.2, noise_scale=0.1)
    master_seed = 123_456
    n_trials = 50

    s_t0 = evaluate(candidate_index=0, pi=pi, t=0, n_trials=n_trials, master_seed=master_seed)
    s_t2 = evaluate(candidate_index=0, pi=pi, t=2, n_trials=n_trials, master_seed=master_seed)

    assert s_t0 != s_t2
