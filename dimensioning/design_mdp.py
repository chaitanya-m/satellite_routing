"""Design-space MDP utilities.

We treat *designs* themselves as the state space of an MDP. The design space
may consist of policies (e.g., Go policies), PPP outcomes (e.g., satellite
candidate constellations), or any other representation of a design. Actions
move the MDP between points in this design space, and each design state can be
evaluated by an external simulator or environment.

Abstract view: design space and design MDP
------------------------------------------

* A design space ``𝔻`` is the set of all admissible designs.
* A *design point* ``d ∈ 𝔻`` is one concrete design (policy, constellation,
  facility layout, etc.).
* An environment or simulator evaluates a design by producing trajectories,
  rewards, or KPIs; an objective then scores the design.

This module provides a small MDP harness where:

* :class:`State.design` represents the current design (or partial design).
* Each step applies an :class:`Action` to update ``design`` and obtain a new
  design state and reward.

Two main examples:

* **Satellite dimensioning with PPP.**
  The design space is the set of all constellations in an orbital region.
  A Poisson point process (PPP) on that region is used once per episode to
  sample a finite cloud of candidate satellite positions. Each design point can
  encode which PPP candidates are selected or how they are configured, and
  each design state is evaluated in terms of coverage characteristics (and
  ultimately blocking probability or other KPIs).

* **Go / AlphaZero-style policy search.**
  The design space is the set of all possible policies (or neural network
  parameters) for playing Go. Each design point is a particular policy (for
  example, the weights of a policy/value network). Actions move the MDP
  between policies in this design space. Each policy (design state) is
  evaluated via self-play, which also trains a policy network and a value
  network. No PPP is needed; the same design-space MDP abstraction applies.

PPP-based sampling is just one way to populate the design space; Poisson
utilities for that case live in :mod:`dimensioning.ppp_utils`. The core
interfaces in this module are neutral and can support both satellite
dimensioning and Go-style optimisation with minimal code changes.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Tuple


@dataclass
class State:
    """State of a design-space MDP.

    Attributes:
        t:
            Discrete decision step index.
        design:
            The current design (or partial design) the agent is considering.
            This may encode a policy, a constellation, or any other
            domain-specific representation of a point in the design space.
    """

    t: int
    design: Any


@dataclass(frozen=True)
class Action:
    """Control decision applied at a state.

    The payload is deliberately unstructured; environments and policies are
    expected to agree on its meaning.
    """

    data: Any


DesignTransitionFn = Callable[[Any, Action], Any]
RewardFn = Callable[[State, Action, State], float]
TerminationFn = Callable[[State], bool]


@dataclass
class DesignEnvironment:
    """Generic MDP environment over a fixed design world.

    The environment is agnostic to how the world was generated; it only assumes
    that designs can be represented as opaque Python objects. This supports
    PPP-based candidate sets for dimensioning, policies for Go-style
    optimisation, or any other context where an external simulator can evaluate
    the quality of a design.
    """

    design_transition: DesignTransitionFn
    reward_fn: RewardFn
    termination_fn: TerminationFn

    def step(self, state: State, action: Action) -> Tuple[State, float, bool, Mapping[str, Any]]:
        """Advance the environment by a single decision step.

        Returns:
            new_state:
                The successor state after applying ``action``.
            reward:
                Scalar reward for the transition.
            done:
                ``True`` if the new state is terminal according to
                :attr:`termination_fn`.
            info:
                Auxiliary data that callers may use for diagnostics. The
                environment currently returns an empty mapping.
        """

        new_design = self.design_transition(state.design, action)

        new_state = State(
            t=state.t + 1,
            design=new_design,
        )

        reward = self.reward_fn(state, action, new_state)
        done = self.termination_fn(new_state)
        info: Mapping[str, Any] = {}
        return new_state, reward, done, info
