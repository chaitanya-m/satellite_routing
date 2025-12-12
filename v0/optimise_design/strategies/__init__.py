"""Entry point for pluggable optimisation strategies.

Concrete strategy adapters for external libraries (e.g. Bayesian optimisers,
evolution strategies, or RL-based policy search) can live in this package and
implement :class:`optimise_design.interface.DesignOptimiser`. Locally
implemented strategies can also live here when needed; in both cases the goal
is to provide thin, reusable optimisation components that plug into the
design-optimisation core.
"""
