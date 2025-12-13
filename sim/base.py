# sim/base.py

from __future__ import annotations

from typing import Any, Protocol


class Simulator(Protocol):
    """Contract: a black-box evaluator mapping a design to measured metrics."""

    def evaluate(self, design: Any) -> dict[str, float]:
        """Run one simulation at `design` and return measured quantities."""
        ...
