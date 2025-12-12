"""
Node abstraction for satsim.

Concrete simulators (satellites, ground stations, etc.)
will implement this interface.
"""

from abc import ABC, abstractmethod


class Node(ABC):
    """Abstract node in satsim."""

    @property
    @abstractmethod
    def id(self) -> str:
        """
        Stable identifier in a given snapshot.
        """
        raise NotImplementedError
