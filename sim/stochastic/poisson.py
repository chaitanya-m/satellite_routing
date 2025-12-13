import math
import random


def sample_poisson(lam: float, rng: random.Random) -> int:
    """Sample Poisson(lam) using Knuth's algorithm.

    Suitable for small-to-moderate lam. Deterministic given rng.
    """
    L = math.exp(-lam)
    k = 0
    p = 1.0
    while p > L:
        k += 1
        p *= rng.random()
    return max(0, k - 1)
