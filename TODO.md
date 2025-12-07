TODO
----

- Precompute per-topology neighbour lists/cost arrays and reuse them inside relax instead of rebuilding neighbour/destination dicts each round.
- Tune convergence work: track touched destinations to stop DV earlier or adjust max_rounds based on diameter.
- Switch from Node-keyed dicts to integer ids with list/array tables to cut hash lookups in the inner loop.
- If Python remains the bottleneck, move the relax loop to a compiled path (NumPy/Numba/Cython/C).
