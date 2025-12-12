TODO
----

1. Precompute per-topology neighbour lists/cost arrays and reuse them inside relax instead of rebuilding neighbour/destination dicts each round.
2. Tune convergence work: track touched destinations to stop DV earlier or adjust max_rounds based on diameter.
3. Switch from Node-keyed dicts to integer ids with list/array tables to cut hash lookups in the inner loop.
4. If Python remains the bottleneck, move the relax loop to a compiled path (NumPy/Numba/Cython/C).


5. Within a given period, which links are visible and how long for each satellite?
6. zig zag persistent homology - subset of simplicial homology - capture system in terms of coverage holes, 
7. Add coverage, intensity metrics to the simulation, as well as appropriate coverage/intensity calculation model
8. Figure out how to make our snapshot simulation a continuous one - either event based or time segment based.