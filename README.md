# satsim – Commit 1 (interfaces only)

Clean starting point with only the required abstractions:

- `Node` interface
- `Graph` interface (directed, weighted)

No implementations yet.

---

Routing abstractions introduced:

- `routing.Router`: per-node routing behaviour and DV handling.
- `routing.RouteEntry`, `routing.DVMessage`: basic routing data types.
- `algorithms.DijkstraEngine`, `algorithms.DistanceVectorEngine`: graph algorithm interfaces used by routers.
