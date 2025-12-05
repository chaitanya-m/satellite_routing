"""
CLI to run routing experiments across multiple seeds and policies.

Reads experiments/experiments.yml, builds constellations, instantiates routers
per policy, and runs DV rounds (or pure Dijkstra) to produce summary metrics.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence
import copy

from algorithms import DijkstraEngine
from dijkstra_engine import SimpleDijkstraEngine
from distance_vector_engine import SimpleDistanceVectorEngine
from nodes import Node
from routers import (
    DijkstraRouter,
    GroundStationRouter,
    RouteSelectionPolicy,
    SatelliteRouter,
    set_route_selection_policy,
)
from routing import Router
from simulation import run_dv_round
from topology_builder import build_constellation_graph


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    satellites: int
    ground_stations: int
    sat_degree: int
    ground_links: int


@dataclass(frozen=True)
class Config:
    seed: int
    seed_count: int
    route_selection_policies: Sequence[str]
    experiments: Sequence[ExperimentConfig]


def load_config(path: Path) -> Config:
    try:
        import yaml  # type: ignore
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to load experiment configs") from exc

    data = yaml.safe_load(path.read_text())
    experiments = [
        ExperimentConfig(
            name=exp["name"],
            satellites=exp["satellites"],
            ground_stations=exp["ground_stations"],
            sat_degree=exp["sat_degree"],
            ground_links=exp["ground_links"],
        )
        for exp in data["experiments"]
    ]
    return Config(
        seed=int(data["seed"]),
        seed_count=int(data["seed_count"]),
        route_selection_policies=list(data["route_selection_policies"]),
        experiments=experiments,
    )


def run_experiments(config_path: Path) -> List[Dict[str, object]]:
    cfg = load_config(config_path)
    results: List[Dict[str, object]] = []

    for exp in cfg.experiments:
        for policy_name in cfg.route_selection_policies:
            policy = RouteSelectionPolicy[policy_name]
            for offset in range(cfg.seed_count):
                seed = cfg.seed + offset
                summary = _run_single(exp, policy, seed)
                results.append(summary)
    return results


def _run_single(exp: ExperimentConfig, policy: RouteSelectionPolicy, seed: int) -> Dict[str, object]:
    graph, satellites, ground = build_constellation_graph(
        expected_sats=exp.satellites,
        expected_ground=exp.ground_stations,
        seed=seed,
        sat_degree=exp.sat_degree,
        ground_links=exp.ground_links,
    )

    dv_engine = SimpleDistanceVectorEngine()
    dijkstra_engine: DijkstraEngine = SimpleDijkstraEngine()

    routers = _build_routers(graph, satellites, ground, policy, dv_engine, dijkstra_engine)

    if policy != RouteSelectionPolicy.DIJKSTRA_ONLY:
        _run_dv_until_stable(graph, routers, max_rounds=5)

    metrics = _summarize_routes(routers)
    return {
        "experiment": exp.name,
        "policy": policy.value,
        "seed": seed,
        "satellites": exp.satellites,
        "ground_stations": exp.ground_stations,
        "metrics": metrics,
    }


def _build_routers(
    graph,
    satellites: Iterable[Node],
    ground: Iterable[Node],
    policy: RouteSelectionPolicy,
    dv_engine: SimpleDistanceVectorEngine,
    dijkstra_engine: DijkstraEngine,
) -> Dict[Node, Router]:
    routers: Dict[Node, Router] = {}
    if policy == RouteSelectionPolicy.DIJKSTRA_ONLY:
        for node in [*satellites, *ground]:
            router = DijkstraRouter(node, dijkstra_engine)
            router.recompute_on_topology(graph, epoch=0)
            routers[node] = router
    else:
        set_route_selection_policy(policy)
        for gs in ground:
            router = GroundStationRouter(gs, dv_engine, dijkstra_engine)
            router.recompute_on_topology(graph, epoch=0)
            routers[gs] = router
        for sat in satellites:
            router = SatelliteRouter(sat, dv_engine)
            router.recompute_on_topology(graph, epoch=0)
            routers[sat] = router
    return routers


def _run_dv_until_stable(graph, routers: Dict[Node, Router], max_rounds: int) -> None:
    prev_tables: Mapping[Node, object] = {}
    for _ in range(max_rounds):
        run_dv_round(graph, routers)
        tables: Mapping[Node, object] = {
            node: copy.deepcopy(getattr(r, "_routing_table", {})) for node, r in routers.items()
        }
        if tables == prev_tables:
            break
        prev_tables = tables


def _summarize_routes(routers: Mapping[Node, Router]) -> Dict[str, object]:
    reachability_counts: List[int] = []
    for r in routers.values():
        table = getattr(r, "_routing_table", {})
        reachability_counts.append(len(table))
    avg_reach = sum(reachability_counts) / len(reachability_counts) if reachability_counts else 0.0
    return {
        "routers": len(routers),
        "avg_reachable": avg_reach,
        "min_reachable": min(reachability_counts) if reachability_counts else 0,
        "max_reachable": max(reachability_counts) if reachability_counts else 0,
    }


def main() -> None:
    config_path = Path(__file__).parent / "experiments" / "experiments.yml"
    results = run_experiments(config_path)
    for res in results:
        print(res)


if __name__ == "__main__":
    main()
