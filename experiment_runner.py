"""
CLI to run routing experiments across multiple seeds and policies.

Reads experiments/experiments.yml, builds constellations, instantiates routers
per policy, and runs DV rounds (or pure Dijkstra) to produce summary metrics.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Set, Tuple
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

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
    import yaml  # type: ignore

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


def run_experiments(
    config_path: Path,
    runs_csv: Path | None = None,
    aggregates_csv: Path | None = None,
    max_workers: int | None = None,
    use_processes: bool = True,
) -> List[Dict[str, object]]:
    cfg = load_config(config_path)
    start = time.time()

    existing_runs = load_runs_csv(runs_csv) if runs_csv else []
    seen_keys: Set[Tuple[str, str, int]] = {
        (str(r.get("experiment")), str(r.get("policy")), int(r.get("seed"))) for r in existing_runs
    }

    tasks: List[tuple[ExperimentConfig, RouteSelectionPolicy, int]] = []
    for exp in cfg.experiments:
        for policy_name in cfg.route_selection_policies:
            policy = RouteSelectionPolicy[policy_name]
            for offset in range(cfg.seed_count):
                seed = cfg.seed + offset
                key = (exp.name, policy.value, seed)
                if key in seen_keys:
                    continue
                tasks.append((exp, policy, seed))

    print(f"[run] queued {len(tasks)} new tasks (existing runs: {len(seen_keys)})")

    new_results: List[Dict[str, object]] = []
    if tasks:
        if use_processes:
            try:
                with ProcessPoolExecutor(max_workers=max_workers) as executor:
                    future_to_task = {
                        executor.submit(_run_task, asdict(exp), policy.value, seed): (exp.name, policy.value, seed)
                        for exp, policy, seed in tasks
                    }
                    for future in as_completed(future_to_task):
                        exp_name, policy_val, seed = future_to_task[future]
                        try:
                            res = future.result()
                            new_results.append(res)
                            if runs_csv:
                                append_run_row(runs_csv, res)
                            print(f"[run] completed experiment={exp_name} policy={policy_val} seed={seed} duration={res['duration_sec']:.2f}s")
                        except Exception as exc:
                            print(f"[run] failed experiment={exp_name} policy={policy_val} seed={seed}: {exc}")
            except (PermissionError, NotImplementedError, OSError) as exc:
                print(f"[run] process pool unavailable ({exc}), falling back to sequential execution")
                use_processes = False
        else:
            print("[run] using sequential execution")

        if not use_processes:
            for exp, policy, seed in tasks:
                res = _run_task(asdict(exp), policy.value, seed)
                new_results.append(res)
                if runs_csv:
                    append_run_row(runs_csv, res)
                print(f"[run] completed experiment={exp.name} policy={policy.value} seed={seed} duration={res['duration_sec']:.2f}s")

    results = existing_runs + new_results

    if aggregates_csv:
        write_aggregates_csv(aggregate_by_policy(results), aggregates_csv)

    elapsed = time.time() - start
    print(f"[run] completed {len(results)} total runs in {elapsed:.2f}s")
    return results


def aggregate_by_policy(results: Iterable[Dict[str, object]]) -> List[Dict[str, float]]:
    """
    Aggregate metrics per (experiment, policy), averaging only across seeds.
    """
    accum: Dict[tuple[str, str], Dict[str, float]] = {}
    counts: Dict[tuple[str, str], int] = {}
    meta: Dict[tuple[str, str], Dict[str, object]] = {}

    for res in results:
        exp = str(res["experiment"])
        policy = str(res["policy"])
        key = (exp, policy)

        metrics = res.get("metrics")
        if metrics is None:
            # Resume path: metrics are flattened at the top level in runs.csv.
            metrics = {
                "avg_reachable": res.get("avg_reachable", 0.0),
                "routers": res.get("routers", 0.0),
            }

        counts[key] = counts.get(key, 0) + 1
        bucket = accum.setdefault(key, {"avg_reachable_sum": 0.0, "routers_sum": 0.0})
        bucket["avg_reachable_sum"] += float(metrics.get("avg_reachable", 0.0))
        bucket["routers_sum"] += float(metrics.get("routers", 0.0))

        meta[key] = {
            "experiment": exp,
            "policy": policy,
            # Use the actual sampled counts recorded on the run.
            "satellites": int(res.get("satellites", 0)),
            "ground_stations": int(res.get("ground_stations", 0)),
        }

    aggregated_rows: List[Dict[str, float]] = []
    for key, sums in accum.items():
        n = counts[key]
        info = meta[key]
        aggregated_rows.append(
            {
                "experiment": info["experiment"],
                "policy": info["policy"],
                "satellites": info["satellites"],
                "ground_stations": info["ground_stations"],
                "runs": float(n),
                "avg_reachable": sums["avg_reachable_sum"] / n if n else 0.0,
                "avg_routers": sums["routers_sum"] / n if n else 0.0,
            }
        )

    return aggregated_rows


def _run_task(exp_dict: Dict[str, object], policy_value: str, seed: int) -> Dict[str, object]:
    start_run = time.time()
    exp = ExperimentConfig(
        name=str(exp_dict["name"]),
        satellites=int(exp_dict["satellites"]),
        ground_stations=int(exp_dict["ground_stations"]),
        sat_degree=int(exp_dict["sat_degree"]),
        ground_links=int(exp_dict["ground_links"]),
    )
    policy = RouteSelectionPolicy(policy_value)
    res = _run_single(exp, policy, seed)
    res["duration_sec"] = time.time() - start_run
    return res


def load_runs_csv(path: Path | None) -> List[Dict[str, object]]:
    if path is None or not path.exists():
        return []
    with path.open() as f:
        reader = csv.DictReader(f)
        rows: List[Dict[str, object]] = []
        for row in reader:
            # Normalize numeric fields so aggregation works on resumed runs.
            row["seed"] = int(row.get("seed", 0))
            for key in ("satellites", "ground_stations", "routers"):
                if key in row and row[key] != "":
                    row[key] = int(row[key])
            for key in (
                "avg_reachable",
                "min_reachable",
                "max_reachable",
                "duration_sec",
                "avg_dv_entries_examined",
                "avg_dv_entries_updated",
                "avg_dv_router_ops",
            ):
                if key in row and row[key] != "":
                    row[key] = float(row[key])
            rows.append(row)
        return rows


def append_run_row(path: Path, res: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "experiment",
                "policy",
                "seed",
                "satellites",
                "ground_stations",
                "routers",
                "avg_reachable",
                "min_reachable",
                "max_reachable",
                "avg_dv_entries_examined",
                "avg_dv_entries_updated",
            ],
        )
        if write_header:
            writer.writeheader()
        metrics = res.get("metrics", {})
        writer.writerow(
            {
                "experiment": res.get("experiment"),
                "policy": res.get("policy"),
                "seed": res.get("seed"),
                "satellites": res.get("satellites"),
                "ground_stations": res.get("ground_stations"),
                "routers": metrics.get("routers"),
                "avg_reachable": metrics.get("avg_reachable"),
                "min_reachable": metrics.get("min_reachable"),
                "max_reachable": metrics.get("max_reachable"),
                "avg_dv_examined": metrics.get("avg_dv_examined", 0.0),
                "avg_dv_updates": metrics.get("avg_dv_updates", 0.0),
            }
        )


def _run_single(exp: ExperimentConfig, policy: RouteSelectionPolicy, seed: int) -> Dict[str, object]:
    graph, satellites, ground = build_constellation_graph(
        expected_sats=exp.satellites,
        expected_ground=exp.ground_stations,
        seed=seed,
        sat_degree=exp.sat_degree,
        ground_links=exp.ground_links,
    )

    actual_sats = len(satellites)
    actual_ground = len(ground)

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
        # Record actual sampled counts, not just expectations.
        "satellites": actual_sats,
        "ground_stations": actual_ground,
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
    for r in routers.values():
        reset = getattr(r, "reset_changed_flag", None)
        if reset:
            reset()

    for _ in range(max_rounds):
        for r in routers.values():
            reset = getattr(r, "reset_changed_flag", None)
            if reset:
                reset()
        run_dv_round(graph, routers)
        if not any(getattr(r, "_changed", False) for r in routers.values()):
            break


def _summarize_routes(routers: Mapping[Node, Router]) -> Dict[str, object]:
    reachability_counts: List[int] = []
    examined_counts: List[int] = []
    update_counts: List[int] = []
    total_ops: List[int] = []
    for r in routers.values():
        table = getattr(r, "_routing_table", {})
        reachability_counts.append(len(table))
        if hasattr(r, "_ops_examined"):
            examined = getattr(r, "_ops_examined", 0)
            examined_counts.append(examined)
        if hasattr(r, "_ops_updates"):
            updates = getattr(r, "_ops_updates", 0)
            update_counts.append(updates)
        if hasattr(r, "_ops_examined") or hasattr(r, "_ops_updates"):
            total = getattr(r, "_ops_examined", 0) + getattr(r, "_ops_updates", 0)
            total_ops.append(total)
    avg_reach = sum(reachability_counts) / len(reachability_counts) if reachability_counts else 0.0
    return {
        "routers": len(routers),
        "avg_reachable": avg_reach,
        "min_reachable": min(reachability_counts) if reachability_counts else 0,
        "max_reachable": max(reachability_counts) if reachability_counts else 0,
        "avg_dv_entries_examined": (sum(examined_counts) / len(examined_counts)) if examined_counts else 0.0,
        "avg_dv_entries_updated": (sum(update_counts) / len(update_counts)) if update_counts else 0.0,
        "avg_dv_router_ops": (sum(total_ops) / len(total_ops)) if total_ops else 0.0,
    }


def write_results_csv(results: Iterable[Dict[str, object]], path: Path) -> None:
    """
    Write per-run results to CSV for downstream analysis.
    """
    fieldnames = [
        "experiment",
        "policy",
        "seed",
        "satellites",
        "ground_stations",
        "routers",
        "avg_reachable",
        "min_reachable",
        "max_reachable",
        "duration_sec",
        "avg_dv_entries_examined",
        "avg_dv_entries_updated",
        "avg_dv_router_ops",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for res in results:
            metrics = res.get("metrics", {})
            row = {
                "experiment": res.get("experiment"),
                "policy": res.get("policy"),
                "seed": res.get("seed"),
                "satellites": res.get("satellites"),
                "ground_stations": res.get("ground_stations"),
                "routers": metrics.get("routers"),
                "avg_reachable": metrics.get("avg_reachable"),
                "min_reachable": metrics.get("min_reachable"),
                "max_reachable": metrics.get("max_reachable"),
                "duration_sec": res.get("duration_sec", 0.0),
                "avg_dv_entries_examined": metrics.get("avg_dv_entries_examined", 0.0),
                "avg_dv_entries_updated": metrics.get("avg_dv_entries_updated", 0.0),
                "avg_dv_router_ops": metrics.get("avg_dv_router_ops", 0.0),
            }
            writer.writerow(row)


def write_aggregates_csv(aggregated: Iterable[Mapping[str, object]], path: Path) -> None:
    """
    Write aggregated metrics by policy to CSV.
    """
    fieldnames = [
        "experiment",
        "policy",
        "satellites",
        "ground_stations",
        "runs",
        "avg_reachable",
        "avg_routers",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in aggregated:
            writer.writerow(
                {
                    "experiment": row.get("experiment", ""),
                    "policy": row.get("policy", ""),
                    "satellites": row.get("satellites", 0),
                    "ground_stations": row.get("ground_stations", 0),
                    "runs": row.get("runs", 0.0),
                    "avg_reachable": row.get("avg_reachable", 0.0),
                    "avg_routers": row.get("avg_routers", 0.0),
                }
            )


def main() -> None:
    config_path = Path(__file__).parent / "experiments" / "experiments.yml"
    out_dir = Path(__file__).parent / "experiments" / "results"
    runs_csv = out_dir / "runs.csv"
    aggregates_csv = out_dir / "aggregates.csv"

    results = run_experiments(config_path, runs_csv=runs_csv, aggregates_csv=aggregates_csv)
    for res in results:
        print(res)
    print("Aggregated by policy:", aggregate_by_policy(results))
    print(f"Wrote runs to {runs_csv} and aggregates to {aggregates_csv}")


if __name__ == "__main__":
    main()
