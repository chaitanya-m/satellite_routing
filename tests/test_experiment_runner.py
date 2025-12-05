from typing import Any, Dict, cast
import pytest
from pathlib import Path

from experiment_runner import aggregate_by_policy, run_experiments
from routers import RouteSelectionPolicy


def test_run_experiments_executes_with_small_config(tmp_path: Path):
    """Smoke-test: runner processes tiny config with DV_ONLY and DIJKSTRA_ONLY."""
    cfg = tmp_path / "exp.yml"
    cfg.write_text(
        """
seed: 1
seed_count: 1
route_selection_policies:
  - DV_ONLY
  - DIJKSTRA_ONLY
experiments:
  - name: tiny
    satellites: 5
    ground_stations: 2
    sat_degree: 2
    ground_links: 2
"""
    )

    results = run_experiments(cfg)
    assert len(results) == 2
    policies = {res["policy"] for res in results}
    assert RouteSelectionPolicy.DV_ONLY.value in policies
    assert RouteSelectionPolicy.DIJKSTRA_ONLY.value in policies
    for res in results:
        assert res["experiment"] == "tiny"
        metrics = cast(Dict[str, Any], res["metrics"])
        assert metrics["routers"] > 0


def test_runner_handles_multiple_seeds_and_aggregates(tmp_path: Path):
    """Runner should handle multiple seeds/policies and aggregate metrics correctly."""
    cfg = tmp_path / "exp_multi.yml"
    cfg.write_text(
        """
seed: 2
seed_count: 2
route_selection_policies:
  - DV_ONLY
  - DIJKSTRA_ONLY
experiments:
  - name: tiny_multi
    satellites: 6
    ground_stations: 3
    sat_degree: 2
    ground_links: 2
"""
    )

    results = run_experiments(cfg)
    # 2 policies * 2 seeds = 4 runs
    assert len(results) == 4
    aggregated = aggregate_by_policy(results)
    assert set(aggregated.keys()) == {
        RouteSelectionPolicy.DV_ONLY.value,
        RouteSelectionPolicy.DIJKSTRA_ONLY.value,
    }
    for policy, metrics in aggregated.items():
        m = cast(Dict[str, float], metrics)
        assert m["runs"] == 2.0
        assert m["avg_routers"] > 0
        assert m["avg_reachable"] >= 0.0
    # Aggregated runs should equal total runs per policy
    assert sum(cast(Dict[str, float], m)["runs"] for m in aggregated.values()) == 4.0


def test_aggregated_values_reflect_inputs():
    """Integration check: run a tiny config with two seeds and verify aggregation."""
    cfg = tmp_path = Path(__file__).parent / "agg_tmp.yml"
    cfg.write_text(
        """
seed: 0
seed_count: 2
route_selection_policies:
  - DV_ONLY
experiments:
  - name: agg_check
    satellites: 4
    ground_stations: 1
    sat_degree: 2
    ground_links: 1
"""
    )

    results = run_experiments(cfg)
    aggregated = aggregate_by_policy(results)

    # Only one policy and two seeds; aggregate should average the raw metrics.
    assert set(aggregated.keys()) == {RouteSelectionPolicy.DV_ONLY.value}
    metrics = cast(Dict[str, float], aggregated[RouteSelectionPolicy.DV_ONLY.value])
    raw_metrics = [cast(Dict[str, float], r["metrics"]) for r in results]
    assert metrics["runs"] == 2.0
    expected_avg_reach = sum(r["avg_reachable"] for r in raw_metrics) / 2
    expected_avg_routers = sum(r["routers"] for r in raw_metrics) / 2
    assert abs(metrics["avg_reachable"] - expected_avg_reach) < 1e-9
    assert abs(metrics["avg_routers"] - expected_avg_routers) < 1e-9
