from typing import Any, Dict, cast
from collections import Counter
import pandas as pd
from pathlib import Path

from experiment_runner import (
    aggregate_by_policy,
    run_experiments,
    write_aggregates_csv,
    write_results_csv,
)
from routers import RouteSelectionPolicy


def test_run_experiments_executes_with_small_config(tmp_path: Path):
    """Smoke-test: runner processes tiny config with DV_ONLY and DIJKSTRA_ONLY."""
    cfg = tmp_path / "exp.yml"
    cfg.write_text(
        """
seed: 1
seed_count: 2
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

    results = run_experiments(cfg, use_processes=False)
    assert len(results) == 4  # 2 policies * 2 seeds
    policy_counts = Counter(res["policy"] for res in results)
    assert policy_counts[RouteSelectionPolicy.DV_ONLY.value] == 2
    assert policy_counts[RouteSelectionPolicy.DIJKSTRA_ONLY.value] == 2
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
    satellites: 5
    ground_stations: 3
    sat_degree: 2
    ground_links: 2
  - name: tiny_multi_b
    satellites: 10
    ground_stations: 4
    sat_degree: 3
    ground_links: 3
"""
    )

    results = run_experiments(cfg, use_processes=False)
    # 2 experiments * 2 seeds * 2 policies = 8 runs
    assert len(results) == 8
    aggregated = aggregate_by_policy(results)
    # Aggregated per (experiment, policy)
    assert len(aggregated) == 4
    for row in aggregated:
        assert row["runs"] == 2.0  # 2 seeds per experiment/policy
        assert row["avg_routers"] > 0
        assert row["avg_reachable"] >= 0.0

    # Also test writing to CSV for both runs and aggregates
    runs_csv = tmp_path / "runs.csv"
    agg_csv = tmp_path / "aggregates.csv"
    write_results_csv(results, runs_csv)
    write_aggregates_csv(aggregated, agg_csv)
    assert runs_csv.exists()
    assert agg_csv.exists()
    runs_df = pd.read_csv(runs_csv)
    agg_df = pd.read_csv(agg_csv)
    assert len(runs_df) == len(results)
    assert len(agg_df) == len(aggregated)
    print("Runs DataFrame:\n", runs_df)
    print("Aggregates DataFrame:\n", agg_df)
    assert "duration_sec" in runs_df.columns


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

    results = run_experiments(cfg, use_processes=False)
    aggregated = aggregate_by_policy(results)

    # Only one experiment/policy and two seeds; aggregate should average the raw metrics.
    assert len(aggregated) == 1
    metrics = cast(Dict[str, float], aggregated[0])
    raw_metrics = [cast(Dict[str, float], r["metrics"]) for r in results]
    assert metrics["runs"] == 2.0
    expected_avg_reach = sum(r["avg_reachable"] for r in raw_metrics) / 2
    expected_avg_routers = sum(r["routers"] for r in raw_metrics) / 2
    assert abs(metrics["avg_reachable"] - expected_avg_reach) < 1e-9
    assert abs(metrics["avg_routers"] - expected_avg_routers) < 1e-9


def test_runner_two_experiments_two_seeds(tmp_path: Path):
    """End-to-end: two experiments, two seeds, single policy, CSV outputs."""
    cfg = tmp_path / "exp_two.yml"
    cfg.write_text(
        """
seed: 3
seed_count: 2
route_selection_policies:
  - DV_ONLY
experiments:
  - name: sats_5
    satellites: 5
    ground_stations: 1
    sat_degree: 2
    ground_links: 1
  - name: sats_10
    satellites: 10
    ground_stations: 2
    sat_degree: 2
    ground_links: 2
"""
    )

    results = run_experiments(cfg, use_processes=False)
    # 2 experiments * 2 seeds * 1 policy = 4 runs
    assert len(results) == 4
    # Ensure each experiment appears twice (once per seed)
    counts = Counter(res["experiment"] for res in results)
    assert counts["sats_5"] == 2
    assert counts["sats_10"] == 2

    aggregated = aggregate_by_policy(results)
    assert len(aggregated) == 2  # one row per experiment/policy
    for row in aggregated:
        assert row["runs"] == 2.0

    runs_csv = tmp_path / "runs_two.csv"
    agg_csv = tmp_path / "aggregates_two.csv"
    write_results_csv(results, runs_csv)
    write_aggregates_csv(aggregated, agg_csv)
    assert runs_csv.exists()
    assert agg_csv.exists()
    runs_df = pd.read_csv(runs_csv)
    agg_df = pd.read_csv(agg_csv)
    assert len(runs_df) == len(results)
    assert len(agg_df) == len(aggregated)
    print("Two-experiment runs DataFrame:\n", runs_df)
    print("Two-experiment aggregates DataFrame:\n", agg_df)
    assert "duration_sec" in runs_df.columns
