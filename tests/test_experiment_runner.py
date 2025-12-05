import importlib.util
import pytest
from pathlib import Path

from experiment_runner import run_experiments
from routers import RouteSelectionPolicy


def test_run_experiments_executes_with_small_config(tmp_path: Path):
    if importlib.util.find_spec("yaml") is None:
        pytest.skip("PyYAML not installed in test environment")

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
        assert res["metrics"]["routers"] > 0
