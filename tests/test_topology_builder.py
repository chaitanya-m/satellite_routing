from topology_builder import build_constellation_graph


def test_topology_builder_reproducible_with_seed():
    g1, sats1, ground1 = build_constellation_graph(
        expected_sats=20, expected_ground=5, seed=123, sat_degree=2, ground_links=1
    )
    g2, sats2, ground2 = build_constellation_graph(
        expected_sats=20, expected_ground=5, seed=123, sat_degree=2, ground_links=1
    )

    assert sats1 == sats2
    assert ground1 == ground2
    # Outgoing maps should match as well for reproducibility
    for node in sats1 + ground1:
        assert g1.outgoing(node) == g2.outgoing(node)


def test_topology_builder_connects_ground_to_satellites():
    g, sats, ground = build_constellation_graph(
        expected_sats=10, expected_ground=2, seed=321, sat_degree=1, ground_links=2
    )
    assert len(ground) > 0 and len(sats) > 0
    for gs in ground:
        out = g.outgoing(gs)
        # Ensure each ground station has at least one satellite neighbour
        assert any(neigh in sats for neigh in out)
