from constellation import (
    GROUND_STATION_BANDS,
    POPULAR_LAT_BANDS,
    generate_ground_stations,
    generate_poisson_constellation,
)


def _within_any_band(lat: float, bands) -> bool:
    return any(b.min_lat <= lat <= b.max_lat for b in bands)


def test_constellation_generation_is_reproducible():
    sats1 = generate_poisson_constellation(50, seed=123)
    sats2 = generate_poisson_constellation(50, seed=123)

    assert sats1 == sats2
    assert all(_within_any_band(s.lat, POPULAR_LAT_BANDS) for s in sats1)


def test_ground_station_generation_is_reproducible():
    gs1 = generate_ground_stations(20, seed=99)
    gs2 = generate_ground_stations(20, seed=99)

    assert gs1 == gs2
    assert all(_within_any_band(gs.lat, GROUND_STATION_BANDS) for gs in gs1)
