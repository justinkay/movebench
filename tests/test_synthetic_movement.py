from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from movebench.movement_math import haversine_m
from movebench.synthetic.movement import (
    CSV_COLUMNS,
    PLANNED_ANOMALY_TYPES,
    InjectionSpec,
    SimulationConfig,
    benchmark_profile,
    demo_profile,
    simulate_movement,
    write_movement_csv,
)
from movebench.trajectories import (
    INDIVIDUAL_COL,
    LAT_COL,
    LON_COL,
    TIME_COL,
    compute_movement_features,
    load_trajectories_csv,
)


def _checkpoint_config(seed: int = 7) -> SimulationConfig:
    return SimulationConfig(
        profile_name="checkpoint-1-test",
        seed=seed,
        n_individuals=2,
        fixes_per_individual=120,
        anomalous_individual_count=1,
        session_min_fixes=12,
        session_max_fixes=24,
        within_session_min_minutes=50,
        within_session_max_minutes=60,
        between_session_min_hours=6,
        between_session_max_hours=12,
        injections=(
            InjectionSpec(
                anomaly_type="gps_spike_single",
                individual_index=1,
                injection_id="gps_spike_single_001",
                severity=1.0,
                notes="checkpoint single GPS spike",
                distance_m=50_000.0,
            ),
        ),
    )


def _time_deltas_minutes(group: pd.DataFrame) -> np.ndarray:
    times = pd.to_datetime(group[TIME_COL])
    return times.diff().dt.total_seconds().dropna().to_numpy() / 60.0


def _individual_for_anomaly(
    result: pd.DataFrame,
    anomaly_type: str,
) -> pd.DataFrame:
    anomaly_rows = result[result["synthetic:anomaly_type"] == anomaly_type]
    assert anomaly_rows[INDIVIDUAL_COL].nunique() == 1
    individual = anomaly_rows[INDIVIDUAL_COL].iloc[0]
    return result[result[INDIVIDUAL_COL] == individual].reset_index(drop=True)


def _step_lengths(group: pd.DataFrame) -> np.ndarray:
    return haversine_m(
        group[LAT_COL].shift().to_numpy(),
        group[LON_COL].shift().to_numpy(),
        group[LAT_COL].to_numpy(),
        group[LON_COL].to_numpy(),
    )


def test_profile_constructors_declare_planned_defaults():
    demo = demo_profile()
    benchmark = benchmark_profile()

    assert (demo.n_individuals, demo.fixes_per_individual) == (20, 240)
    assert demo.anomalous_individual_count == 7
    assert len({item.individual_index for item in demo.injections}) == 7

    assert (benchmark.n_individuals, benchmark.fixes_per_individual) == (100, 720)
    assert benchmark.anomalous_individual_count == 18
    assert len({item.individual_index for item in benchmark.injections}) == 18

    for config in (demo, benchmark):
        assert (config.session_min_fixes, config.session_max_fixes) == (12, 36)
        assert (
            config.within_session_min_minutes,
            config.within_session_max_minutes,
        ) == (50, 60)
        assert (
            config.between_session_min_hours,
            config.between_session_max_hours,
        ) == (6, 12)
        assert config.turn_concentration_kappa == 6.25
        assert config.gps_noise_sd_m == 3.0


def test_turn_concentration_must_be_nonnegative():
    config = replace(_checkpoint_config(), turn_concentration_kappa=-1.0)

    with pytest.raises(ValueError, match="turn_concentration_kappa"):
        simulate_movement(config)


def test_gps_noise_standard_deviation_must_be_nonnegative():
    config = replace(_checkpoint_config(), gps_noise_sd_m=-0.1)

    with pytest.raises(ValueError, match="gps_noise_sd_m"):
        simulate_movement(config)


def test_global_gps_noise_is_deterministic_and_only_changes_coordinates():
    latent_config = replace(_checkpoint_config(), gps_noise_sd_m=0.0)
    observed_config = replace(latent_config, gps_noise_sd_m=3.0)
    latent = simulate_movement(latent_config)
    observed = simulate_movement(observed_config)
    repeated = simulate_movement(observed_config)

    pd.testing.assert_frame_equal(observed, repeated)
    non_coordinate_columns = [
        column for column in CSV_COLUMNS if column not in {LON_COL, LAT_COL}
    ]
    pd.testing.assert_frame_equal(
        latent[non_coordinate_columns],
        observed[non_coordinate_columns],
    )
    coordinate_changed = (
        latent[[LON_COL, LAT_COL]].to_numpy()
        != observed[[LON_COL, LAT_COL]].to_numpy()
    ).any(axis=1)
    assert coordinate_changed.all()


def test_unknown_anomaly_type_raises_clear_error():
    config = replace(
        _checkpoint_config(),
        injections=(
            InjectionSpec(
                anomaly_type="not_a_supported_anomaly",
                individual_index=1,
                injection_id="unknown_001",
            ),
        ),
    )

    with pytest.raises(NotImplementedError, match="not_a_supported_anomaly"):
        simulate_movement(config)


def test_simulation_is_deterministic_and_seed_changes_coordinates():
    config = _checkpoint_config()
    first = simulate_movement(config)
    second = simulate_movement(config)
    changed = simulate_movement(replace(config, seed=config.seed + 1))

    pd.testing.assert_frame_equal(first, second)
    assert not first[[LON_COL, LAT_COL]].equals(changed[[LON_COL, LAT_COL]])


def test_schema_order_ids_coordinates_and_row_order():
    result = simulate_movement(_checkpoint_config())

    assert result.columns.tolist() == CSV_COLUMNS
    assert result["event-id"].is_unique
    assert result[LON_COL].between(-180.0, 180.0).all()
    assert result[LAT_COL].between(-90.0, 90.0).all()

    expected = result.sort_values(
        [INDIVIDUAL_COL, TIME_COL, "event-id"],
        kind="mergesort",
    ).reset_index(drop=True)
    pd.testing.assert_frame_equal(result, expected)


def test_timestamps_are_strictly_increasing_and_form_multiple_sessions():
    config = _checkpoint_config()
    result = simulate_movement(config)

    for _individual, group in result.groupby(INDIVIDUAL_COL, sort=False):
        deltas = _time_deltas_minutes(group)
        assert (deltas > 0).all()
        within_session = deltas <= 60.0
        between_sessions = deltas > 60.0
        assert within_session.any()
        assert between_sessions.sum() >= 2
        assert deltas[within_session].min() >= config.within_session_min_minutes
        assert deltas[within_session].max() <= config.within_session_max_minutes
        assert deltas[between_sessions].min() >= config.between_session_min_hours * 60
        assert deltas[between_sessions].max() <= config.between_session_max_hours * 60


def test_normal_and_injected_label_invariants():
    result = simulate_movement(_checkpoint_config())
    anomaly = result[result["synthetic:is_anomaly"]]
    normal = result[~result["synthetic:is_anomaly"]]

    assert len(anomaly) == 1
    assert normal["visible"].all()
    assert not normal["manually-marked-outlier"].any()
    assert not normal["algorithm-marked-outlier"].any()
    assert (normal["synthetic:anomaly_type"] == "normal").all()
    assert (normal["synthetic:injection_id"] == "none").all()
    assert (normal["synthetic:severity"] == 0.0).all()
    assert (normal["synthetic:notes"] == "baseline").all()

    injected = anomaly.iloc[0]
    assert not bool(injected["visible"])
    assert bool(injected["manually-marked-outlier"])
    assert not bool(injected["algorithm-marked-outlier"])
    assert injected["synthetic:anomaly_type"] == "gps_spike_single"
    assert injected["synthetic:injection_id"] == "gps_spike_single_001"
    assert injected["synthetic:severity"] == 1.0
    assert injected["synthetic:notes"] == "checkpoint single GPS spike"


def test_spike_is_inside_session_and_extreme_relative_to_baseline():
    result = simulate_movement(_checkpoint_config())
    individual = result[
        result[INDIVIDUAL_COL] == "synthetic-002"
    ].reset_index(drop=True)
    anomaly_index = int(np.flatnonzero(individual["synthetic:is_anomaly"])[0])
    times = pd.to_datetime(individual[TIME_COL])
    deltas_s = times.diff().dt.total_seconds().to_numpy()

    assert anomaly_index >= 2
    assert anomaly_index + 2 < len(individual)
    assert np.all(deltas_s[anomaly_index - 1 : anomaly_index + 3] <= 3600.0)

    step_lengths = haversine_m(
        individual[LAT_COL].shift().to_numpy(),
        individual[LON_COL].shift().to_numpy(),
        individual[LAT_COL].to_numpy(),
        individual[LON_COL].to_numpy(),
    )
    adjacent = np.asarray(
        [step_lengths[anomaly_index], step_lengths[anomaly_index + 1]]
    )
    baseline_mask = ~individual["synthetic:is_anomaly"].to_numpy()
    baseline_mask[anomaly_index + 1] = False
    baseline_steps = step_lengths[baseline_mask & np.isfinite(step_lengths)]
    baseline_q99 = float(np.quantile(baseline_steps, 0.99))
    baseline_median = float(np.median(baseline_steps))

    adjacent_dt_s = np.asarray(
        [deltas_s[anomaly_index], deltas_s[anomaly_index + 1]]
    )
    adjacent_speeds = adjacent / adjacent_dt_s
    baseline_speeds = baseline_steps / np.median(
        deltas_s[np.isfinite(deltas_s) & (deltas_s <= 3600.0)]
    )

    assert adjacent.min() > baseline_q99 * 100
    assert adjacent_speeds.min() > np.median(baseline_speeds) * 100
    assert adjacent.min() > baseline_median * 100


def test_csv_bytes_are_deterministic_and_movebench_compatible(tmp_path):
    config = _checkpoint_config()
    first_path = write_movement_csv(tmp_path / "first.csv", config)
    second_path = write_movement_csv(tmp_path / "second.csv", config)

    assert first_path.read_bytes() == second_path.read_bytes()
    raw_text = first_path.read_text(encoding="utf-8")
    assert "\r\n" not in raw_text
    assert raw_text.splitlines()[0].split(",") == CSV_COLUMNS

    loaded = load_trajectories_csv(str(first_path))
    assert len(loaded) == config.n_individuals * config.fixes_per_individual
    assert loaded[TIME_COL].notna().all()
    featured = compute_movement_features(loaded)
    assert len(featured) == len(loaded)
    assert {"step_length_m", "time_since_last", "turn_angle_rad"}.issubset(
        featured.columns
    )


def test_demo_profile_dimensions_allocation_and_labels():
    result = simulate_movement(demo_profile(seed=31))
    anomalies = result[result["synthetic:is_anomaly"]]
    normal = result[~result["synthetic:is_anomaly"]]
    anomalous_individuals = set(anomalies[INDIVIDUAL_COL])
    all_individuals = set(result[INDIVIDUAL_COL])

    assert result.shape == (20 * 240, len(CSV_COLUMNS))
    assert result[INDIVIDUAL_COL].nunique() == 20
    assert len(anomalous_individuals) == 7
    assert len(all_individuals - anomalous_individuals) == 13
    assert set(anomalies["synthetic:anomaly_type"]) == set(
        PLANNED_ANOMALY_TYPES
    )
    assert (
        anomalies.groupby("synthetic:anomaly_type")[INDIVIDUAL_COL].nunique()
        == 1
    ).all()
    assert (
        anomalies.groupby("synthetic:anomaly_type")["synthetic:injection_id"]
        .nunique()
        .eq(1)
        .all()
    )

    assert not anomalies["visible"].any()
    assert anomalies["manually-marked-outlier"].all()
    assert anomalies["synthetic:is_anomaly"].all()
    assert anomalies["synthetic:severity"].between(0.0, 1.0).all()
    assert anomalies["synthetic:injection_id"].ne("none").all()
    assert anomalies["synthetic:notes"].str.strip().ne("").all()
    assert normal["visible"].all()
    assert not normal["manually-marked-outlier"].any()
    assert not normal["synthetic:is_anomaly"].any()
    assert normal["synthetic:anomaly_type"].eq("normal").all()
    assert normal["synthetic:injection_id"].eq("none").all()
    assert normal["synthetic:severity"].eq(0.0).all()
    assert normal["synthetic:notes"].eq("baseline").all()
    assert not result["algorithm-marked-outlier"].any()


def test_demo_is_deterministic_seed_sensitive_and_temporally_valid():
    config = demo_profile(seed=41)
    first = simulate_movement(config)
    second = simulate_movement(config)
    changed = simulate_movement(demo_profile(seed=42))

    pd.testing.assert_frame_equal(first, second)
    assert not first[[LON_COL, LAT_COL]].equals(changed[[LON_COL, LAT_COL]])
    assert first[LON_COL].between(-180.0, 180.0).all()
    assert first[LAT_COL].between(-90.0, 90.0).all()
    for _individual, group in first.groupby(INDIVIDUAL_COL, sort=False):
        assert (_time_deltas_minutes(group) > 0).all()


def test_demo_injections_have_buffered_nonoverlapping_session_windows():
    result = simulate_movement(demo_profile(seed=31))
    anomaly_rows = result[result["synthetic:is_anomaly"]]

    for injection_id, injected in anomaly_rows.groupby(
        "synthetic:injection_id",
        sort=False,
    ):
        assert injection_id != "none"
        assert injected["synthetic:anomaly_type"].nunique() == 1
        individual = result[
            result[INDIVIDUAL_COL] == injected[INDIVIDUAL_COL].iloc[0]
        ].reset_index(drop=True)
        indexes = individual.index[
            individual["synthetic:injection_id"] == injection_id
        ].to_numpy()
        first, last = int(indexes.min()), int(indexes.max())
        assert first >= 2
        assert last + 2 < len(individual)
        assert not individual.loc[
            [first - 2, first - 1, last + 1, last + 2],
            "synthetic:is_anomaly",
        ].any()
        times = pd.to_datetime(individual[TIME_COL])
        surrounding_gaps = (
            times.iloc[first - 2 : last + 3]
            .diff()
            .dt.total_seconds()
            .dropna()
        )
        assert (surrounding_gaps <= 3600.0).all()


def test_explicit_overlapping_injections_are_rejected():
    config = replace(
        _checkpoint_config(),
        injections=(
            InjectionSpec(
                anomaly_type="gps_spike_single",
                individual_index=1,
                fix_index=50,
                injection_id="first",
            ),
            InjectionSpec(
                anomaly_type="gps_spike_single",
                individual_index=1,
                fix_index=51,
                injection_id="overlap",
            ),
        ),
    )

    with pytest.raises(ValueError, match="cannot be placed"):
        simulate_movement(config)


def test_demo_spike_signatures_are_extreme_and_few_are_isolated():
    result = simulate_movement(demo_profile(seed=31))
    for anomaly_type in ("gps_spike_single", "gps_spike_few"):
        individual = _individual_for_anomaly(result, anomaly_type)
        indexes = np.flatnonzero(
            individual["synthetic:anomaly_type"].eq(anomaly_type).to_numpy()
        )
        steps = _step_lengths(individual)
        baseline_median = float(
            np.nanmedian(
                steps[~individual["synthetic:is_anomaly"].to_numpy()]
            )
        )
        assert np.min(steps[indexes]) > baseline_median * 100
        if anomaly_type == "gps_spike_few":
            assert len(indexes) >= 2
            assert (np.diff(indexes) > 1).all()


def test_demo_remote_cluster_is_compact_and_far_from_baseline():
    result = simulate_movement(demo_profile(seed=31))
    individual = _individual_for_anomaly(result, "gps_spike_cluster")
    indexes = np.flatnonzero(
        individual["synthetic:anomaly_type"].eq("gps_spike_cluster").to_numpy()
    )
    cluster = individual.iloc[indexes]
    centroid_lat = float(cluster[LAT_COL].mean())
    centroid_lon = float(cluster[LON_COL].mean())
    cluster_radius = haversine_m(
        centroid_lat,
        centroid_lon,
        cluster[LAT_COL].to_numpy(),
        cluster[LON_COL].to_numpy(),
    )
    boundary_distance = haversine_m(
        centroid_lat,
        centroid_lon,
        individual.loc[indexes[0] - 1, LAT_COL],
        individual.loc[indexes[0] - 1, LON_COL],
    )

    assert float(np.max(cluster_radius)) < 200.0
    assert float(boundary_distance) > 10_000.0


@pytest.mark.parametrize(
    "anomaly_type",
    ["location_ping_pong", "location_clusters_interleaved"],
)
def test_demo_interleaved_location_signatures_repeat_large_steps(anomaly_type):
    result = simulate_movement(demo_profile(seed=31))
    individual = _individual_for_anomaly(result, anomaly_type)
    indexes = np.flatnonzero(
        individual["synthetic:anomaly_type"].eq(anomaly_type).to_numpy()
    )
    steps = _step_lengths(individual)
    baseline_median = float(
        np.nanmedian(steps[~individual["synthetic:is_anomaly"].to_numpy()])
    )
    internal_steps = steps[indexes[1:]]

    assert len(internal_steps) >= 5
    assert np.quantile(internal_steps, 0.25) > baseline_median * 100


def test_demo_transport_is_fast_and_straight():
    result = simulate_movement(demo_profile(seed=31))
    individual = _individual_for_anomaly(result, "transport_fast_linear")
    indexes = np.flatnonzero(
        individual["synthetic:anomaly_type"].eq(
            "transport_fast_linear"
        ).to_numpy()
    )
    steps = _step_lengths(individual)
    times = pd.to_datetime(individual[TIME_COL])
    deltas_s = times.diff().dt.total_seconds().to_numpy()
    baseline_speed = np.nanmedian(
        steps[~individual["synthetic:is_anomaly"].to_numpy()]
        / deltas_s[~individual["synthetic:is_anomaly"].to_numpy()]
    )
    transport_speed = np.median(steps[indexes] / deltas_s[indexes])
    path_length = float(np.sum(steps[indexes[1:]]))
    net_distance = float(
        haversine_m(
            individual.loc[indexes[0], LAT_COL],
            individual.loc[indexes[0], LON_COL],
            individual.loc[indexes[-1], LAT_COL],
            individual.loc[indexes[-1], LON_COL],
        )
    )
    downstream_step = steps[indexes[-1] + 1]
    baseline_steps = steps[
        ~individual["synthetic:is_anomaly"].to_numpy()
        & np.isfinite(steps)
    ]

    assert transport_speed > baseline_speed * 20
    assert net_distance / path_length > 0.99
    assert downstream_step < np.quantile(baseline_steps, 0.99) * 2


def test_demo_stationary_is_sustained_and_near_stationary():
    config = demo_profile(seed=31)
    result = simulate_movement(config)
    individual = _individual_for_anomaly(result, "stationary")
    indexes = np.flatnonzero(
        individual["synthetic:anomaly_type"].eq("stationary").to_numpy()
    )
    steps = _step_lengths(individual)
    baseline_median = float(
        np.nanmedian(steps[~individual["synthetic:is_anomaly"].to_numpy()])
    )
    duration_hours = (
        pd.to_datetime(individual.loc[indexes[-1], TIME_COL])
        - pd.to_datetime(individual.loc[indexes[0], TIME_COL])
    ).total_seconds() / 3600.0
    downstream_step = steps[indexes[-1] + 1]
    baseline_steps = steps[
        ~individual["synthetic:is_anomaly"].to_numpy()
        & np.isfinite(steps)
    ]

    assert len(indexes) >= 8
    assert duration_hours >= 6.0
    assert np.median(steps[indexes[1:]]) < baseline_median * 0.1
    assert downstream_step < np.quantile(baseline_steps, 0.99) * 2

    latent = simulate_movement(replace(config, gps_noise_sd_m=0.0))
    latent_stationary = _individual_for_anomaly(latent, "stationary")
    latent_indexes = np.flatnonzero(
        latent_stationary["synthetic:anomaly_type"].eq("stationary").to_numpy()
    )
    assert (
        latent_stationary.loc[latent_indexes, [LON_COL, LAT_COL]]
        .drop_duplicates()
        .shape[0]
        == 1
    )
    assert (
        individual.loc[indexes, [LON_COL, LAT_COL]].drop_duplicates().shape[0]
        == len(indexes)
    )


def test_demo_csv_remains_movebench_compatible(tmp_path):
    config = demo_profile(seed=31)
    path = write_movement_csv(tmp_path / "demo.csv", config)
    loaded = load_trajectories_csv(str(path))
    featured = compute_movement_features(loaded)

    assert len(loaded) == 20 * 240
    assert len(featured) == len(loaded)
    assert {"step_length_m", "time_since_last", "turn_angle_rad"}.issubset(
        featured.columns
    )
