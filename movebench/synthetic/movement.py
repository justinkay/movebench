"""Deterministic synthetic Movebank-like movement data generation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from movebench.movement_math import EARTH_RADIUS_M


CSV_COLUMNS = [
    "event-id",
    "visible",
    "timestamp",
    "location-long",
    "location-lat",
    "algorithm-marked-outlier",
    "manually-marked-outlier",
    "sensor-type",
    "individual-taxon-canonical-name",
    "tag-local-identifier",
    "individual-local-identifier",
    "study-name",
    "set",
    "synthetic:is_anomaly",
    "synthetic:anomaly_type",
    "synthetic:injection_id",
    "synthetic:severity",
    "synthetic:notes",
]

GPS_SPIKE_SINGLE = "gps_spike_single"
GPS_SPIKE_FEW = "gps_spike_few"
GPS_SPIKE_CLUSTER = "gps_spike_cluster"
LOCATION_PING_PONG = "location_ping_pong"
LOCATION_CLUSTERS_INTERLEAVED = "location_clusters_interleaved"
TRANSPORT_FAST_LINEAR = "transport_fast_linear"
STATIONARY = "stationary"
PLANNED_ANOMALY_TYPES = (
    GPS_SPIKE_SINGLE,
    GPS_SPIKE_FEW,
    GPS_SPIKE_CLUSTER,
    LOCATION_PING_PONG,
    LOCATION_CLUSTERS_INTERLEAVED,
    TRANSPORT_FAST_LINEAR,
    STATIONARY,
)
IMPLEMENTED_ANOMALY_TYPES = frozenset(PLANNED_ANOMALY_TYPES)

DEFAULT_ANCHOR_LON = -110.5885
DEFAULT_ANCHOR_LAT = 44.4279
DEFAULT_START_TIMESTAMP = "2024-01-01 00:00:00+00:00"


@dataclass(frozen=True)
class InjectionSpec:
    """One fix-level anomaly episode.

    ``individual_index`` is zero-based. ``fix_index`` identifies the first
    anomalous fix; when omitted, the simulator deterministically chooses an
    eligible in-session window.
    """

    anomaly_type: str
    individual_index: int
    fix_index: int | None = None
    injection_id: str | None = None
    severity: float = 1.0
    notes: str = ""
    distance_m: float = 50_000.0
    bearing_degrees: float | None = None
    count: int | None = None
    duration_fixes: int | None = None


@dataclass(frozen=True)
class SimulationConfig:
    """Configuration for a synthetic Movebank-like movement CSV."""

    profile_name: str = "custom"
    seed: int = 0
    anchor_lon: float = DEFAULT_ANCHOR_LON
    anchor_lat: float = DEFAULT_ANCHOR_LAT
    n_individuals: int = 2
    fixes_per_individual: int = 120
    anomalous_individual_count: int = 1
    session_min_fixes: int = 12
    session_max_fixes: int = 36
    within_session_min_minutes: int = 50
    within_session_max_minutes: int = 60
    between_session_min_hours: int = 6
    between_session_max_hours: int = 12
    start_timestamp: str = DEFAULT_START_TIMESTAMP
    mean_step_length_m: float = 60.0
    step_length_cv: float = 0.35
    turn_concentration_kappa: float = 6.25
    individual_start_spread_m: float = 4_000.0
    gps_noise_sd_m: float = 3.0
    gps_quantization_m: float | None = 1.0
    study_name: str = "Synthetic movement benchmark"
    taxon_name: str = "Synthetic animal"
    set_name: str = "train"
    injections: tuple[InjectionSpec, ...] = field(default_factory=tuple)


def demo_profile(
    seed: int = 0,
    anchor_lon: float = DEFAULT_ANCHOR_LON,
    anchor_lat: float = DEFAULT_ANCHOR_LAT,
) -> SimulationConfig:
    """Return the planned small visual smoke-test profile."""
    anomaly_types = PLANNED_ANOMALY_TYPES
    injections = tuple(
        InjectionSpec(
            anomaly_type=anomaly_type,
            individual_index=13 + index,
            injection_id=f"{anomaly_type}_001",
            severity=0.65,
            notes=f"demo {anomaly_type} injection",
            distance_m=50_000.0,
            duration_fixes=72 if anomaly_type == STATIONARY else None,
        )
        for index, anomaly_type in enumerate(anomaly_types)
    )
    return SimulationConfig(
        profile_name="demo",
        seed=seed,
        anchor_lon=anchor_lon,
        anchor_lat=anchor_lat,
        n_individuals=20,
        fixes_per_individual=240,
        anomalous_individual_count=7,
        session_min_fixes=12,
        session_max_fixes=36,
        within_session_min_minutes=50,
        within_session_max_minutes=60,
        between_session_min_hours=6,
        between_session_max_hours=12,
        individual_start_spread_m=4_000.0,
        study_name="Synthetic movement benchmark demo",
        injections=injections,
    )


def benchmark_profile(
    seed: int = 0,
    anchor_lon: float = DEFAULT_ANCHOR_LON,
    anchor_lat: float = DEFAULT_ANCHOR_LAT,
) -> SimulationConfig:
    """Return the planned anomaly-sparse evaluation profile."""
    allocation = (
        (GPS_SPIKE_SINGLE, (0.25, 0.45, 0.65, 0.85, 1.0)),
        (GPS_SPIKE_FEW, (0.25, 0.45, 0.65, 0.85, 1.0)),
        (GPS_SPIKE_CLUSTER, (0.35, 0.60, 0.80, 1.0)),
        (LOCATION_PING_PONG, (0.35, 0.60, 0.80, 1.0)),
        (LOCATION_CLUSTERS_INTERLEAVED, (0.35, 0.60, 0.80, 1.0)),
        (TRANSPORT_FAST_LINEAR, (0.35, 0.60, 0.80, 1.0)),
        (STATIONARY, (0.35, 0.60, 0.80, 1.0)),
    )
    allocation_rng = np.random.default_rng(
        np.random.SeedSequence([int(seed), 5])
    )
    anomalous_individuals = allocation_rng.permutation(100)[:30].tolist()
    injections: list[InjectionSpec] = []
    allocation_index = 0
    type_counts: dict[str, int] = {}
    for anomaly_type, severities in allocation:
        for severity in severities:
            ordinal = type_counts.get(anomaly_type, 0) + 1
            type_counts[anomaly_type] = ordinal
            distance_m = 20_000.0 + 80_000.0 * severity
            injections.append(
                InjectionSpec(
                    anomaly_type=anomaly_type,
                    individual_index=anomalous_individuals[allocation_index],
                    injection_id=f"{anomaly_type}_{ordinal:03d}",
                    severity=severity,
                    notes=f"benchmark {anomaly_type} injection",
                    distance_m=distance_m,
                )
            )
            allocation_index += 1

    return SimulationConfig(
        profile_name="benchmark",
        seed=seed,
        anchor_lon=anchor_lon,
        anchor_lat=anchor_lat,
        n_individuals=100,
        fixes_per_individual=720,
        anomalous_individual_count=30,
        session_min_fixes=12,
        session_max_fixes=36,
        within_session_min_minutes=50,
        within_session_max_minutes=60,
        between_session_min_hours=6,
        between_session_max_hours=12,
        individual_start_spread_m=15_000.0,
        study_name="Synthetic movement benchmark",
        injections=tuple(injections),
    )


def simulate_movement(config: SimulationConfig) -> pd.DataFrame:
    """Generate a deterministic synthetic movement table."""
    _validate_config(config)
    injections = tuple(config.injections)
    unsupported = sorted(
        {spec.anomaly_type for spec in injections}
        - IMPLEMENTED_ANOMALY_TYPES
    )
    if unsupported:
        raise NotImplementedError(
            "Unsupported synthetic anomaly type(s): "
            + ", ".join(unsupported)
        )

    injections_by_individual: dict[int, list[InjectionSpec]] = {}
    for spec in injections:
        injections_by_individual.setdefault(spec.individual_index, []).append(spec)

    rows: list[dict[str, object]] = []
    injection_ordinal = 0
    for individual_index in range(config.n_individuals):
        timestamp_rng = _individual_rng(config.seed, individual_index, 1)
        movement_rng = _individual_rng(config.seed, individual_index, 2)
        injection_rng = _individual_rng(config.seed, individual_index, 3)
        observation_rng = _individual_rng(config.seed, individual_index, 4)
        start_offset_rng = _individual_rng(config.seed, individual_index, 5)

        timestamps, session_ids = _generate_timestamps(config, timestamp_rng)
        east_m, north_m = _generate_correlated_random_walk(config, movement_rng)
        start_east_m, start_north_m = _generate_individual_start_offset(
            config.individual_start_spread_m,
            start_offset_rng,
        )
        east_m += start_east_m
        north_m += start_north_m
        labels = [_normal_labels() for _ in range(config.fixes_per_individual)]

        reserved_fix_indexes: set[int] = set()
        individual_specs = injections_by_individual.get(individual_index, [])
        for spec in individual_specs:
            injection_ordinal += 1
            fix_indexes, reserved_indexes = _resolve_injection_indexes(
                spec,
                session_ids=session_ids,
                reserved_fix_indexes=reserved_fix_indexes,
                rng=injection_rng,
            )
            injection_id = (
                str(spec.injection_id).strip()
                if spec.injection_id not in (None, "")
                else f"{spec.anomaly_type}_{injection_ordinal:03d}"
            )
            _apply_injection(
                east_m,
                north_m,
                anomaly_type=spec.anomaly_type,
                fix_indexes=fix_indexes,
                spec=spec,
                injection_id=injection_id,
                rng=injection_rng,
                labels=labels,
            )
            reserved_fix_indexes.update(reserved_indexes)

        observed_east_m, observed_north_m = _apply_gps_observation_model(
            east_m,
            north_m,
            standard_deviation_m=config.gps_noise_sd_m,
            quantization_m=config.gps_quantization_m,
            rng=observation_rng,
        )
        longitudes, latitudes = _local_meters_to_lon_lat(
            observed_east_m,
            observed_north_m,
            anchor_lon=config.anchor_lon,
            anchor_lat=config.anchor_lat,
        )
        _validate_coordinates(longitudes, latitudes)

        individual_id = _individual_identifier(
            individual_index,
            individual_specs,
        )
        tag_id = f"synthetic-tag-{individual_index + 1:03d}"
        for fix_index in range(config.fixes_per_individual):
            event_id = 9_000_000_000_000 + (individual_index + 1) * 1_000_000 + fix_index + 1
            rows.append(
                {
                    "event-id": event_id,
                    **labels[fix_index],
                    "timestamp": _format_movebank_timestamp(timestamps[fix_index]),
                    "location-long": float(longitudes[fix_index]),
                    "location-lat": float(latitudes[fix_index]),
                    "sensor-type": "gps",
                    "individual-taxon-canonical-name": config.taxon_name,
                    "tag-local-identifier": tag_id,
                    "individual-local-identifier": individual_id,
                    "study-name": config.study_name,
                    "set": config.set_name,
                }
            )

    result = pd.DataFrame(rows, columns=CSV_COLUMNS)
    result = result.sort_values(
        ["individual-local-identifier", "timestamp", "event-id"],
        kind="mergesort",
    ).reset_index(drop=True)
    return result


def write_movement_csv(path: str | Path, config: SimulationConfig) -> Path:
    """Generate and write a deterministic Movebank-like CSV."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    exported = simulate_movement(config).copy()

    for column in (
        "visible",
        "algorithm-marked-outlier",
        "manually-marked-outlier",
        "synthetic:is_anomaly",
    ):
        exported[column] = exported[column].map({True: "true", False: "false"})
    for column in ("location-long", "location-lat"):
        exported[column] = exported[column].map(lambda value: f"{float(value):.7f}")
    exported["synthetic:severity"] = exported["synthetic:severity"].map(
        _format_severity
    )

    exported.to_csv(output_path, index=False, lineterminator="\n")
    return output_path


def _validate_config(config: SimulationConfig) -> None:
    if config.n_individuals <= 0:
        raise ValueError("n_individuals must be positive")
    if config.fixes_per_individual <= 0:
        raise ValueError("fixes_per_individual must be positive")
    if not 0 <= config.anomalous_individual_count <= config.n_individuals:
        raise ValueError(
            "anomalous_individual_count must be between zero and n_individuals"
        )
    if config.session_min_fixes <= 0:
        raise ValueError("session_min_fixes must be positive")
    if config.session_max_fixes < config.session_min_fixes:
        raise ValueError(
            "session_max_fixes must be at least session_min_fixes"
        )
    if config.within_session_min_minutes <= 0:
        raise ValueError("within_session_min_minutes must be positive")
    if config.within_session_max_minutes < config.within_session_min_minutes:
        raise ValueError(
            "within_session_max_minutes must be at least "
            "within_session_min_minutes"
        )
    if config.within_session_max_minutes > 60:
        raise ValueError(
            "within_session_max_minutes must not exceed 60 for the planned "
            "vibecleaning burst threshold"
        )
    if config.between_session_min_hours <= 1:
        raise ValueError("between_session_min_hours must exceed one hour")
    if config.between_session_max_hours < config.between_session_min_hours:
        raise ValueError(
            "between_session_max_hours must be at least "
            "between_session_min_hours"
        )
    if config.mean_step_length_m <= 0:
        raise ValueError("mean_step_length_m must be positive")
    if config.step_length_cv <= 0:
        raise ValueError("step_length_cv must be positive")
    if config.turn_concentration_kappa < 0:
        raise ValueError("turn_concentration_kappa must be nonnegative")
    if config.individual_start_spread_m < 0:
        raise ValueError("individual_start_spread_m must be nonnegative")
    if config.gps_noise_sd_m < 0:
        raise ValueError("gps_noise_sd_m must be nonnegative")
    if config.gps_quantization_m is not None and config.gps_quantization_m < 0:
        raise ValueError("gps_quantization_m must be nonnegative or None")
    if not -180.0 <= config.anchor_lon <= 180.0:
        raise ValueError("anchor_lon must be between -180 and 180")
    if not -85.0 <= config.anchor_lat <= 85.0:
        raise ValueError(
            "anchor_lat must be between -85 and 85 for local conversion"
        )
    if not str(config.set_name).strip():
        raise ValueError("set_name must not be empty")

    injections = tuple(config.injections)
    injection_ids: set[str] = set()
    injected_individuals: set[int] = set()
    for spec in injections:
        if not str(spec.anomaly_type).strip():
            raise ValueError("injection anomaly_type must not be empty")
        if not 0 <= spec.individual_index < config.n_individuals:
            raise ValueError(
                f"Injection individual_index out of range: {spec.individual_index}"
            )
        if spec.fix_index is not None and not (
            0 <= spec.fix_index < config.fixes_per_individual
        ):
            raise ValueError(
                f"Injection fix_index out of range: {spec.fix_index}"
            )
        if not 0.0 < float(spec.severity) <= 1.0:
            raise ValueError("injection severity must satisfy 0 < severity <= 1")
        if float(spec.distance_m) <= 0.0:
            raise ValueError("injection distance_m must be positive")
        if spec.count is not None and spec.count <= 0:
            raise ValueError("injection count must be positive")
        if spec.duration_fixes is not None and spec.duration_fixes <= 0:
            raise ValueError("injection duration_fixes must be positive")
        if spec.injection_id not in (None, ""):
            injection_id = str(spec.injection_id).strip()
            if injection_id in injection_ids:
                raise ValueError(f"Duplicate injection_id: {injection_id}")
            injection_ids.add(injection_id)
        injected_individuals.add(spec.individual_index)

    if injections and len(injected_individuals) != config.anomalous_individual_count:
        raise ValueError(
            "anomalous_individual_count does not match the number of individuals "
            "referenced by injections"
        )

    start = pd.Timestamp(config.start_timestamp)
    if pd.isna(start):
        raise ValueError("start_timestamp is invalid")


def _individual_rng(seed: int, individual_index: int, stream: int) -> np.random.Generator:
    seed_sequence = np.random.SeedSequence(
        [int(seed), int(individual_index), int(stream)]
    )
    return np.random.default_rng(seed_sequence)


def _individual_identifier(
    individual_index: int,
    injections: Sequence[InjectionSpec],
) -> str:
    base = f"synthetic-{individual_index + 1:03d}"
    anomaly_types = sorted({spec.anomaly_type for spec in injections})
    if not anomaly_types:
        return base
    return f"{base}-{'-'.join(anomaly_types)}"


def _generate_timestamps(
    config: SimulationConfig,
    rng: np.random.Generator,
) -> tuple[list[pd.Timestamp], np.ndarray]:
    start = pd.Timestamp(config.start_timestamp)
    if start.tzinfo is None:
        start = start.tz_localize("UTC")
    else:
        start = start.tz_convert("UTC")

    timestamps = [start]
    session_ids = np.zeros(config.fixes_per_individual, dtype=np.int64)
    written = 0
    session_id = 0
    while written < config.fixes_per_individual:
        remaining = config.fixes_per_individual - written
        session_length = min(
            remaining,
            int(
                rng.integers(
                    config.session_min_fixes,
                    config.session_max_fixes + 1,
                )
            ),
        )
        session_ids[written : written + session_length] = session_id

        for offset in range(session_length):
            absolute_index = written + offset
            if absolute_index == 0:
                continue
            if offset == 0:
                hours = int(
                    rng.integers(
                        config.between_session_min_hours,
                        config.between_session_max_hours + 1,
                    )
                )
                delta = pd.Timedelta(hours=hours)
            else:
                minutes = int(
                    rng.integers(
                        config.within_session_min_minutes,
                        config.within_session_max_minutes + 1,
                    )
                )
                delta = pd.Timedelta(minutes=minutes)
            timestamps.append(timestamps[-1] + delta)

        written += session_length
        session_id += 1

    if len(timestamps) != config.fixes_per_individual:
        raise RuntimeError("Internal timestamp generation length mismatch")
    return timestamps, session_ids


def _generate_correlated_random_walk(
    config: SimulationConfig,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    count = config.fixes_per_individual
    east_m = np.zeros(count, dtype=np.float64)
    north_m = np.zeros(count, dtype=np.float64)
    if count == 1:
        return east_m, north_m

    shape = 1.0 / (config.step_length_cv**2)
    scale = config.mean_step_length_m / shape
    step_lengths = rng.gamma(shape=shape, scale=scale, size=count - 1)
    headings = np.empty(count - 1, dtype=np.float64)
    heading = float(rng.uniform(-np.pi, np.pi))
    for index in range(count - 1):
        if index:
            heading += float(
                rng.vonmises(mu=0.0, kappa=config.turn_concentration_kappa)
            )
        headings[index] = heading

    east_m[1:] = east_m[0] + np.cumsum(step_lengths * np.cos(headings))
    north_m[1:] = north_m[0] + np.cumsum(step_lengths * np.sin(headings))
    return east_m, north_m


def _generate_individual_start_offset(
    spread_m: float,
    rng: np.random.Generator,
) -> tuple[float, float]:
    if spread_m == 0.0:
        return 0.0, 0.0
    radius_m = spread_m * np.sqrt(rng.random())
    bearing_rad = rng.uniform(-np.pi, np.pi)
    return (
        float(radius_m * np.cos(bearing_rad)),
        float(radius_m * np.sin(bearing_rad)),
    )


def _resolve_injection_indexes(
    spec: InjectionSpec,
    *,
    session_ids: np.ndarray,
    reserved_fix_indexes: set[int],
    rng: np.random.Generator,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    anomaly_offsets, placement_span = _injection_shape(spec)
    candidates = _eligible_injection_starts(
        session_ids=session_ids,
        placement_span=placement_span,
        reserved_fix_indexes=reserved_fix_indexes,
        require_single_session=spec.anomaly_type != STATIONARY,
    )
    if spec.fix_index is not None:
        if int(spec.fix_index) not in candidates:
            raise ValueError(
                f"Injection {spec.injection_id or spec.anomaly_type!r} fix_index "
                f"{spec.fix_index} cannot be placed inside a session with at "
                "least two normal fixes before and after it"
            )
        start = int(spec.fix_index)
    else:
        if not candidates:
            raise ValueError(
                f"No eligible within-session window is available for "
                f"{spec.injection_id or spec.anomaly_type!r}"
            )
        start = int(rng.choice(np.asarray(candidates, dtype=np.int64)))

    fix_indexes = tuple(start + offset for offset in anomaly_offsets)
    reserved_indexes = tuple(range(start - 2, start + placement_span + 2))
    return fix_indexes, reserved_indexes


def _injection_shape(spec: InjectionSpec) -> tuple[tuple[int, ...], int]:
    severity = float(spec.severity)
    if spec.anomaly_type == GPS_SPIKE_SINGLE:
        return (0,), 1
    if spec.anomaly_type == GPS_SPIKE_FEW:
        count = spec.count or max(2, int(round(2 + 3 * severity)))
        offsets = tuple(range(0, 2 * count - 1, 2))
        return offsets, offsets[-1] + 1

    default_lengths = {
        GPS_SPIKE_CLUSTER: max(3, int(round(3 + 4 * severity))),
        LOCATION_PING_PONG: max(6, int(round(6 + 4 * severity))),
        LOCATION_CLUSTERS_INTERLEAVED: max(6, int(round(6 + 6 * severity))),
        TRANSPORT_FAST_LINEAR: max(5, int(round(5 + 5 * severity))),
        STATIONARY: max(48, int(round(48 + 36 * severity))),
    }
    length = spec.duration_fixes or default_lengths[spec.anomaly_type]
    return tuple(range(length)), length


def _eligible_injection_starts(
    *,
    session_ids: np.ndarray,
    placement_span: int,
    reserved_fix_indexes: set[int],
    require_single_session: bool,
) -> list[int]:
    count = len(session_ids)
    lower_track_bound = int(np.floor(count * 0.2))
    upper_track_bound = int(np.ceil(count * 0.8))
    candidates: list[int] = []
    if not require_single_session:
        first = max(2, lower_track_bound)
        last = min(count - placement_span - 2, upper_track_bound - placement_span)
        for start in range(first, last + 1):
            reservation = range(start - 2, start + placement_span + 2)
            if not any(index in reserved_fix_indexes for index in reservation):
                candidates.append(start)
        return candidates

    for session_id in np.unique(session_ids):
        session_indexes = np.flatnonzero(session_ids == session_id)
        session_start = int(session_indexes[0])
        session_end = int(session_indexes[-1]) + 1
        first = max(session_start + 2, lower_track_bound)
        last = min(
            session_end - placement_span - 2,
            upper_track_bound - placement_span,
        )
        for start in range(first, last + 1):
            reservation = range(start - 2, start + placement_span + 2)
            if not any(index in reserved_fix_indexes for index in reservation):
                candidates.append(start)
    return candidates


def _inject_gps_spike_single(
    east_m: np.ndarray,
    north_m: np.ndarray,
    *,
    fix_index: int,
    spec: InjectionSpec,
    injection_id: str,
    rng: np.random.Generator,
    labels: list[dict[str, object]],
) -> None:
    bearing_rad = _bearing_radians(spec, rng)
    distance_m = float(spec.distance_m)
    east_m[fix_index] += distance_m * np.cos(bearing_rad)
    north_m[fix_index] += distance_m * np.sin(bearing_rad)
    labels[fix_index] = _injected_labels(
        spec,
        injection_id,
        f"single GPS fix displaced by {distance_m:.0f} m",
    )


def _apply_injection(
    east_m: np.ndarray,
    north_m: np.ndarray,
    *,
    anomaly_type: str,
    fix_indexes: tuple[int, ...],
    spec: InjectionSpec,
    injection_id: str,
    rng: np.random.Generator,
    labels: list[dict[str, object]],
) -> None:
    if anomaly_type == GPS_SPIKE_SINGLE:
        _inject_gps_spike_single(
            east_m,
            north_m,
            fix_index=fix_indexes[0],
            spec=spec,
            injection_id=injection_id,
            rng=rng,
            labels=labels,
        )
        return

    injectors = {
        GPS_SPIKE_FEW: _inject_gps_spike_few,
        GPS_SPIKE_CLUSTER: _inject_gps_spike_cluster,
        LOCATION_PING_PONG: _inject_location_ping_pong,
        LOCATION_CLUSTERS_INTERLEAVED: _inject_location_clusters_interleaved,
        TRANSPORT_FAST_LINEAR: _inject_transport_fast_linear,
        STATIONARY: _inject_stationary,
    }
    injector = injectors.get(anomaly_type)
    if injector is None:
        raise NotImplementedError(f"Unsupported synthetic anomaly type: {anomaly_type}")
    injector(
        east_m,
        north_m,
        fix_indexes=fix_indexes,
        spec=spec,
        injection_id=injection_id,
        rng=rng,
        labels=labels,
    )


def _inject_gps_spike_few(
    east_m: np.ndarray,
    north_m: np.ndarray,
    *,
    fix_indexes: tuple[int, ...],
    spec: InjectionSpec,
    injection_id: str,
    rng: np.random.Generator,
    labels: list[dict[str, object]],
) -> None:
    for ordinal, fix_index in enumerate(fix_indexes, start=1):
        bearing = _bearing_radians(spec, rng) + (
            (ordinal - 1) * 2 * np.pi / len(fix_indexes)
        )
        distance = float(spec.distance_m) * (0.85 + 0.3 * rng.random())
        east_m[fix_index] += distance * np.cos(bearing)
        north_m[fix_index] += distance * np.sin(bearing)
        labels[fix_index] = _injected_labels(
            spec,
            injection_id,
            f"isolated GPS spike {ordinal} of {len(fix_indexes)}",
        )


def _inject_gps_spike_cluster(
    east_m: np.ndarray,
    north_m: np.ndarray,
    *,
    fix_indexes: tuple[int, ...],
    spec: InjectionSpec,
    injection_id: str,
    rng: np.random.Generator,
    labels: list[dict[str, object]],
) -> None:
    reference = fix_indexes[0] - 1
    bearing = _bearing_radians(spec, rng)
    center = np.asarray([east_m[reference], north_m[reference]]) + (
        float(spec.distance_m)
        * np.asarray([np.cos(bearing), np.sin(bearing)])
    )
    jitter_m = max(5.0, 40.0 * (1.1 - float(spec.severity)))
    for fix_index in fix_indexes:
        position = center + rng.normal(0.0, jitter_m, size=2)
        east_m[fix_index], north_m[fix_index] = position
        labels[fix_index] = _injected_labels(
            spec,
            injection_id,
            "compact cluster of remote GPS fixes",
        )


def _inject_location_ping_pong(
    east_m: np.ndarray,
    north_m: np.ndarray,
    *,
    fix_indexes: tuple[int, ...],
    spec: InjectionSpec,
    injection_id: str,
    rng: np.random.Generator,
    labels: list[dict[str, object]],
) -> None:
    centers = _remote_cluster_centers(
        east_m, north_m, fix_indexes[0] - 1, spec, rng
    )
    for ordinal, fix_index in enumerate(fix_indexes):
        position = centers[ordinal % 2] + rng.normal(0.0, 5.0, size=2)
        east_m[fix_index], north_m[fix_index] = position
        labels[fix_index] = _injected_labels(
            spec,
            injection_id,
            "alternating fixes between two remote locations",
        )


def _inject_location_clusters_interleaved(
    east_m: np.ndarray,
    north_m: np.ndarray,
    *,
    fix_indexes: tuple[int, ...],
    spec: InjectionSpec,
    injection_id: str,
    rng: np.random.Generator,
    labels: list[dict[str, object]],
) -> None:
    centers = _remote_cluster_centers(
        east_m, north_m, fix_indexes[0] - 1, spec, rng
    )
    positions = [center.copy() for center in centers]
    for ordinal, fix_index in enumerate(fix_indexes):
        cluster = ordinal % 2
        positions[cluster] += rng.normal(0.0, 35.0, size=2)
        east_m[fix_index], north_m[fix_index] = positions[cluster]
        labels[fix_index] = _injected_labels(
            spec,
            injection_id,
            "interleaved fixes from two remote spatial clusters",
        )


def _inject_transport_fast_linear(
    east_m: np.ndarray,
    north_m: np.ndarray,
    *,
    fix_indexes: tuple[int, ...],
    spec: InjectionSpec,
    injection_id: str,
    rng: np.random.Generator,
    labels: list[dict[str, object]],
) -> None:
    reference = fix_indexes[0] - 1
    original_end = np.asarray([east_m[fix_indexes[-1]], north_m[fix_indexes[-1]]])
    bearing = _bearing_radians(spec, rng)
    direction = np.asarray([np.cos(bearing), np.sin(bearing)])
    start = np.asarray([east_m[reference], north_m[reference]])
    step_distance = float(spec.distance_m) / len(fix_indexes)
    for ordinal, fix_index in enumerate(fix_indexes, start=1):
        position = start + ordinal * step_distance * direction
        east_m[fix_index], north_m[fix_index] = position
        labels[fix_index] = _injected_labels(
            spec,
            injection_id,
            "fast straight transport segment",
        )
    _translate_downstream(
        east_m,
        north_m,
        last_injected_index=fix_indexes[-1],
        original_end=original_end,
    )


def _inject_stationary(
    east_m: np.ndarray,
    north_m: np.ndarray,
    *,
    fix_indexes: tuple[int, ...],
    spec: InjectionSpec,
    injection_id: str,
    rng: np.random.Generator,
    labels: list[dict[str, object]],
) -> None:
    reference = fix_indexes[0] - 1
    original_end = np.asarray([east_m[fix_indexes[-1]], north_m[fix_indexes[-1]]])
    anchor = np.asarray([east_m[reference], north_m[reference]])
    for fix_index in fix_indexes:
        east_m[fix_index], north_m[fix_index] = anchor
        labels[fix_index] = _injected_labels(
            spec,
            injection_id,
            "sustained near-stationary tag period",
        )
    _translate_downstream(
        east_m,
        north_m,
        last_injected_index=fix_indexes[-1],
        original_end=original_end,
    )


def _apply_gps_observation_model(
    east_m: np.ndarray,
    north_m: np.ndarray,
    *,
    standard_deviation_m: float,
    quantization_m: float | None,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    observed_east_m = east_m.copy()
    observed_north_m = north_m.copy()
    if standard_deviation_m != 0.0:
        observed_east_m += rng.normal(
            0.0,
            standard_deviation_m,
            size=len(observed_east_m),
        )
        observed_north_m += rng.normal(
            0.0,
            standard_deviation_m,
            size=len(observed_north_m),
        )
    if quantization_m not in (None, 0.0):
        observed_east_m = (
            np.round(observed_east_m / quantization_m) * quantization_m
        )
        observed_north_m = (
            np.round(observed_north_m / quantization_m) * quantization_m
        )
    return observed_east_m, observed_north_m


def _remote_cluster_centers(
    east_m: np.ndarray,
    north_m: np.ndarray,
    reference_index: int,
    spec: InjectionSpec,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    bearing = _bearing_radians(spec, rng)
    direction = np.asarray([np.cos(bearing), np.sin(bearing)])
    perpendicular = np.asarray([-direction[1], direction[0]])
    reference = np.asarray([east_m[reference_index], north_m[reference_index]])
    remote_center = reference + float(spec.distance_m) * direction
    separation = 0.5 * float(spec.distance_m) * perpendicular
    return remote_center - separation, remote_center + separation


def _translate_downstream(
    east_m: np.ndarray,
    north_m: np.ndarray,
    *,
    last_injected_index: int,
    original_end: np.ndarray,
) -> None:
    new_end = np.asarray(
        [east_m[last_injected_index], north_m[last_injected_index]]
    )
    translation = new_end - original_end
    east_m[last_injected_index + 1 :] += translation[0]
    north_m[last_injected_index + 1 :] += translation[1]


def _bearing_radians(
    spec: InjectionSpec,
    rng: np.random.Generator,
) -> float:
    if spec.bearing_degrees is None:
        return float(rng.uniform(-np.pi, np.pi))
    return float(np.deg2rad(spec.bearing_degrees))


def _injected_labels(
    spec: InjectionSpec,
    injection_id: str,
    default_notes: str,
) -> dict[str, object]:
    return {
        "visible": False,
        "algorithm-marked-outlier": False,
        "manually-marked-outlier": True,
        "synthetic:is_anomaly": True,
        "synthetic:anomaly_type": spec.anomaly_type,
        "synthetic:injection_id": injection_id,
        "synthetic:severity": float(spec.severity),
        "synthetic:notes": str(spec.notes).strip() or default_notes,
    }


def _normal_labels() -> dict[str, object]:
    return {
        "visible": True,
        "algorithm-marked-outlier": False,
        "manually-marked-outlier": False,
        "synthetic:is_anomaly": False,
        "synthetic:anomaly_type": "normal",
        "synthetic:injection_id": "none",
        "synthetic:severity": 0.0,
        "synthetic:notes": "baseline",
    }


def _local_meters_to_lon_lat(
    east_m: Sequence[float],
    north_m: Sequence[float],
    *,
    anchor_lon: float,
    anchor_lat: float,
) -> tuple[np.ndarray, np.ndarray]:
    east = np.asarray(east_m, dtype=np.float64)
    north = np.asarray(north_m, dtype=np.float64)
    anchor_lat_rad = np.deg2rad(anchor_lat)
    latitudes = anchor_lat + np.rad2deg(north / EARTH_RADIUS_M)
    longitudes = anchor_lon + np.rad2deg(
        east / (EARTH_RADIUS_M * np.cos(anchor_lat_rad))
    )
    return longitudes, latitudes


def _validate_coordinates(
    longitudes: np.ndarray,
    latitudes: np.ndarray,
) -> None:
    if not np.isfinite(longitudes).all() or not np.isfinite(latitudes).all():
        raise ValueError("Simulation produced non-finite coordinates")
    if (longitudes < -180.0).any() or (longitudes > 180.0).any():
        raise ValueError("Simulation produced longitude outside [-180, 180]")
    if (latitudes < -90.0).any() or (latitudes > 90.0).any():
        raise ValueError("Simulation produced latitude outside [-90, 90]")


def _format_movebank_timestamp(timestamp: pd.Timestamp) -> str:
    return timestamp.strftime("%Y-%m-%d %H:%M:%S.000")


def _format_severity(value: object) -> str:
    number = float(value)
    if number == 0.0:
        return "0.0"
    return format(number, ".6g")
