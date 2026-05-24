from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from movebench.thumbnail_config import (
    build_thumbnail_generation_config,
    thumbnail_build_kwargs,
)
from movebench.trajectories import INDIVIDUAL_COL, LAT_COL, LON_COL, SET_COL, TIME_COL


def _thumbnail_functions():
    pytest.importorskip("matplotlib")
    pytest.importorskip("sklearn")
    from movebench.datasets.thumbnails import assign_time_gap_bursts, build_track_thumbnails

    return assign_time_gap_bursts, build_track_thumbnails


def _write_thumbnail_csv(path: Path) -> Path:
    rows = [
        ("a1", "train", "2024-01-01T00:00:00Z", -70.0, 40.0, 0),
        ("a1", "train", "2024-01-01T01:00:00Z", -70.1, 40.1, 0),
        ("a1", "train", "2024-01-01T02:00:01Z", -70.2, 40.2, 1),
        ("a1", "train", "2024-01-01T02:30:00Z", -70.3, 40.3, 0),
        ("a1", "test", "2024-01-02T00:00:00Z", -71.0, 41.0, 0),
        ("a1", "test", "2024-01-02T02:00:01Z", -71.1, 41.1, 0),
        ("b1", "train", "2024-01-01T00:00:00Z", -72.0, 42.0, 0),
        ("b1", "train", "2024-01-01T00:20:00Z", -72.1, 42.1, 0),
    ]
    pd.DataFrame(
        rows,
        columns=[
            INDIVIDUAL_COL,
            SET_COL,
            TIME_COL,
            LON_COL,
            LAT_COL,
            "manually-marked-outlier",
        ],
    ).to_csv(path, index=False)
    return path


def test_assign_time_gap_bursts_uses_strict_greater_than_threshold():
    assign_time_gap_bursts, _build_track_thumbnails = _thumbnail_functions()
    df = pd.DataFrame(
        {
            INDIVIDUAL_COL: ["a", "a", "a", "a"],
            SET_COL: ["train", "train", "train", "train"],
            TIME_COL: pd.to_datetime(
                [
                    "2024-01-01T00:00:00Z",
                    "2024-01-01T01:00:00Z",
                    "2024-01-01T02:00:01Z",
                    "2024-01-01T03:00:01Z",
                ]
            ),
        }
    )

    out = assign_time_gap_bursts(df, burst_gap_seconds=3600)

    assert out["burst_idx"].tolist() == [0, 0, 1, 1]


def test_assign_time_gap_bursts_resets_by_individual_and_split():
    assign_time_gap_bursts, _build_track_thumbnails = _thumbnail_functions()
    df = pd.DataFrame(
        {
            INDIVIDUAL_COL: ["a", "a", "a", "a", "b", "b"],
            SET_COL: ["train", "train", "test", "test", "train", "train"],
            TIME_COL: pd.to_datetime(
                [
                    "2024-01-01T00:00:00Z",
                    "2024-01-01T02:00:00Z",
                    "2024-01-01T00:00:00Z",
                    "2024-01-01T02:00:00Z",
                    "2024-01-01T00:00:00Z",
                    "2024-01-01T02:00:00Z",
                ]
            ),
        }
    )

    out = assign_time_gap_bursts(df, burst_gap_seconds=3600)

    assert out.groupby([INDIVIDUAL_COL, SET_COL])["burst_idx"].apply(list).to_dict() == {
        ("a", "test"): [0, 1],
        ("a", "train"): [0, 1],
        ("b", "train"): [0, 1],
    }


def test_build_track_thumbnails_fixed_mode_is_default(tmp_path):
    _assign_time_gap_bursts, build_track_thumbnails = _thumbnail_functions()
    csv_path = _write_thumbnail_csv(tmp_path / "tracks.csv")

    build = build_track_thumbnails(
        csv_path,
        split="train",
        chunk_size=2,
        min_chunk_size=1,
        scale_mode="chunk",
        basemap="none",
    )

    assert build.metadata["chunk_mode"].tolist() == ["fixed", "fixed", "fixed"]
    assert build.metadata["n_fixes"].tolist() == [2, 2, 2]
    assert build.metadata["burst_id"].tolist() == ["", "", ""]


def test_build_track_thumbnails_burst_mode_uses_gap_and_min_size(tmp_path):
    _assign_time_gap_bursts, build_track_thumbnails = _thumbnail_functions()
    csv_path = _write_thumbnail_csv(tmp_path / "tracks.csv")

    build = build_track_thumbnails(
        csv_path,
        split="train",
        chunk_mode="burst",
        burst_gap_seconds=3600,
        min_chunk_size=2,
        scale_mode="chunk",
        basemap="none",
    )
    meta = build.metadata.sort_values(["individual_id", "start_time"]).reset_index(drop=True)

    assert meta["individual_id"].tolist() == ["a1", "a1", "b1"]
    assert meta["chunk_mode"].tolist() == ["burst", "burst", "burst"]
    assert meta["burst_idx"].tolist() == [0, 1, 0]
    assert meta["n_fixes"].tolist() == [2, 2, 2]
    assert meta["burst_gap_seconds"].tolist() == [3600.0, 3600.0, 3600.0]
    assert all(value.startswith(f"{row.individual_id}:{row.split}:burst_") for row, value in zip(meta.itertuples(), meta["burst_id"]))


def test_build_track_thumbnails_quantile_burst_gap_uses_whole_source(tmp_path):
    _assign_time_gap_bursts, build_track_thumbnails = _thumbnail_functions()
    csv_path = _write_thumbnail_csv(tmp_path / "tracks.csv")

    build = build_track_thumbnails(
        csv_path,
        split="train",
        chunk_mode="burst",
        burst_gap_mode="quantile",
        burst_gap_seconds=3600,
        burst_gap_quantile=1.0,
        min_chunk_size=2,
        scale_mode="chunk",
        basemap="none",
    )
    meta = build.metadata.sort_values(["individual_id", "start_time"]).reset_index(drop=True)

    assert build.burst_gap["mode"] == "quantile"
    assert build.burst_gap["gap_count"] == 5
    assert build.burst_gap["effective_seconds"] == 7201.0
    assert meta["individual_id"].tolist() == ["a1", "b1"]
    assert meta["n_fixes"].tolist() == [4, 2]
    assert meta["burst_gap_seconds"].tolist() == [7201.0, 7201.0]
    assert meta["burst_gap_mode"].tolist() == ["quantile", "quantile"]


def test_thumbnail_config_round_trips_burst_settings(tmp_path):
    csv_path = _write_thumbnail_csv(tmp_path / "tracks.csv")

    config = build_thumbnail_generation_config(
        csv_path=csv_path,
        split="train",
        params={
            "img_size": 64,
            "chunk_mode": "burst",
            "chunk_size": 50,
            "stride": None,
            "min_chunk_size": 2,
            "burst_gap_mode": "quantile",
            "burst_gap_seconds": 1800,
            "burst_gap_quantile": 0.95,
            "line_width": 1,
            "marker_radius": 1,
            "padding": 2,
            "scale_mode": "chunk",
            "scale_quantile": 0.5,
        },
        color_by=None,
        basemap="none",
    )

    kwargs = thumbnail_build_kwargs(config)

    assert config["thumbnail"]["chunk_mode"] == "burst"
    assert config["thumbnail"]["burst_gap_mode"] == "quantile"
    assert config["thumbnail"]["burst_gap_seconds"] == 1800.0
    assert config["thumbnail"]["burst_gap_quantile"] == 0.95
    assert kwargs["chunk_mode"] == "burst"
    assert kwargs["burst_gap_mode"] == "quantile"
    assert kwargs["burst_gap_seconds"] == 1800.0
    assert kwargs["burst_gap_quantile"] == 0.95


def test_thumbnail_config_legacy_burst_settings_default_to_manual(tmp_path):
    csv_path = _write_thumbnail_csv(tmp_path / "tracks.csv")

    kwargs = thumbnail_build_kwargs(
        {
            "data": {"csv_path": str(csv_path), "split": "train"},
            "thumbnail": {
                "img_size": 64,
                "chunk_mode": "burst",
                "chunk_size": 50,
                "stride": None,
                "min_chunk_size": 2,
                "burst_gap_seconds": 1800,
                "line_width": 1,
                "marker_radius": 1,
                "padding": 2,
                "scale_mode": "chunk",
                "scale_quantile": 0.5,
                "color_by": None,
                "basemap": "none",
            },
        }
    )

    assert kwargs["burst_gap_mode"] == "manual"
    assert kwargs["burst_gap_seconds"] == 1800.0
    assert kwargs["burst_gap_quantile"] == 0.999


def test_thumbnail_embedding_sources_recurse_nested_csvs(tmp_path):
    pytest.importorskip("sklearn")
    from movebench.thumbnail_embedding import list_embedding_sources

    nested = tmp_path / "nested"
    nested.mkdir()
    csv_path = _write_thumbnail_csv(nested / "tracks.csv")

    list_embedding_sources.cache_clear()
    try:
        sources = list_embedding_sources(tmp_path)
    finally:
        list_embedding_sources.cache_clear()

    assert [source["path"] for source in sources] == [str(csv_path.resolve())]
    assert sources[0]["default_split"] == "train"


def test_thumbnail_embedding_explorer_exposes_burst_and_basemap_controls():
    source = Path("thumbnail_embedding_explorer.py").read_text(encoding="utf-8")

    assert 'id="chunk-mode"' in source
    assert 'id="burst-gap-seconds"' in source
    assert 'id="thumbnail-basemap"' in source
    assert '"basemap_options"' in source
    assert "chunk_mode: document.getElementById(\"chunk-mode\").value" in source
    assert "burst_gap_seconds: Number(document.getElementById(\"burst-gap-seconds\").value)" in source
    assert "basemap: document.getElementById(\"thumbnail-basemap\").value" in source


def test_thumbnail_explorer_exposes_quantile_burst_gap_controls():
    source = Path("thumbnail_explorer_app.py").read_text(encoding="utf-8")

    assert '"burst_gap_mode": "quantile"' in source
    assert "draft_burst_gap_mode" in source
    assert "draft_burst_gap_quantile" in source
    assert "format_burst_gap_summary" in source
    assert "burst_gap_mode=str(params.get(\"burst_gap_mode\", \"manual\"))" in source
