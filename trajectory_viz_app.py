"""MoveBench trajectory viewer (Streamlit).

Run:
  pip install -e .[apps]
  streamlit run trajectory_viz_app.py

Optional:
  MOVEBENCH_DATA_ROOT=/path/to/data/movebench streamlit run trajectory_viz_app.py
"""

from __future__ import annotations

import colorsys
import os
from pathlib import Path


# If the user hasn't configured MOVEBENCH_DATA_ROOT, default to the repo-local
# data path so Study.list_available() works out-of-the-box for this checkout.
_default_data_root = (Path(__file__).resolve().parent / "data" / "movebench").resolve()
if "MOVEBENCH_DATA_ROOT" not in os.environ and _default_data_root.exists():
    os.environ["MOVEBENCH_DATA_ROOT"] = str(_default_data_root)

import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st

from movebench import Study, paths
from movebench.trajectories import INDIVIDUAL_COL, LAT_COL, LON_COL, SET_COL, TIME_COL


MAP_STYLE = "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json"
MAX_POINTS_PER_PATH = 2_000

TRAIN_RGB = [31, 119, 180]  # blue
TEST_RGB = [255, 127, 14]  # orange
PATH_ALPHA = 180


def _uniform_downsample(path: list[list[float]], max_points: int) -> list[list[float]]:
    n = len(path)
    if n <= max_points:
        return path
    idx = np.linspace(0, n - 1, num=max_points, dtype=np.int64)
    return [path[int(i)] for i in idx]


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _make_shade_palette(base_rgb: list[int], keys: list[str], alpha: int) -> dict[str, list[int]]:
    if not keys:
        return {}

    r, g, b = (float(c) / 255.0 for c in base_rgb)
    h, l, s = colorsys.rgb_to_hls(r, g, b)

    l_min = _clamp(l - 0.12, 0.15, 0.85)
    l_max = _clamp(l + 0.12, 0.15, 0.85)

    n = len(keys)
    if n == 1:
        lightness_values = [l]
    else:
        lightness_values = np.linspace(l_min, l_max, num=n, dtype=np.float64).tolist()

    out: dict[str, list[int]] = {}
    for key, li in zip(keys, lightness_values, strict=True):
        rr, gg, bb = colorsys.hls_to_rgb(h, float(li), s)
        out[key] = [int(round(rr * 255)), int(round(gg * 255)), int(round(bb * 255)), int(alpha)]
    return out


def _span_to_zoom(span_deg: float) -> float:
    # Deterministic, dependency-free heuristic.
    if not np.isfinite(span_deg) or span_deg <= 0:
        return 2.0
    if span_deg > 120:
        return 1.0
    if span_deg > 60:
        return 2.0
    if span_deg > 30:
        return 3.0
    if span_deg > 15:
        return 4.0
    if span_deg > 8:
        return 5.0
    if span_deg > 4:
        return 6.0
    if span_deg > 2:
        return 7.0
    if span_deg > 1:
        return 8.0
    if span_deg > 0.5:
        return 9.0
    if span_deg > 0.25:
        return 10.0
    return 11.0


@st.cache_data(show_spinner=False)
def list_study_ids() -> list[str]:
    return Study.list_available()


@st.cache_data(show_spinner=False)
def load_study_df(study_id: str) -> pd.DataFrame:
    s = Study(study_id)
    df = pd.read_csv(s.trajectories_path)

    # Parse + clean.
    df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="coerce")
    df = df.dropna(subset=[TIME_COL, LON_COL, LAT_COL])

    # Keep only what we need.
    keep = [INDIVIDUAL_COL, TIME_COL, LON_COL, LAT_COL, SET_COL]
    df = df[[c for c in keep if c in df.columns]].copy()
    df = df.sort_values([INDIVIDUAL_COL, TIME_COL], kind="mergesort")
    return df


@st.cache_data(show_spinner=False)
def build_path_rows(df: pd.DataFrame) -> list[dict]:
    individuals = sorted(df[INDIVIDUAL_COL].dropna().astype(str).unique().tolist())
    train_palette = _make_shade_palette(TRAIN_RGB, individuals, PATH_ALPHA)
    test_palette = _make_shade_palette(TEST_RGB, individuals, PATH_ALPHA)

    rows: list[dict] = []
    for split, palette, fallback_rgb in (
        ("train", train_palette, TRAIN_RGB),
        ("test", test_palette, TEST_RGB),
    ):
        fallback_color = [int(fallback_rgb[0]), int(fallback_rgb[1]), int(fallback_rgb[2]), int(PATH_ALPHA)]
        sdf = df[df[SET_COL] == split]
        if len(sdf) == 0:
            continue
        for indiv, g in sdf.groupby(INDIVIDUAL_COL, sort=False):
            indiv_str = str(indiv)
            coords = g[[LON_COL, LAT_COL]].to_numpy(dtype=np.float64, copy=False)
            finite = np.isfinite(coords).all(axis=1)
            coords = coords[finite]
            if len(coords) < 2:
                continue
            path = coords.tolist()
            path = _uniform_downsample(path, MAX_POINTS_PER_PATH)
            rows.append(
                {"individual": indiv_str, "set": split, "path": path, "color": palette.get(indiv_str, fallback_color)}
            )
    return rows


def compute_initial_view(df: pd.DataFrame) -> pdk.ViewState:
    if len(df) == 0:
        return pdk.ViewState(latitude=0.0, longitude=0.0, zoom=1.0)

    lons = df[LON_COL].to_numpy(dtype=np.float64, copy=False)
    lats = df[LAT_COL].to_numpy(dtype=np.float64, copy=False)
    finite = np.isfinite(lons) & np.isfinite(lats)
    if not finite.any():
        return pdk.ViewState(latitude=0.0, longitude=0.0, zoom=1.0)

    lons = lons[finite]
    lats = lats[finite]

    min_lon = float(np.min(lons))
    max_lon = float(np.max(lons))
    min_lat = float(np.min(lats))
    max_lat = float(np.max(lats))

    center_lon = 0.5 * (min_lon + max_lon)
    center_lat = 0.5 * (min_lat + max_lat)

    span = max(max_lon - min_lon, max_lat - min_lat)
    zoom = _span_to_zoom(float(span))

    return pdk.ViewState(latitude=center_lat, longitude=center_lon, zoom=zoom)


def main() -> None:
    st.set_page_config(page_title="MoveBench Trajectory Viewer", layout="wide")

    study_ids = list_study_ids()
    if not study_ids:
        st.error(
            "No studies found. "
            f"Expected study directories under: {paths.studies_root()}. "
            "Set MOVEBENCH_DATA_ROOT to your data root (the directory containing studies_10ind/)."
        )
        st.stop()

    study_id = st.selectbox("Study", study_ids)
    df = load_study_df(study_id)

    view_state_study_id = st.session_state.get("_view_state_study_id")
    if view_state_study_id != study_id or "_initial_view_state" not in st.session_state:
        st.session_state["_view_state_study_id"] = study_id
        st.session_state["_initial_view_state"] = compute_initial_view(df)
    view_state = st.session_state["_initial_view_state"]

    individuals = sorted(df[INDIVIDUAL_COL].dropna().astype(str).unique().tolist())
    if not individuals:
        st.error("No individuals found in trajectories.csv.")
        st.stop()

    last_study_id = st.session_state.get("_last_study_id")
    if last_study_id != study_id:
        st.session_state["_last_study_id"] = study_id
        st.session_state["visible_individuals"] = individuals

    visible_individuals = st.session_state.get("visible_individuals", individuals)
    if not isinstance(visible_individuals, list):
        visible_individuals = individuals
    visible_individuals = [v for v in visible_individuals if v in individuals]
    st.session_state["visible_individuals"] = visible_individuals

    if "show_train" not in st.session_state:
        st.session_state["show_train"] = True
    if "show_test" not in st.session_state:
        st.session_state["show_test"] = True

    with st.expander("Individuals", expanded=True):
        st.markdown("**Splits**")
        s1, s2 = st.columns(2)
        s1.checkbox("Show train", key="show_train")
        s2.checkbox("Show test", key="show_test")

        st.markdown("**Individuals**")
        c1, c2 = st.columns(2)
        if c1.button("Select all", use_container_width=True):
            st.session_state["visible_individuals"] = individuals
        if c2.button("Clear", use_container_width=True):
            st.session_state["visible_individuals"] = []

        st.multiselect("Visible individuals", options=individuals, key="visible_individuals")

    selected = set(st.session_state.get("visible_individuals", []))
    if not selected:
        st.warning("No individuals selected.")

    enabled_splits: list[str] = []
    if bool(st.session_state.get("show_train", True)):
        enabled_splits.append("train")
    if bool(st.session_state.get("show_test", True)):
        enabled_splits.append("test")

    if not enabled_splits:
        st.warning("No splits selected.")
    else:
        missing_splits = [s for s in enabled_splits if (df[SET_COL] == s).sum() == 0]
        if missing_splits:
            st.warning(f"No data for enabled split(s): {', '.join(missing_splits)} (rendering what exists).")

    all_rows = build_path_rows(df)
    if selected and enabled_splits:
        rows = [r for r in all_rows if r.get("individual") in selected and r.get("set") in enabled_splits]
    else:
        rows = []

    layer = pdk.Layer(
        "PathLayer",
        rows,
        get_path="path",
        get_color="color",
        pickable=True,
        width_min_pixels=2,
    )

    deck = pdk.Deck(
        map_style=MAP_STYLE,
        initial_view_state=view_state,
        layers=[layer],
        tooltip={"text": "{set} | {individual}"},
    )

    try:
        st.pydeck_chart(deck, use_container_width=True, key="trajectory_map")
    except TypeError:
        st.pydeck_chart(deck, use_container_width=True)


if __name__ == "__main__":
    main()
