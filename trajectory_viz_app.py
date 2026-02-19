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

PATH_ALPHA = 180

BASE_SATURATION = 0.9
TRAIN_LIGHTNESS = 0.65  # lighter shade
TEST_LIGHTNESS = 0.35  # darker shade


def _uniform_downsample(path: list[list[float]], max_points: int) -> list[list[float]]:
    n = len(path)
    if n <= max_points:
        return path
    idx = np.linspace(0, n - 1, num=max_points, dtype=np.int64)
    return [path[int(i)] for i in idx]


def _alternating_hue_order(n: int) -> list[int]:
    """Return indices 0..n-1 in an order that maximizes adjacent hue separation."""
    half = (n + 1) // 2
    out: list[int] = []
    for i in range(half):
        out.append(i)
        j = i + half
        if j < n:
            out.append(j)
    return out[:n]


def _hls_to_rgba(h: float, l: float, s: float, alpha: int) -> list[int]:
    r, g, b = colorsys.hls_to_rgb(float(h), float(l), float(s))
    return [
        int(round(r * 255)),
        int(round(g * 255)),
        int(round(b * 255)),
        int(alpha),
    ]


def _make_split_palettes(individuals: list[str], alpha: int) -> tuple[dict[str, list[int]], dict[str, list[int]]]:
    n = len(individuals)
    if n == 0:
        return {}, {}

    order = _alternating_hue_order(n)
    train: dict[str, list[int]] = {}
    test: dict[str, list[int]] = {}
    for idx, indiv in enumerate(individuals):
        h = float(order[idx]) / float(n)
        train[indiv] = _hls_to_rgba(h, TRAIN_LIGHTNESS, BASE_SATURATION, alpha)
        test[indiv] = _hls_to_rgba(h, TEST_LIGHTNESS, BASE_SATURATION, alpha)
    return train, test


def _color_dot_html(rgba: list[int], *, enabled: bool) -> str:
    r, g, b, a = rgba
    alpha = float(a) / 255.0
    if not enabled:
        alpha *= 0.25
    border_alpha = 0.35 if enabled else 0.15
    return (
        "<span "
        "style="
        "\"display:inline-block;"
        "width:12px;"
        "height:12px;"
        "border-radius:999px;"
        f"background:rgba({int(r)},{int(g)},{int(b)},{alpha:.3f});"
        f"border:1px solid rgba(0,0,0,{border_alpha:.3f});"
        "\""
        "></span>"
    )


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
    train_palette, test_palette = _make_split_palettes(individuals, PATH_ALPHA)

    rows: list[dict] = []
    for split, palette in (("train", train_palette), ("test", test_palette)):
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
            color = palette.get(indiv_str)
            if color is None:
                # Shouldn't happen, but keep the app robust to unexpected ids.
                fallback_l = TRAIN_LIGHTNESS if split == "train" else TEST_LIGHTNESS
                color = _hls_to_rgba(0.0, fallback_l, BASE_SATURATION, PATH_ALPHA)
            rows.append({"individual": indiv_str, "set": split, "path": path, "color": color})
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
        st.sidebar.error(
            "No studies found. "
            f"Expected study directories under: {paths.studies_root()}. "
            "Set MOVEBENCH_DATA_ROOT to your data root (the directory containing studies_10ind/)."
        )
        st.stop()

    study_id = st.sidebar.selectbox("Study", study_ids, key="study_id")
    df = load_study_df(study_id)

    view_state_study_id = st.session_state.get("_view_state_study_id")
    if view_state_study_id != study_id or "_initial_view_state" not in st.session_state:
        st.session_state["_view_state_study_id"] = study_id
        st.session_state["_initial_view_state"] = compute_initial_view(df)
    view_state = st.session_state["_initial_view_state"]

    individuals = sorted(df[INDIVIDUAL_COL].dropna().astype(str).unique().tolist())
    if not individuals:
        st.sidebar.error("No individuals found in trajectories.csv.")
        st.stop()

    if "show_train" not in st.session_state:
        st.session_state["show_train"] = True
    if "show_test" not in st.session_state:
        st.session_state["show_test"] = True

    train_palette, test_palette = _make_split_palettes(individuals, PATH_ALPHA)

    def _indiv_visible_key(indiv: str) -> str:
        return f"visible::{study_id}::{indiv}"

    last_study_id = st.session_state.get("_last_study_id")
    if last_study_id != study_id:
        st.session_state["_last_study_id"] = study_id
        for indiv in individuals:
            st.session_state[_indiv_visible_key(indiv)] = True

    # ---- table header ----
    h1, h2, h3 = st.sidebar.columns([3, 1, 1])
    h1.markdown("**Individual**")
    h2.checkbox("Train", key="show_train")
    h3.checkbox("Test", key="show_test")

    # ---- table rows ----
    for indiv in individuals:
        k = _indiv_visible_key(indiv)
        if k not in st.session_state:
            st.session_state[k] = True

        c1, c2, c3 = st.sidebar.columns([3, 1, 1])
        visible = c1.checkbox(indiv, key=k)

        train_color = train_palette[indiv]
        test_color = test_palette[indiv]
        c2.markdown(
            _color_dot_html(train_color, enabled=visible and bool(st.session_state.get("show_train", True))),
            unsafe_allow_html=True,
        )
        c3.markdown(
            _color_dot_html(test_color, enabled=visible and bool(st.session_state.get("show_test", True))),
            unsafe_allow_html=True,
        )

    sidebar_messages = st.sidebar.container()
    selected = {indiv for indiv in individuals if bool(st.session_state.get(_indiv_visible_key(indiv), False))}
    if not selected:
        sidebar_messages.warning("No individuals selected.")

    enabled_splits: list[str] = []
    if bool(st.session_state.get("show_train", True)):
        enabled_splits.append("train")
    if bool(st.session_state.get("show_test", True)):
        enabled_splits.append("test")

    if not enabled_splits:
        sidebar_messages.warning("No splits selected.")
    else:
        missing_splits = [s for s in enabled_splits if (df[SET_COL] == s).sum() == 0]
        if missing_splits:
            sidebar_messages.warning(f"No data for enabled split(s): {', '.join(missing_splits)} (rendering what exists).")

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
