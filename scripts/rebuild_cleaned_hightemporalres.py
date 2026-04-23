from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


INDIVIDUAL_COL = "individual-local-identifier"
TIME_COL = "timestamp"
EVENT_COL = "event-id"
SET_COL = "set"
LOGICAL_COL = "__logical_name"

CLEAN_ROOT = Path(
    "/data/vision/beery/scratch/shir/codes/vibecleaning/data/movement_example/cleaned_topout_v1"
)
RAW_ROOT = Path("/data/vision/beery/scratch/shir/codes/vibecleaning/data/movement_example")
OUT_ROOT = Path(
    "/data/vision/beery/scratch/shir/codes/vibecleaning/data/movement_example/cleaned_hightemporalres"
)


def _resolve_raw_sources(clean_path: Path, clean_df: pd.DataFrame) -> list[Path]:
    if LOGICAL_COL in clean_df.columns and clean_df[LOGICAL_COL].notna().any():
        raw_paths = [RAW_ROOT / str(name) for name in sorted(clean_df[LOGICAL_COL].dropna().astype(str).unique())]
        missing = [str(path) for path in raw_paths if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Missing logical-name raw sources for {clean_path.name}: {missing}")
        return raw_paths

    exact = RAW_ROOT / clean_path.name
    if exact.exists():
        return [exact]

    match = re.match(r"^(.*?_raw(?:_merged)?)(?:_.+)?\.csv$", clean_path.name)
    if match:
        candidate = RAW_ROOT / f"{match.group(1)}.csv"
        if candidate.exists():
            return [candidate]

    raise FileNotFoundError(f"Could not resolve raw source for {clean_path.name}")


def _load_clean(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="coerce")
    df["_indiv_str"] = df[INDIVIDUAL_COL].astype(str)
    return df


def _load_raw(paths: list[Path]) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for path in paths:
        df = pd.read_csv(path, low_memory=False)
        df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="coerce")
        df["_source_file"] = path.name
        df["_indiv_str"] = df[INDIVIDUAL_COL].astype(str)
        if EVENT_COL in df.columns:
            df["_event_str"] = df[EVENT_COL].astype(str)
        parts.append(df)
    return pd.concat(parts, ignore_index=True, sort=False)


def _boundary_frame(clean_df: pd.DataFrame) -> pd.DataFrame:
    train_max = (
        clean_df.loc[clean_df[SET_COL] == "train", ["_indiv_str", TIME_COL]]
        .groupby("_indiv_str", sort=False)[TIME_COL]
        .max()
        .rename("max_train")
    )
    test_min = (
        clean_df.loc[clean_df[SET_COL] == "test", ["_indiv_str", TIME_COL]]
        .groupby("_indiv_str", sort=False)[TIME_COL]
        .min()
        .rename("min_test")
    )
    bounds = pd.concat([train_max, test_min], axis=1).reset_index()
    bounds["midpoint"] = bounds["max_train"] + (bounds["min_test"] - bounds["max_train"]) / 2
    return bounds


def _exact_split_frame(clean_df: pd.DataFrame) -> pd.DataFrame:
    if EVENT_COL not in clean_df.columns:
        return pd.DataFrame(columns=["_indiv_str", "_event_str", "set_exact"])
    exact = clean_df[["_indiv_str", EVENT_COL, SET_COL]].dropna(subset=[EVENT_COL, SET_COL]).copy()
    exact["_event_str"] = exact[EVENT_COL].astype(str)
    exact = exact.rename(columns={SET_COL: "set_exact"})
    return exact[["_indiv_str", "_event_str", "set_exact"]].drop_duplicates()


def rebuild_one(clean_path: Path) -> dict:
    clean_df = _load_clean(clean_path)
    raw_paths = _resolve_raw_sources(clean_path, clean_df)
    raw_df = _load_raw(raw_paths)

    selected_individuals = set(clean_df["_indiv_str"].dropna().unique())
    raw_df = raw_df[raw_df["_indiv_str"].isin(selected_individuals)].copy()
    if len(raw_df) == 0:
        raise ValueError(f"No raw rows remained after filtering individuals for {clean_path.name}")

    exact = _exact_split_frame(clean_df)
    if len(exact) and "_event_str" in raw_df.columns:
        raw_df = raw_df.merge(exact, on=["_indiv_str", "_event_str"], how="left")
    else:
        raw_df["set_exact"] = pd.NA

    bounds = _boundary_frame(clean_df)
    raw_df = raw_df.merge(bounds, on="_indiv_str", how="left")
    raw_df[SET_COL] = raw_df["set_exact"]

    unresolved = raw_df[SET_COL].isna() & raw_df[TIME_COL].notna()
    has_both = unresolved & raw_df["midpoint"].notna()
    raw_df.loc[has_both & (raw_df[TIME_COL] <= raw_df["midpoint"]), SET_COL] = "train"
    raw_df.loc[has_both & (raw_df[TIME_COL] > raw_df["midpoint"]), SET_COL] = "test"

    unresolved = raw_df[SET_COL].isna() & raw_df[TIME_COL].notna()
    train_only = unresolved & raw_df["max_train"].notna() & raw_df["min_test"].isna()
    test_only = unresolved & raw_df["min_test"].notna() & raw_df["max_train"].isna()
    raw_df.loc[train_only, SET_COL] = "train"
    raw_df.loc[test_only, SET_COL] = "test"

    unresolved = raw_df[SET_COL].isna()
    if unresolved.any():
        unresolved_df = raw_df.loc[
            unresolved,
            [INDIVIDUAL_COL, TIME_COL] + ([EVENT_COL] if EVENT_COL in raw_df.columns else []),
        ]
        raise ValueError(
            f"Unresolved split assignments for {clean_path.name}: {len(unresolved_df)} rows"
        )

    midpoint_assigned = int(raw_df["set_exact"].isna().sum())

    if LOGICAL_COL in clean_df.columns:
        raw_df[LOGICAL_COL] = raw_df["_source_file"]

    clean_cols = list(clean_df.columns)
    helper_cols = {"_source_file", "_indiv_str", "_event_str", "set_exact", "max_train", "min_test", "midpoint"}
    ordered_cols: list[str] = []
    for col in clean_cols:
        if col in helper_cols:
            continue
        if col in raw_df.columns and col not in ordered_cols:
            ordered_cols.append(col)
    for col in raw_df.columns:
        if col in helper_cols:
            continue
        if col not in ordered_cols:
            ordered_cols.append(col)

    out_df = raw_df.sort_values([INDIVIDUAL_COL, TIME_COL], kind="mergesort").reset_index(drop=True)
    out_path = OUT_ROOT / clean_path.name
    out_df[ordered_cols].to_csv(out_path, index=False)

    return {
        "file": clean_path.name,
        "raw_sources": ",".join(path.name for path in raw_paths),
        "clean_rows": int(len(clean_df)),
        "output_rows": int(len(out_df)),
        "individuals": int(out_df["_indiv_str"].nunique()),
        "train_rows": int((out_df[SET_COL] == "train").sum()),
        "test_rows": int((out_df[SET_COL] == "test").sum()),
        "midpoint_assigned_rows": midpoint_assigned,
    }


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict] = []
    for clean_path in sorted(CLEAN_ROOT.glob("*.csv")):
        print(f"Rebuilding {clean_path.name}...")
        summary_rows.append(rebuild_one(clean_path))

    summary = pd.DataFrame(summary_rows).sort_values("file", kind="mergesort")
    summary_path = OUT_ROOT / "_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"\nWrote {len(summary)} files to {OUT_ROOT}")
    print(summary.to_string(index=False))
    print(f"\nSummary CSV: {summary_path}")


if __name__ == "__main__":
    main()
