"""Ingest Movebank-formatted CSVs into movebench as new studies.

Migrated from `scripts/ingest.py` so the original script can be deleted.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import uuid
from collections import Counter
from pathlib import Path

import pandas as pd


def _parse_int_with_underscores(value: str) -> int:
    s = str(value).strip()
    if not s:
        raise argparse.ArgumentTypeError("Expected an integer")
    try:
        return int(s.replace("_", ""))
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"Invalid integer: {value!r}") from e


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Movebank often uses hyphenated column names; allow underscore variants.
    return df.rename(columns=lambda c: c.replace("_", "-"))


def _ensure_required_columns(df: pd.DataFrame) -> pd.DataFrame:
    from movebench.trajectories import (
        INDIVIDUAL_COL,
        LAT_COL,
        LON_COL,
        TIME_COL,
    )

    df = df.copy()

    # Fill individual-local-identifier from individual-id if needed.
    if INDIVIDUAL_COL not in df.columns or df[INDIVIDUAL_COL].isna().all():
        if "individual-id" in df.columns:
            df[INDIVIDUAL_COL] = df["individual-id"]
        else:
            raise ValueError(
                "Missing required individual column. Expected either "
                f"{INDIVIDUAL_COL!r} or 'individual-id'."
            )

    required = [TIME_COL, LAT_COL, LON_COL, INDIVIDUAL_COL]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required column(s): {missing}")

    return df


def _parse_and_filter(df: pd.DataFrame) -> pd.DataFrame:
    from movebench.trajectories import LAT_COL, LON_COL, TIME_COL

    df = df.copy()
    df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="coerce")
    df = df.dropna(subset=[TIME_COL, LAT_COL, LON_COL])
    return df


def _subsample_individuals(
    df: pd.DataFrame,
    *,
    max_individuals: int,
    max_points: int,
    random_seed: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    from movebench.trajectories import INDIVIDUAL_COL

    if max_individuals <= 0:
        raise ValueError(f"max_individuals must be > 0, got {max_individuals}")
    if max_points <= 0:
        raise ValueError(f"max_points must be > 0, got {max_points}")

    df = df.copy()
    df = df.dropna(subset=[INDIVIDUAL_COL])

    n_points_before = int(len(df))
    n_individuals_before = int(df[INDIVIDUAL_COL].nunique())
    if n_points_before == 0 or n_individuals_before == 0:
        raise ValueError("No valid rows/individuals remaining after filtering timestamps/lat/lon/individual ids.")

    counts = df.groupby(INDIVIDUAL_COL).size()
    individual_ids = sorted(counts.index.tolist(), key=lambda x: str(x))
    random.Random(int(random_seed)).shuffle(individual_ids)

    selected: list[object] = []
    cumulative_points = 0
    stop_reason = "all_individuals"

    for indiv in individual_ids:
        selected.append(indiv)
        cumulative_points += int(counts.loc[indiv])
        if len(selected) == int(max_individuals):
            stop_reason = "max_individuals"
            break
        if cumulative_points > int(max_points):
            stop_reason = "max_points"
            break

    df = df[df[INDIVIDUAL_COL].isin(selected)].copy()

    info: dict[str, object] = {
        "max_individuals": int(max_individuals),
        "max_points": int(max_points),
        "random_seed": int(random_seed),
        "stop_reason": str(stop_reason),
        "selected_individuals": [str(x) for x in selected],
        "n_points_before": int(n_points_before),
        "n_individuals_before": int(n_individuals_before),
        "n_points_after": int(len(df)),
        "n_individuals_after": int(df[INDIVIDUAL_COL].nunique()),
    }
    return df, info


def _ensure_set_split(df: pd.DataFrame) -> pd.DataFrame:
    from movebench.trajectories import INDIVIDUAL_COL, SET_COL, TIME_COL

    df = df.copy()

    if SET_COL in df.columns:
        vals = df[SET_COL].dropna().astype(str).unique().tolist()
        if set(vals).issubset({"train", "test"}):
            return df

    # Compute per-individual midpoint split.
    spans = df.groupby(INDIVIDUAL_COL)[TIME_COL].agg(start="min", end="max")
    split_times = spans["start"] + (spans["end"] - spans["start"]) / 2
    df["split_time"] = df[INDIVIDUAL_COL].map(split_times)
    df[SET_COL] = (df[TIME_COL] <= df["split_time"]).map({True: "train", False: "test"})
    return df


def _sort_and_assign_point_id(df: pd.DataFrame) -> pd.DataFrame:
    from movebench.trajectories import INDIVIDUAL_COL, POINT_ID_COL, TIME_COL

    df = df.sort_values([INDIVIDUAL_COL, TIME_COL]).reset_index(drop=True)
    df[POINT_ID_COL] = df.index.astype(int)
    return df


def _write_manifest(
    *,
    study_dir: Path,
    study_id: str,
    source_csv: str,
    df: pd.DataFrame,
    subsample: dict[str, object] | None = None,
) -> None:
    from movebench.features.manifest import utc_now_iso
    from movebench.trajectories import INDIVIDUAL_COL, SET_COL

    n_points = int(len(df))
    n_individuals = int(df[INDIVIDUAL_COL].nunique())
    n_train = int((df[SET_COL] == "train").sum()) if SET_COL in df.columns else 0
    n_test = int((df[SET_COL] == "test").sum()) if SET_COL in df.columns else 0

    manifest = {
        "schema_version": "1",
        "created_at": utc_now_iso(),
        "study_id": str(study_id),
        "source_csv": str(source_csv),
        "trajectories": {
            "file": "trajectories.csv",
            "n_points": n_points,
            "n_individuals": n_individuals,
            "n_train": n_train,
            "n_test": n_test,
        },
    }
    if subsample is not None:
        manifest["subsample"] = dict(subsample)
    (study_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def _commit_staged_study_dir(*, staged_dir: Path, final_dir: Path, force: bool) -> None:
    if not staged_dir.exists():
        raise FileNotFoundError(f"Staged study directory missing: {staged_dir}")

    if final_dir.exists():
        if not force:
            raise FileExistsError(f"Study directory already exists: {final_dir} (use --force to overwrite)")
        backup = final_dir.with_name(f"{final_dir.name}.bak_ingest_{os.getpid()}_{uuid.uuid4().hex}")
        final_dir.rename(backup)
        try:
            staged_dir.rename(final_dir)
        except Exception:
            if not final_dir.exists() and backup.exists():
                backup.rename(final_dir)
            raise
        else:
            shutil.rmtree(backup)
        return

    staged_dir.rename(final_dir)


def _ingest_one_csv(
    *,
    csv_path: Path,
    study_id: str,
    study_dir: Path,
    max_individuals: int,
    max_points: int,
    random_seed: int,
    dry_run: bool,
    force: bool,
    compact_logs: bool = False,
    log_prefix: str = "",
) -> None:
    from movebench.trajectories import INDIVIDUAL_COL, SET_COL

    df = pd.read_csv(csv_path, low_memory=False)
    df = _normalize_columns(df)
    df = _ensure_required_columns(df)
    df = _parse_and_filter(df)
    df, subsample_info = _subsample_individuals(
        df,
        max_individuals=max_individuals,
        max_points=max_points,
        random_seed=random_seed,
    )

    if len(df) == 0:
        raise ValueError("No valid rows remaining after filtering/subsampling.")

    df = _ensure_set_split(df)
    df = _sort_and_assign_point_id(df)

    n_train = int((df[SET_COL] == "train").sum())
    n_test = int((df[SET_COL] == "test").sum())
    if n_train == 0:
        raise ValueError("No train rows found after split assignment; cannot proceed.")

    n_points = int(len(df))
    n_individuals = int(df[INDIVIDUAL_COL].nunique())

    if compact_logs:
        before_p = int(subsample_info["n_points_before"])
        before_i = int(subsample_info["n_individuals_before"])
        reason = str(subsample_info["stop_reason"])
        print(
            f"{log_prefix}{csv_path.name} -> {study_id} "
            f"points={n_points}/{before_p} individuals={n_individuals}/{before_i} stop={reason} "
            f"split=train={n_train} test={n_test}"
        )
    else:
        print(f"{log_prefix}csv_path:   {csv_path}")
        print(f"{log_prefix}study_id:   {study_id}")
        print(f"{log_prefix}study_dir:  {study_dir}")
        print(
            f"{log_prefix}subsample:  "
            f"max_individuals={int(max_individuals)} max_points={int(max_points)} random_seed={int(random_seed)} "
            f"stop={subsample_info['stop_reason']}"
        )
        print(f"{log_prefix}n_points:   {n_points} (from {int(subsample_info['n_points_before'])})")
        print(f"{log_prefix}individuals:{n_individuals} (from {int(subsample_info['n_individuals_before'])})")
        print(f"{log_prefix}split:      train={n_train} test={n_test}")

    if n_test == 0:
        print(f"{log_prefix}WARNING: no test rows found after split assignment (all points are train).")

    if dry_run:
        return

    if study_dir.exists():
        if not force:
            raise FileExistsError(f"Study directory already exists: {study_dir} (use --force to overwrite)")
        shutil.rmtree(study_dir)

    study_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(study_dir / "trajectories.csv", index=False)
    _write_manifest(
        study_dir=study_dir,
        study_id=study_id,
        source_csv=str(csv_path),
        df=df,
        subsample=subsample_info,
    )


def register(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "ingest",
        help="Ingest Movebank CSV(s) into movebench.",
        description="Ingest Movebank-formatted CSVs into movebench as new studies.",
    )
    parser.add_argument(
        "source_path",
        type=Path,
        help="Path to a Movebank-formatted CSV or a directory of CSVs.",
    )
    parser.add_argument("--study-id", type=str, default=None, help="Study id (default: CSV stem).")
    parser.add_argument("--data-root", type=Path, default=None, help="Override MOVEBENCH_DATA_ROOT for this run.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing study directory.")
    parser.add_argument("--dry-run", action="store_true", help="Print what would happen without writing.")
    parser.add_argument(
        "--max-individuals",
        type=_parse_int_with_underscores,
        default=10,
        help="Maximum number of individuals to keep (default: 10).",
    )
    parser.add_argument(
        "--max-points",
        type=_parse_int_with_underscores,
        default=100_000,
        help="Stop after cumulative points exceeds this threshold (default: 100_000).",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=0,
        help="Random seed for individual subsampling (default: 0).",
    )
    parser.set_defaults(func=run, _parser=parser)


def run(args: argparse.Namespace) -> int:
    parser: argparse.ArgumentParser = args._parser

    if args.data_root is not None:
        os.environ["MOVEBENCH_DATA_ROOT"] = str(args.data_root)

    from movebench import paths

    source_path = args.source_path.expanduser().resolve()
    if not source_path.exists():
        parser.error(f"Path not found: {source_path}")

    if source_path.is_dir():
        if args.study_id is not None:
            parser.error("--study-id cannot be used when source_path is a directory (one study per CSV).")

        csv_paths = sorted(p for p in source_path.rglob("*.csv") if p.is_file())
        if not csv_paths:
            parser.error(f"No CSV files found under directory: {source_path}")

        stem_counts = Counter([p.stem for p in csv_paths])
        dupes = sorted([s for s, c in stem_counts.items() if c > 1])
        if dupes:
            parser.error(f"Duplicate CSV stems (study_id collisions) found in directory: {dupes}")

        print(f"dir_path:   {source_path}")
        print(f"csvs:       {len(csv_paths)}")

        jobs: list[tuple[Path, str, Path]] = []
        for p in csv_paths:
            sid = p.stem
            sdir = paths.study_dir(sid)
            jobs.append((p, sid, sdir))

        if not args.dry_run:
            existing = [sdir for _, _, sdir in jobs if sdir.exists()]
            if existing and not args.force:
                raise FileExistsError(
                    "Study directory already exists (use --force to overwrite): "
                    + ", ".join([str(p) for p in existing])
                )

        total = len(jobs)
        if args.dry_run:
            for i, (csv_path, study_id, study_dir) in enumerate(jobs, start=1):
                _ingest_one_csv(
                    csv_path=csv_path,
                    study_id=study_id,
                    study_dir=study_dir,
                    max_individuals=int(args.max_individuals),
                    max_points=int(args.max_points),
                    random_seed=int(args.random_seed),
                    dry_run=True,
                    force=bool(args.force),
                    compact_logs=True,
                    log_prefix=f"[{i}/{total}] ",
                )
            return 0

        tmp_root = paths.studies_root() / f".tmp_ingest_{os.getpid()}_{uuid.uuid4().hex}"
        tmp_root.mkdir(parents=True, exist_ok=False)
        try:
            for i, (csv_path, study_id, _) in enumerate(jobs, start=1):
                staged_dir = tmp_root / study_id
                _ingest_one_csv(
                    csv_path=csv_path,
                    study_id=study_id,
                    study_dir=staged_dir,
                    max_individuals=int(args.max_individuals),
                    max_points=int(args.max_points),
                    random_seed=int(args.random_seed),
                    dry_run=False,
                    force=True,
                    compact_logs=True,
                    log_prefix=f"[{i}/{total}] ",
                )

            for _, study_id, final_dir in jobs:
                staged_dir = tmp_root / study_id
                _commit_staged_study_dir(staged_dir=staged_dir, final_dir=final_dir, force=bool(args.force))
        except Exception:
            shutil.rmtree(tmp_root, ignore_errors=True)
            raise
        else:
            shutil.rmtree(tmp_root, ignore_errors=True)
        return 0

    # ---- single file mode ----
    csv_path = source_path
    study_id = str(args.study_id) if args.study_id else csv_path.stem
    study_dir = paths.study_dir(study_id)
    _ingest_one_csv(
        csv_path=csv_path,
        study_id=study_id,
        study_dir=study_dir,
        max_individuals=int(args.max_individuals),
        max_points=int(args.max_points),
        random_seed=int(args.random_seed),
        dry_run=bool(args.dry_run),
        force=bool(args.force),
        compact_logs=False,
        log_prefix="",
    )
    return 0
