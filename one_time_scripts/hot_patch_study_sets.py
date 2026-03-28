from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from movebench.cli.ingest import _ensure_required_columns, _normalize_columns, _parse_and_filter
from movebench.trajectories import INDIVIDUAL_COL, SET_COL


DEFAULT_STUDIES_ROOT = Path("/archive/vision/beery/movebench/studies")
DEFAULT_V2_ROOT = Path("/archive/vision/beery/movebench/cleaned_csvs_v2")


@dataclass
class StudyPatchPlan:
    study_id: str
    study_dir: Path
    manifest_path: Path
    trajectories_path: Path
    v2_csv_path: Path
    updated_trajectories: pd.DataFrame
    updated_manifest: dict[str, object]
    n_rows_changed: int
    n_rows_missing_in_v2: int
    n_rows_unused_in_v2: int
    n_rows_unused_after_patch: int
    set_transitions: dict[str, int]


def _selected_individuals(manifest: dict[str, object]) -> list[str] | None:
    subsample = manifest.get("subsample")
    if not isinstance(subsample, dict):
        return None
    selected = subsample.get("selected_individuals")
    if not isinstance(selected, list):
        return None
    return [str(value) for value in selected]


def _load_v2_source(
    v2_csv_path: Path, *, selected_individuals: list[str] | None
) -> pd.DataFrame:
    df = pd.read_csv(v2_csv_path, low_memory=False)
    df = _normalize_columns(df)
    df = _ensure_required_columns(df)
    df = _parse_and_filter(df)
    if selected_individuals is not None:
        df = df[df[INDIVIDUAL_COL].astype(str).isin(selected_individuals)].copy()
    if "event-id" not in df.columns:
        raise ValueError(f"Source CSV is missing 'event-id': {v2_csv_path}")
    if df["event-id"].isna().any():
        raise ValueError(f"Source CSV has null 'event-id' values: {v2_csv_path}")
    if df["event-id"].duplicated().any():
        raise ValueError(f"Source CSV has duplicate 'event-id' values: {v2_csv_path}")
    if SET_COL not in df.columns:
        raise ValueError(f"Source CSV is missing {SET_COL!r}: {v2_csv_path}")
    return df


def _updated_manifest(
    manifest: dict[str, object], updated_trajectories: pd.DataFrame
) -> dict[str, object]:
    split_counts = Counter(updated_trajectories[SET_COL].astype(str).tolist())
    updated = dict(manifest)
    trajectories = dict(updated.get("trajectories", {}))
    trajectories.pop("n_test", None)
    trajectories["n_points"] = int(len(updated_trajectories))
    trajectories["n_individuals"] = int(updated_trajectories[INDIVIDUAL_COL].nunique())
    trajectories["n_train"] = int(split_counts.get("train", 0))
    trajectories["n_test_temporal"] = int(split_counts.get("test-temporal", 0))
    trajectories["n_test_individual"] = int(split_counts.get("test-individual", 0))
    trajectories["n_unused"] = int(split_counts.get("unused", 0))
    updated["trajectories"] = trajectories
    return updated


def plan_study_patch(*, study_dir: Path, v2_csv_path: Path) -> StudyPatchPlan:
    manifest_path = study_dir / "manifest.json"
    trajectories_path = study_dir / "trajectories.csv"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    trajectories = pd.read_csv(trajectories_path, low_memory=False)
    if "event-id" not in trajectories.columns:
        raise ValueError(f"Trajectories CSV is missing 'event-id': {trajectories_path}")
    if trajectories["event-id"].isna().any():
        raise ValueError(f"Trajectories CSV has null 'event-id' values: {trajectories_path}")
    if trajectories["event-id"].duplicated().any():
        raise ValueError(f"Trajectories CSV has duplicate 'event-id' values: {trajectories_path}")

    source_v2 = _load_v2_source(
        v2_csv_path,
        selected_individuals=_selected_individuals(manifest),
    )
    set_by_event_id = source_v2.set_index("event-id")[SET_COL]
    updated_trajectories = trajectories.copy()
    old_set = updated_trajectories[SET_COL].astype(str)
    mapped_set = updated_trajectories["event-id"].map(set_by_event_id)
    new_set = mapped_set.fillna("unused")
    updated_trajectories[SET_COL] = new_set

    changed_mask = old_set != new_set
    transitions = (
        pd.DataFrame({"old": old_set.loc[changed_mask], "new": new_set.loc[changed_mask]})
        .value_counts(sort=False)
        .sort_index()
    )
    updated_manifest = _updated_manifest(manifest, updated_trajectories)
    return StudyPatchPlan(
        study_id=str(manifest["study_id"]),
        study_dir=study_dir,
        manifest_path=manifest_path,
        trajectories_path=trajectories_path,
        v2_csv_path=v2_csv_path,
        updated_trajectories=updated_trajectories,
        updated_manifest=updated_manifest,
        n_rows_changed=int(changed_mask.sum()),
        n_rows_missing_in_v2=int(mapped_set.isna().sum()),
        n_rows_unused_in_v2=int(mapped_set.eq("unused").sum()),
        n_rows_unused_after_patch=int(new_set.eq("unused").sum()),
        set_transitions={
            f"{old}->{new}": int(count)
            for (old, new), count in transitions.items()
        },
    )


def _write_manifest(path: Path, manifest: dict[str, object]) -> None:
    tmp_path = path.with_name(f"{path.name}.tmp_hot_patch")
    tmp_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    tmp_path.replace(path)


def _write_trajectories(path: Path, trajectories: pd.DataFrame) -> None:
    tmp_path = path.with_name(f"{path.name}.tmp_hot_patch")
    trajectories.to_csv(tmp_path, index=False)
    tmp_path.replace(path)


def apply_study_patch(*, study_dir: Path, v2_csv_path: Path) -> StudyPatchPlan:
    plan = plan_study_patch(study_dir=study_dir, v2_csv_path=v2_csv_path)
    _write_trajectories(plan.trajectories_path, plan.updated_trajectories)
    _write_manifest(plan.manifest_path, plan.updated_manifest)
    return plan


def build_dry_run_report(*, studies_root: Path, v2_root: Path) -> dict[str, object]:
    study_reports: list[dict[str, object]] = []
    transition_totals: Counter[str] = Counter()
    for manifest_path in sorted(studies_root.glob("*/manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        study_dir = manifest_path.parent
        v2_csv_path = v2_root / Path(str(manifest["source_csv"])).name
        plan = plan_study_patch(study_dir=study_dir, v2_csv_path=v2_csv_path)
        transition_totals.update(plan.set_transitions)
        study_reports.append(
            {
                "study_id": plan.study_id,
                "study_dir": str(plan.study_dir),
                "manifest_path": str(plan.manifest_path),
                "trajectories_path": str(plan.trajectories_path),
                "v2_csv_path": str(plan.v2_csv_path),
                "n_rows_changed": plan.n_rows_changed,
                "n_rows_missing_in_v2": plan.n_rows_missing_in_v2,
                "n_rows_unused_in_v2": plan.n_rows_unused_in_v2,
                "n_rows_unused_after_patch": plan.n_rows_unused_after_patch,
                "set_transitions": plan.set_transitions,
                "manifest_before": json.loads(plan.manifest_path.read_text(encoding="utf-8"))[
                    "trajectories"
                ],
                "manifest_after": plan.updated_manifest["trajectories"],
            }
        )
    changed_studies = [study for study in study_reports if study["n_rows_changed"] > 0]
    return {
        "studies_root": str(studies_root),
        "v2_root": str(v2_root),
        "n_studies_total": len(study_reports),
        "n_studies_changed": len(changed_studies),
        "n_rows_changed_total": int(sum(study["n_rows_changed"] for study in study_reports)),
        "n_rows_missing_in_v2_total": int(
            sum(study["n_rows_missing_in_v2"] for study in study_reports)
        ),
        "n_rows_unused_in_v2_total": int(
            sum(study["n_rows_unused_in_v2"] for study in study_reports)
        ),
        "n_rows_unused_after_patch_total": int(
            sum(study["n_rows_unused_after_patch"] for study in study_reports)
        ),
        "set_transitions_total": dict(sorted(transition_totals.items())),
        "studies": changed_studies,
    }


def _print_summary(report: dict[str, object], *, top_n: int) -> None:
    print(
        "dry-run summary:"
        f" studies_changed={report['n_studies_changed']}/{report['n_studies_total']}"
        f" rows_changed={report['n_rows_changed_total']}"
        f" rows_missing_in_v2={report['n_rows_missing_in_v2_total']}"
        f" rows_unused_in_v2={report['n_rows_unused_in_v2_total']}"
        f" rows_unused_after_patch={report['n_rows_unused_after_patch_total']}"
    )
    print("set transitions:")
    for key, count in report["set_transitions_total"].items():
        print(f"  {key}: {count}")
    print("top changed studies:")
    studies = sorted(
        report["studies"],
        key=lambda study: (-int(study["n_rows_changed"]), str(study["study_id"])),
    )
    for study in studies[: int(top_n)]:
        print(
            f"  {study['study_id']}: rows_changed={study['n_rows_changed']} "
            f"missing_in_v2={study['n_rows_missing_in_v2']} "
            f"unused_in_v2={study['n_rows_unused_in_v2']} "
            f"unused_after_patch={study['n_rows_unused_after_patch']} "
            f"transitions={study['set_transitions']}"
        )


def apply_all_studies(*, studies_root: Path, v2_root: Path) -> dict[str, int]:
    n_studies_applied = 0
    n_studies_changed = 0
    for manifest_path in sorted(studies_root.glob("*/manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        study_dir = manifest_path.parent
        v2_csv_path = v2_root / Path(str(manifest["source_csv"])).name
        plan = apply_study_patch(study_dir=study_dir, v2_csv_path=v2_csv_path)
        n_studies_applied += 1
        if plan.n_rows_changed > 0:
            n_studies_changed += 1
    return {
        "n_studies_applied": n_studies_applied,
        "n_studies_changed": n_studies_changed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--studies-root", type=Path, default=DEFAULT_STUDIES_ROOT)
    parser.add_argument("--v2-root", type=Path, default=DEFAULT_V2_ROOT)
    parser.add_argument("--report-path", type=Path, default=None)
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    report = build_dry_run_report(
        studies_root=args.studies_root,
        v2_root=args.v2_root,
    )
    _print_summary(report, top_n=int(args.top_n))
    if args.report_path is not None:
        args.report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(f"wrote report: {args.report_path}")
    if args.apply:
        counts = apply_all_studies(
            studies_root=args.studies_root,
            v2_root=args.v2_root,
        )
        print(
            "applied patch:"
            f" studies={counts['n_studies_applied']}"
            f" changed={counts['n_studies_changed']}"
        )


if __name__ == "__main__":
    main()
