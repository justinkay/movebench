from __future__ import annotations

import hashlib
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from movebench.download_reports import (
    load_chip_request_status_for_extract,
    summarize_failed_chip_report,
)
from movebench.features.config import SamplerConfig, load_sampler_config
from movebench.features.manifest import read_manifest
from movebench.features.points import (
    point_store_manifest_path,
    point_store_train_stats_path,
    resolve_sampler_hash,
)
from movebench.layers.config import load_layers_config
from movebench.layers.registry import REGISTRY
from movebench.study import Study, normalize_split_keys, resolve_study_selection

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_LAYERS_CONFIG = _REPO_ROOT / "configs" / "layers" / "all.yaml"
_DEFAULT_SAMPLER_CONFIG = _REPO_ROOT / "configs" / "samplers" / "issf_defaults.yaml"


@dataclass(frozen=True)
class ExtractionRun:
    layers_config_path: str
    sampler_config_path: str
    plan: Any
    sampler: SamplerConfig | dict[str, Any]
    combo_stem: str
    combo_label: str
    log_dir_name: str


def parse_multi_path_arg(value: str, *, flag_name: str) -> list[str]:
    out = [part.strip() for part in str(value).split(",") if part.strip()]
    if not out:
        raise ValueError(
            f"{flag_name} must include at least one path or a comma-separated list of paths."
        )
    return out


def split_cli_values(values: list[str] | None) -> list[str] | None:
    if values is None:
        return None

    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        for item in str(value).split(","):
            text = item.strip()
            if not text or text in seen:
                continue
            out.append(text)
            seen.add(text)
    return out


def normalize_split_values(values: list[str] | None) -> list[str] | None:
    resolved = split_cli_values(values)
    if resolved is None:
        return None
    return normalize_split_keys(resolved)


def build_extraction_runs(
    *,
    layers_configs: list[list[str]] | None,
    sampler_configs: list[list[str]] | None,
) -> list[ExtractionRun]:
    raw_layers = layers_configs or [[str(_DEFAULT_LAYERS_CONFIG)]]
    raw_samplers = sampler_configs or [[str(_DEFAULT_SAMPLER_CONFIG)]]
    normalized_layers = _dedupe_paths(raw_layers)
    normalized_samplers = _dedupe_paths(raw_samplers)
    return _finalize_log_dir_names(
        [
            _load_extraction_run(
                layers_config_path=layers_config_path,
                sampler_config_path=sampler_config_path,
            )
            for layers_config_path in normalized_layers
            for sampler_config_path in normalized_samplers
        ]
    )

def resolve_requested_studies(studies_arg: str | None) -> list[str]:
    return resolve_study_selection(studies_arg)


def default_splits(study: Study) -> list[str]:
    split_keys = [str(split) for split in study.split_inventory()["split_keys"]]
    return split_keys or ["train"]


def uses_gee_layers(plan: Any) -> bool:
    required = set(str(layer) for layer in getattr(plan, "layers_in_order", []))
    for parents in (getattr(plan, "derived_dependencies", {}) or {}).values():
        required.update(str(parent) for parent in parents)
    for layer in sorted(required):
        if getattr(REGISTRY.get(str(layer)), "source", None) in {"gee_chip", "gee_tile"}:
            return True
    return False


def is_extract_complete(
    *,
    study_dir: Path,
    run: ExtractionRun,
    splits: list[str],
) -> tuple[bool, str]:
    sampler_hash = resolve_sampler_hash(run.sampler)
    expected_band_ids = (
        set(run.plan.band_ids_in_order())
        if hasattr(run.plan, "band_ids_in_order")
        else None
    )
    for split in splits:
        manifest_path = point_store_manifest_path(
            study_dir=Path(study_dir),
            split=str(split),
            rasters_hash=str(run.plan.rasters_hash),
            sampler_hash=sampler_hash,
        )
        if not manifest_path.exists():
            return False, sampler_hash
        try:
            manifest = read_manifest(manifest_path)
        except Exception:
            return False, sampler_hash
        if manifest.rasters_hash != str(run.plan.rasters_hash):
            return False, sampler_hash
        if manifest.sampler_hash != str(sampler_hash):
            return False, sampler_hash
        if (
            expected_band_ids is not None
            and set(manifest.bands.keys()) != expected_band_ids
        ):
            return False, sampler_hash
    stats_path = point_store_train_stats_path(
        study_dir=Path(study_dir),
        rasters_hash=str(run.plan.rasters_hash),
        sampler_hash=sampler_hash,
    )
    return stats_path.exists(), sampler_hash


def describe_known_download_failure(status: dict[str, object] | None) -> str:
    if not isinstance(status, dict):
        return ""
    parts: list[str] = []
    report_path = status.get("report_path")
    if report_path:
        parts.append(f"report={report_path}")
    report = status.get("report")
    if isinstance(report, dict):
        summary = summarize_failed_chip_report(report)
        if summary:
            parts.append(summary)
    return " ".join(parts)


def run_extract_job(
    *,
    study_id: str,
    run: ExtractionRun,
    n_workers: int,
    splits: list[str] | None = None,
    skip_existing: bool = True,
    allow_known_download_failures: bool = False,
) -> int:
    study = Study(str(study_id))
    resolved_splits = (
        default_splits(study) if splits is None else normalize_split_keys(list(splits))
    )
    sampler_hash = resolve_sampler_hash(run.sampler)
    done = False

    if skip_existing:
        done, sampler_hash = is_extract_complete(
            study_dir=Path(study.study_dir),
            run=run,
            splits=resolved_splits,
        )

    request_status = None
    if done:
        print(
            f"[extract] study={study_id} layers={run.layers_config_path} "
            f"sampler={run.sampler_config_path} point stores already exist "
            f"(sampler_hash={sampler_hash})"
        )
    else:
        if uses_gee_layers(run.plan):
            rasters_root = rasters_root_for(run.plan, study_dir=Path(study.study_dir))
            if not rasters_root.exists():
                print(
                    f"[extract] ERROR: missing raster cache root for study={study_id}: {rasters_root}",
                    file=sys.stderr,
                )
                return 2
            request_status = load_chip_request_status_for_extract(
                study=study,
                plan=run.plan,
                splits=resolved_splits,
            )
            if (
                isinstance(request_status, dict)
                and str(request_status.get("status")) == "failed"
                and not allow_known_download_failures
            ):
                print(
                    "[extract] ERROR: known download failures for "
                    f"study={study_id} layers={run.layers_config_path} "
                    f"sampler={run.sampler_config_path} "
                    f"{describe_known_download_failure(request_status)}",
                    file=sys.stderr,
                )
                return 2
            if (
                isinstance(request_status, dict)
                and str(request_status.get("status")) == "failed"
                and allow_known_download_failures
            ):
                print(
                    "[extract] WARNING: proceeding despite known download failures for "
                    f"study={study_id} layers={run.layers_config_path} "
                    f"sampler={run.sampler_config_path} "
                    f"{describe_known_download_failure(request_status)}",
                    file=sys.stderr,
                )

        try:
            result = study.extract_point_features(
                layers_config=str(run.layers_config_path),
                sampler_config=run.sampler,
                splits=resolved_splits,
                n_workers=int(n_workers),
                skip_existing=bool(skip_existing),
            )
        except FileNotFoundError as exc:
            suffix = describe_known_download_failure(request_status)
            if "rasters" in str(exc):
                print(
                    f"[extract] ERROR: missing rasters while extracting for study={study_id}: {exc}",
                    file=sys.stderr,
                )
            elif suffix:
                print(f"[extract] ERROR: {exc} ({suffix})", file=sys.stderr)
            else:
                print(f"[extract] ERROR: {exc}", file=sys.stderr)
            return 2
        sampler_hash = str(result["sampler_hash"])
        print(
            f"[extract] study={study_id} layers={run.layers_config_path} "
            f"sampler={run.sampler_config_path} extracted point stores "
            f"(sampler_hash={sampler_hash})"
        )

    study.compute_normalization_vals(
        from_="points",
        layers_config=str(run.layers_config_path),
        sampler_config=run.sampler,
        split="train",
        skip_existing=True,
    )
    print(
        f"[extract] OK study_id={study_id} layers={run.layers_config_path} "
        f"sampler={run.sampler_config_path} rasters_hash={run.plan.rasters_hash} "
        f"sampler_hash={sampler_hash}"
    )
    return 0


def rasters_root_for(plan: Any, *, study_dir: Path) -> Path:
    return Path(study_dir) / "rasters" / str(plan.rasters_hash)


def _dedupe_paths(groups: list[list[str]]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for group in groups:
        for config_path in group:
            normalized = _normalize_path(config_path)
            if normalized in seen:
                continue
            seen.add(normalized)
            out.append(normalized)
    return out


def _normalize_path(path: str) -> str:
    resolved = Path(str(path)).expanduser()
    if not resolved.is_absolute():
        resolved = (Path.cwd() / resolved).resolve()
    else:
        resolved = resolved.resolve()
    return str(resolved)


def _combo_stem(layers_config_path: str, sampler_config_path: str) -> str:
    return f"{Path(str(layers_config_path)).stem}__{Path(str(sampler_config_path)).stem}"


def _combo_label(layers_config_path: str, sampler_config_path: str) -> str:
    digest = hashlib.sha1(
        f"{layers_config_path}|{sampler_config_path}".encode("utf-8")
    ).hexdigest()[:8]
    return f"{_combo_stem(layers_config_path, sampler_config_path)}-{digest}"


def _load_extraction_run(
    *,
    layers_config_path: str,
    sampler_config_path: str,
) -> ExtractionRun:
    normalized_layers = _normalize_path(layers_config_path)
    normalized_sampler = _normalize_path(sampler_config_path)
    combo_stem = _combo_stem(normalized_layers, normalized_sampler)
    return ExtractionRun(
        layers_config_path=normalized_layers,
        sampler_config_path=normalized_sampler,
        plan=REGISTRY.resolve(load_layers_config(normalized_layers)),
        sampler=load_sampler_config(normalized_sampler),
        combo_stem=combo_stem,
        combo_label=_combo_label(normalized_layers, normalized_sampler),
        log_dir_name=combo_stem,
    )


def _finalize_log_dir_names(runs: list[ExtractionRun]) -> list[ExtractionRun]:
    counts: dict[str, int] = {}
    for run in runs:
        counts[run.combo_stem] = counts.get(run.combo_stem, 0) + 1

    out: list[ExtractionRun] = []
    for run in runs:
        log_dir_name = run.combo_stem
        if counts[run.combo_stem] > 1:
            log_dir_name = run.combo_label
        out.append(replace(run, log_dir_name=log_dir_name))
    return out
