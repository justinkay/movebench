import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import JSONResponse, Response

import yaml

from app.configs import default_layer_config, list_layer_configs
from app.summary import build_trajectory_summary


def json_error(message: str, status_code: int) -> JSONResponse:
    return JSONResponse({"error": message}, status_code=status_code)


def resolve_studies_root() -> Path:
    data_root = os.environ.get("MOVEBENCH_DATA_ROOT")
    if data_root:
        return (Path(data_root) / "studies").resolve()
    config_path = Path(__file__).resolve().parents[1] / "configs" / "paths.yaml"
    with config_path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}
    value = cfg.get("data_root")
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Could not resolve data_root from {config_path}")
    return (Path(value) / "studies").resolve()


def resolve_study_dir(studies_root: Path, study_id: str) -> Path:
    value = str(study_id).strip()
    if not value or "/" in value or "\\" in value:
        raise ValueError("Invalid artifact")
    path = (studies_root / value).resolve()
    if path.parent != studies_root:
        raise ValueError("Invalid artifact")
    if not path.is_dir():
        raise ValueError("Unknown artifact")
    return path


def artifact_payload(study_dir: Path) -> dict:
    trajectory_path = study_dir / "trajectories.csv"
    return {
        "id": study_dir.name,
        "label": f"{study_dir.name}/trajectories.csv",
        "study_id": study_dir.name,
        "path": str(trajectory_path),
        "exists": trajectory_path.is_file(),
    }


def list_artifacts(studies_root: Path) -> list[dict]:
    if not studies_root.exists():
        return []
    return [
        artifact_payload(study_dir)
        for study_dir in sorted(path for path in studies_root.iterdir() if path.is_dir())
        if (study_dir / "trajectories.csv").is_file()
    ]


def register_routes(app: FastAPI, *, studies_root: Path | None = None):
    studies_root = (studies_root or resolve_studies_root()).resolve()
    app.state.studies_root = studies_root

    @app.get("/api/config")
    async def get_config():
        return JSONResponse({"studies_root": str(studies_root)})

    @app.get("/api/artifacts")
    async def get_artifacts():
        layer_configs = list_layer_configs()
        return JSONResponse({
            "studies_root": str(studies_root),
            "artifacts": list_artifacts(studies_root),
            "layer_configs": layer_configs,
            "default_layer_config": default_layer_config(layer_configs),
        })

    @app.get("/api/artifacts/{study_id}/summary")
    async def get_artifact_summary(study_id: str):
        try:
            study_dir = resolve_study_dir(studies_root, study_id)
            path = study_dir / "trajectories.csv"
            if not path.is_file():
                raise ValueError("Artifact is missing trajectories.csv")
            summary = build_trajectory_summary(path)
        except ValueError as exc:
            return json_error(str(exc), 404)
        return JSONResponse({
            **summary,
            "artifact": artifact_payload(study_dir),
        })

    @app.get("/api/artifacts/{study_id}/covariates")
    async def get_covariates(study_id: str, config: str | None = None, point_id: int | None = None):
        try:
            resolve_study_dir(studies_root, study_id)
            from app.covariates import build_covariates_payload

            return JSONResponse(build_covariates_payload(study_id, config=config, point_id=point_id))
        except (ValueError, FileNotFoundError) as exc:
            return json_error(str(exc), 404)

    @app.get("/api/artifacts/{study_id}/covariates/steps")
    async def get_covariate_steps(study_id: str, config: str | None = None, point_id: int | None = None):
        try:
            resolve_study_dir(studies_root, study_id)
            from app.covariates import build_covariate_steps_payload

            return JSONResponse(build_covariate_steps_payload(study_id, config=config, point_id=point_id))
        except (ValueError, FileNotFoundError) as exc:
            return json_error(str(exc), 404)

    @app.get("/api/artifacts/{study_id}/covariates/chip-groups/{group_name}/bands/{band_index}.png")
    async def get_covariate_chip_group_png(
        study_id: str,
        group_name: str,
        band_index: int,
        config: str | None = None,
        point_id: int | None = None,
    ):
        try:
            resolve_study_dir(studies_root, study_id)
            from app.covariates import chip_group_band_png

            return Response(
                content=chip_group_band_png(
                    study_id,
                    config=config,
                    point_id=point_id,
                    group_name=group_name,
                    band_index=band_index,
                ),
                media_type="image/png",
            )
        except (ValueError, FileNotFoundError) as exc:
            return json_error(str(exc), 404)

    @app.get("/api/artifacts/{study_id}/covariates/tile-layers/{layer}/bands/{band_index}.png")
    async def get_covariate_tile_layer_png(
        study_id: str,
        layer: str,
        band_index: int,
        config: str | None = None,
        point_id: int | None = None,
    ):
        try:
            resolve_study_dir(studies_root, study_id)
            from app.covariates import tile_layer_band_png

            return Response(
                content=tile_layer_band_png(
                    study_id,
                    config=config,
                    point_id=point_id,
                    layer=layer,
                    band_index=band_index,
                ),
                media_type="image/png",
            )
        except (ValueError, FileNotFoundError) as exc:
            return json_error(str(exc), 404)

    @app.get("/api/artifacts/{study_id}/covariates/derived-layers/{layer}/bands/{band_index}.png")
    async def get_covariate_derived_layer_png(
        study_id: str,
        layer: str,
        band_index: int,
        config: str | None = None,
        point_id: int | None = None,
    ):
        try:
            resolve_study_dir(studies_root, study_id)
            from app.covariates import derived_layer_band_png

            return Response(
                content=derived_layer_band_png(
                    study_id,
                    config=config,
                    point_id=point_id,
                    layer=layer,
                    band_index=band_index,
                ),
                media_type="image/png",
            )
        except (ValueError, FileNotFoundError) as exc:
            return json_error(str(exc), 404)
