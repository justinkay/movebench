from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LAYER_CONFIG = "configs/layers/all_ae_1d_1d.yaml"


def list_layer_configs(repo_root: Path = REPO_ROOT) -> list[dict[str, str]]:
    config_dir = repo_root / "configs" / "layers"
    if not config_dir.exists():
        return []
    return [
        {"path": str(path.relative_to(repo_root)), "label": path.name}
        for path in sorted(config_dir.glob("*.yaml"))
    ]


def default_layer_config(configs: list[dict[str, str]]) -> str | None:
    if not configs:
        return None
    for item in configs:
        if item["path"] == DEFAULT_LAYER_CONFIG:
            return item["path"]
    return configs[0]["path"]


def resolve_layer_config(
    raw_value: str | None,
    *,
    repo_root: Path = REPO_ROOT,
) -> tuple[str, Path]:
    configs = list_layer_configs(repo_root)
    default_value = default_layer_config(configs)
    if default_value is None:
        raise ValueError("No layer configs found under configs/layers/")

    value = str(raw_value or default_value).strip()
    if not value:
        raise ValueError("Invalid layer config")

    path = (repo_root / value).resolve()
    if repo_root.resolve() not in path.parents:
        raise ValueError("Invalid layer config")
    if not path.is_file():
        raise ValueError("Unknown layer config")
    return value, path
