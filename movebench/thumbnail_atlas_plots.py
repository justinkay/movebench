from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def available_thumbnail_label_columns(df: pd.DataFrame) -> list[str]:
    return sorted(c for c in df.columns if c.startswith("has_"))


def _coerce_label_mask(series: pd.Series) -> np.ndarray:
    return series.fillna(False).astype(bool).to_numpy()


def plot_embedding_label_scatter(
    df: pd.DataFrame,
    label_col: str,
    *,
    x: str = "projection_x",
    y: str = "projection_y",
    ax=None,
) -> None:
    available = available_thumbnail_label_columns(df)
    if label_col not in df.columns:
        raise KeyError(
            f"{label_col!r} not found. Available has_* columns: {available}"
        )

    mask = _coerce_label_mask(df[label_col])
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))

    ax.scatter(
        df.loc[~mask, x],
        df.loc[~mask, y],
        c="#94a3b8",
        s=18,
        alpha=0.35,
        linewidths=0,
        label="false",
    )
    ax.scatter(
        df.loc[mask, x],
        df.loc[mask, y],
        c="#d26531",
        s=28,
        alpha=0.9,
        linewidths=0.4,
        edgecolors="white",
        label="true",
    )
    ax.set_title(label_col.replace("_", " "))
    ax.set_xlabel(x.replace("_", " "))
    ax.set_ylabel(y.replace("_", " "))
    ax.legend(title="label", frameon=False)


def system_speed_category(df: pd.DataFrame) -> pd.Categorical:
    required = ["has_system_outlier", "has_speed_outlier"]
    available = available_thumbnail_label_columns(df)
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise KeyError(
            "Missing required chunk-level label columns: "
            f"{missing}. Available has_* columns: {available}"
        )

    system_mask = _coerce_label_mask(df["has_system_outlier"])
    speed_mask = _coerce_label_mask(df["has_speed_outlier"])
    labels = np.full(len(df), "neither", dtype=object)
    labels[system_mask & ~speed_mask] = "system only"
    labels[~system_mask & speed_mask] = "speed only"
    labels[system_mask & speed_mask] = "both"
    return pd.Categorical(
        labels,
        categories=["neither", "system only", "speed only", "both"],
        ordered=True,
    )


def plot_system_speed_comparison(
    df: pd.DataFrame,
    *,
    x: str = "projection_x",
    y: str = "projection_y",
    ax=None,
) -> None:
    categories = system_speed_category(df)
    palette = {
        "neither": "#94a3b8",
        "system only": "#2563eb",
        "speed only": "#d97706",
        "both": "#b91c1c",
    }
    sizes = {
        "neither": 18,
        "system only": 26,
        "speed only": 26,
        "both": 34,
    }
    alphas = {
        "neither": 0.28,
        "system only": 0.78,
        "speed only": 0.78,
        "both": 0.92,
    }
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))

    for label in categories.categories:
        mask = categories == label
        ax.scatter(
            df.loc[mask, x],
            df.loc[mask, y],
            c=palette[label],
            s=sizes[label],
            alpha=alphas[label],
            linewidths=0.4 if label == "both" else 0,
            edgecolors="white" if label == "both" else "none",
            label=label,
        )

    ax.set_title("system vs speed outlier overlap")
    ax.set_xlabel(x.replace("_", " "))
    ax.set_ylabel(y.replace("_", " "))
    ax.legend(title="category", frameon=False)


def save_projection_diagnostics(
    df: pd.DataFrame,
    *,
    out_dir: str | Path,
) -> dict[str, str]:
    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6.5, 5.5), constrained_layout=True)
    system_path = target / "projection_system_outlier.png"
    plot_embedding_label_scatter(df, "has_system_outlier", ax=ax)
    fig.savefig(system_path, dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 5.5), constrained_layout=True)
    speed_path = target / "projection_speed_outlier.png"
    plot_embedding_label_scatter(df, "has_speed_outlier", ax=ax)
    fig.savefig(speed_path, dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 5.5), constrained_layout=True)
    combined_path = target / "projection_system_speed_overlap.png"
    plot_system_speed_comparison(df, ax=ax)
    fig.savefig(combined_path, dpi=160)
    plt.close(fig)

    return {
        "projection_system_outlier": str(system_path),
        "projection_speed_outlier": str(speed_path),
        "projection_system_speed_overlap": str(combined_path),
    }
