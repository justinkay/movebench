from __future__ import annotations

import colorsys
import io
from textwrap import fill

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages


def format_class_name(col: str) -> str:
    if col == "clean":
        return "Clean"
    if col == "has_any_outlier":
        return "Any Outlier"
    if col.startswith("has_"):
        col = col[4:]
    return col.replace("_", " ").title()


def _hls_to_rgba(h: float, l: float, s: float, alpha: int) -> list[int]:
    r, g, b = colorsys.hls_to_rgb(float(h), float(l), float(s))
    return [
        int(round(r * 255)),
        int(round(g * 255)),
        int(round(b * 255)),
        int(alpha),
    ]


def class_color_palette(class_columns: list[str]) -> dict[str, list[int]]:
    base = {
        "clean": [46, 125, 50, 210],
        "has_any_outlier": [244, 98, 42, 210],
    }
    extras = [c for c in class_columns if c not in base]
    for idx, col in enumerate(extras):
        hue = (0.04 + idx * 0.19) % 1.0
        base[col] = _hls_to_rgba(hue, 0.52, 0.75, 210)
    return base


def primary_class(row: pd.Series, class_columns: list[str]) -> str:
    specific = [c for c in class_columns if c not in {"clean", "has_any_outlier"}]
    for col in specific:
        if bool(row.get(col, False)):
            return col
    if bool(row.get("has_any_outlier", False)):
        return "has_any_outlier"
    return "clean"


def safe_filename_part(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(text)).strip("_")


def ordered_pdf_indices(meta: pd.DataFrame, indices: list[int], order_mode: str) -> list[int]:
    if not indices:
        return []
    subset = meta.loc[indices].copy()
    if order_mode == "outlier_frac":
        subset = subset.sort_values(
            ["outlier_frac", "start_time", "individual_id", "chunk_idx"],
            ascending=[True, True, True, True],
            kind="mergesort",
        )
    else:
        subset = subset.sort_values(
            ["start_time", "individual_id", "chunk_idx"],
            ascending=[True, True, True],
            kind="mergesort",
        )
    return subset.index.tolist()


def _pdf_thumbnail_labels(row: pd.Series) -> str:
    label_cols = sorted(
        col
        for col in row.index
        if col.startswith("has_") and col != "has_any_outlier" and bool(row.get(col, False))
    )
    if not label_cols:
        return "clean"
    return ", ".join(format_class_name(col) for col in label_cols)


def _pdf_class_columns(meta: pd.DataFrame) -> list[str]:
    cols = [c for c in ("clean", "has_any_outlier") if c in meta.columns]
    cols += sorted(c for c in meta.columns if c.startswith("has_") and c != "has_any_outlier")
    return cols


def build_thumbnail_pdf_bytes(
    *,
    source_name: str,
    split: str,
    export_label: str,
    images: np.ndarray,
    meta: pd.DataFrame,
    indices: list[int],
    params: dict,
    order_mode: str,
    cols: int = 5,
    rows: int = 4,
) -> tuple[bytes, str]:
    ordered_indices = ordered_pdf_indices(meta, indices, order_mode)
    class_columns = _pdf_class_columns(meta)
    class_colors = class_color_palette(class_columns)
    per_page = cols * rows
    buf = io.BytesIO()
    order_label = {
        "temporal": "temporal order",
        "outlier_frac": "outlier fraction (low to high)",
    }[order_mode]
    title = (
        f"{source_name} | {export_label} | split={split} | {order_label} | "
        f"chunk={params['chunk_size']} stride={params['stride'] or params['chunk_size']} "
        f"img={params['img_size']} scale={params['scale_mode']}"
    )
    if params.get("scale_mode") == "individual":
        title += f" q={float(params.get('scale_quantile', 0.5)):.2f}"
    with PdfPages(buf) as pdf:
        for page_start in range(0, len(ordered_indices), per_page):
            page_indices = ordered_indices[page_start : page_start + per_page]
            fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.4, rows * 2.6))
            axes_arr = np.atleast_1d(axes).ravel()
            for ax in axes_arr:
                ax.set_xticks([])
                ax.set_yticks([])
            for ax, idx in zip(axes_arr, page_indices):
                row = meta.loc[idx]
                if images[idx].ndim == 2:
                    ax.imshow(images[idx], cmap="gray", vmin=0, vmax=255)
                else:
                    ax.imshow(images[idx])
                row_primary_class = primary_class(row, class_columns)
                border_color = [
                    channel / 255.0
                    for channel in class_colors.get(row_primary_class, [120, 120, 120, 210])[:3]
                ]
                for spine in ax.spines.values():
                    spine.set_visible(True)
                    spine.set_linewidth(2.0)
                    spine.set_edgecolor(border_color)
                ax.set_title(
                    f"#{int(row['global_chunk_idx'])} | {row['individual_id']}\n"
                    f"{pd.Timestamp(row['start_time']).strftime('%Y-%m-%d %H:%M')} | frac={float(row.get('outlier_frac', 0.0)):.2f}",
                    fontsize=7.5,
                    pad=12,
                )
                ax.text(
                    0.5,
                    -0.12,
                    fill(_pdf_thumbnail_labels(row), width=24),
                    transform=ax.transAxes,
                    ha="center",
                    va="top",
                    fontsize=6.5,
                    color="#475467",
                )
            for ax in axes_arr[len(page_indices) :]:
                ax.axis("off")
            fig.suptitle(title, fontsize=12)
            fig.tight_layout(rect=[0, 0.03, 1, 0.95], h_pad=2.4, w_pad=1.0)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
    filename = (
        f"{safe_filename_part(source_name)}__{safe_filename_part(split)}__"
        f"{safe_filename_part(export_label)}__{safe_filename_part(order_mode)}__"
        f"chunk{params['chunk_size']}__img{params['img_size']}.pdf"
    )
    return buf.getvalue(), filename
