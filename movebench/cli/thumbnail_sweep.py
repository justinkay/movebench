from __future__ import annotations

import argparse

from movebench.thumbnail_sweep import DEFAULT_CHUNK_SIZES
from movebench.thumbnail_sweep import DEFAULT_COLOR_MODES
from movebench.thumbnail_sweep import DEFAULT_SCALE_MODES
from movebench.thumbnail_sweep import run_thumbnail_sweep


def register(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "sweep-thumbnails",
        help="Run a thumbnail sweep over chunk size, color mode, and scale mode.",
    )
    parser.add_argument("base_config", help="Path to a thumbnail generation config JSON file.")
    parser.add_argument("out_root", help="Directory where run outputs should be written.")
    parser.add_argument("--chunk-sizes", nargs="+", type=int, default=list(DEFAULT_CHUNK_SIZES))
    parser.add_argument("--color-modes", nargs="+", default=list(DEFAULT_COLOR_MODES))
    parser.add_argument("--scale-modes", nargs="+", default=list(DEFAULT_SCALE_MODES))
    parser.add_argument("--embedding-config", default=None)
    parser.add_argument("--projection-config", default=None)
    parser.add_argument("--save-thumbnail-pngs", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.set_defaults(func=run)


def run(args: argparse.Namespace) -> int:
    import json
    from pathlib import Path

    def _load(path: str | None) -> dict:
        if path is None:
            return {}
        return json.loads(Path(path).read_text(encoding="utf-8"))

    run_thumbnail_sweep(
        args.base_config,
        args.out_root,
        chunk_sizes=args.chunk_sizes,
        color_modes=args.color_modes,
        scale_modes=args.scale_modes,
        embedding_overrides=_load(args.embedding_config),
        projection_overrides=_load(args.projection_config),
        save_thumbnail_pngs=bool(args.save_thumbnail_pngs),
        force=bool(args.force),
        show_progress=not bool(args.quiet),
    )
    return 0
