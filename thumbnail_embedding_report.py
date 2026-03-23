from __future__ import annotations

import argparse

from movebench.thumbnail_embedding import CACHE_ROOT, export_embedding_report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export a static thumbnail embedding report from a cached run.")
    parser.add_argument("--run-id", required=True, help="Completed run id to export.")
    parser.add_argument("--baseline-run-id", default=None, help="Optional baseline run id for comparison.")
    parser.add_argument("--out-dir", default=None, help="Optional output directory. Defaults under the explorer cache.")
    parser.add_argument("--cache-root", default=str(CACHE_ROOT), help="Cache root containing cached run artifacts.")
    args = parser.parse_args(argv)

    out_path = export_embedding_report(
        args.run_id,
        baseline_run_id=args.baseline_run_id,
        out_dir=args.out_dir,
        cache_root=args.cache_root,
    )
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
