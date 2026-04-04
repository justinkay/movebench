#!/usr/bin/env python

from __future__ import annotations

import _bootstrap as bootstrap  # noqa: F401
import sys

from movebench.slurm import extract as slurm_extract


def _translate_legacy_args(argv: list[str]) -> list[str]:
    out: list[str] = []
    for arg in argv:
        if arg == "--study":
            out.append("--worker-study")
            continue
        out.append(arg)
    return out


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    return slurm_extract.main(_translate_legacy_args(list(argv)))


if __name__ == "__main__":
    raise SystemExit(main())
