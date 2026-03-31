"""Next-generation MoveBench pipeline (work-in-progress)."""

from __future__ import annotations

__all__ = ["Study"]


def __getattr__(name: str):
    if name != "Study":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from movebench.study import Study

    return Study
