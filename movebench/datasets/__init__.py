"""PyTorch datasets built on top of movebench feature stores and rasters."""

from __future__ import annotations

__all__ = ["ChipDataset", "DeepSSFDataset", "SSFDataset", "iSSFDataset"]


def __getattr__(name: str):
    if name == "ChipDataset":
        from movebench.datasets.chips import ChipDataset

        return ChipDataset
    if name == "DeepSSFDataset":
        from movebench.datasets.deepssf import DeepSSFDataset

        return DeepSSFDataset
    if name == "SSFDataset":
        from movebench.datasets.ssf import SSFDataset

        return SSFDataset
    if name == "iSSFDataset":
        from movebench.datasets.ssf import iSSFDataset

        return iSSFDataset
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
