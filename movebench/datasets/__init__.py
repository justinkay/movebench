"""PyTorch datasets built on top of movebench feature stores and rasters.

This module intentionally avoids importing dataset implementations at import-time so that
`import movebench.datasets` works in lightweight environments where optional heavy deps (e.g.
`rasterio`) are not installed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

__all__ = ["ChipDataset", "DeepSSFDataset", "SSFDataset", "iSSFDataset"]

if TYPE_CHECKING:
    from movebench.datasets.chips import ChipDataset as ChipDataset
    from movebench.datasets.deepssf import DeepSSFDataset as DeepSSFDataset
    from movebench.datasets.ssf import SSFDataset as SSFDataset
    from movebench.datasets.ssf import iSSFDataset as iSSFDataset


def __getattr__(name: str):
    if name == "ChipDataset":
        from movebench.datasets.chips import ChipDataset as _ChipDataset

        return _ChipDataset
    if name == "DeepSSFDataset":
        from movebench.datasets.deepssf import DeepSSFDataset as _DeepSSFDataset

        return _DeepSSFDataset
    if name == "SSFDataset":
        from movebench.datasets.ssf import SSFDataset as _SSFDataset

        return _SSFDataset
    if name == "iSSFDataset":
        from movebench.datasets.ssf import iSSFDataset as _iSSFDataset

        return _iSSFDataset
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
