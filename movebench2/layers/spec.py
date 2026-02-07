from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


LayerSource = Literal["gee_chip", "gee_tile", "local_hdf", "local_tiff", "derived"]


@dataclass(frozen=True)
class LayerSpec:
    """Specification for a single layer in the layer registry."""

    name: str
    source: LayerSource
    bands: list[str]

    categorical: bool = False
    num_classes: int = 0
    normalize: bool = True

    # For derived layers, the parent layer names needed to compute this layer.
    derived_from: list[str] | None = None

    # Data transforms applied at read/extraction time.
    scale_factor: float = 1.0
    valid_range: tuple[float, float] | None = None

