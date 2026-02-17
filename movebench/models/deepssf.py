"""MoveBench wrapper for the reference deepSSF implementation.

The underlying model lives in `movebench.lib.deepSSF.deepSSF_model`. This file
provides a MoveBench-friendly `nn.Module` wrapper that:
- exposes a stable constructor interface,
- auto-inferrs `dense_dim_in_all` (avoids notebook-only manual bookkeeping),
- provides parameter groups for separate learning rates.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from movebench.lib.deepSSF.deepSSF_model import Conv2d_block_toFC, ConvJointModel, ModelParams


def _infer_dense_dim_in_all(
    *,
    image_dim: int,
    input_channels: int,
    output_channels: int,
    kernel_size: int,
    stride: int,
    padding: int,
    kernel_size_mp: int,
    stride_mp: int,
) -> int:
    """Infer the flattened conv feature size entering the movement FCN."""
    params = ModelParams(
        {
            "batch_size": 1,
            "image_dim": int(image_dim),
            "pixel_size": 1.0,
            "dim_in_nonspatial_to_grid": 0,
            "dense_dim_in_nonspatial": 0,
            "dense_dim_hidden": 1,
            "dense_dim_in_all": 1,
            "input_channels": int(input_channels),
            "output_channels": int(output_channels),
            "kernel_size": int(kernel_size),
            "stride": int(stride),
            "kernel_size_mp": int(kernel_size_mp),
            "stride_mp": int(stride_mp),
            "padding": int(padding),
            "num_movement_params": 1,
            "dropout": 0.0,
            "device": torch.device("cpu"),
        }
    )

    block = Conv2d_block_toFC(params).to(torch.device("cpu"))
    x = torch.zeros((1, int(input_channels), int(image_dim), int(image_dim)), dtype=torch.float32)
    with torch.no_grad():
        y = block(x)
    if y.ndim != 2 or int(y.shape[0]) != 1:
        raise RuntimeError(f"Unexpected Conv2d_block_toFC output shape: {tuple(y.shape)}")
    return int(y.shape[1])


class DeepSSF(nn.Module):
    """deepSSF next-step prediction model (grid-based)."""

    def __init__(
        self,
        env_channels: int,
        *,
        image_dim: int = 101,
        pixel_size_m: float,
        time_channels: int = 4,
        output_channels: int = 4,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        kernel_size_mp: int = 2,
        stride_mp: int = 2,
        num_movement_params: int = 12,
        dense_dim_hidden: int = 128,
        dropout: float = 0.1,
        device: torch.device | str = "cpu",
    ) -> None:
        super().__init__()
        if int(env_channels) <= 0:
            raise ValueError(f"env_channels must be > 0; got {env_channels}")
        if int(image_dim) <= 0:
            raise ValueError(f"image_dim must be > 0; got {image_dim}")
        if float(pixel_size_m) <= 0:
            raise ValueError(f"pixel_size_m must be > 0; got {pixel_size_m}")
        if int(time_channels) <= 0:
            raise ValueError(f"time_channels must be > 0; got {time_channels}")

        self.env_channels = int(env_channels)
        self.image_dim = int(image_dim)
        self.pixel_size_m = float(pixel_size_m)
        self.time_channels = int(time_channels)

        input_channels = int(env_channels) + int(time_channels)
        dense_dim_in_all = _infer_dense_dim_in_all(
            image_dim=int(image_dim),
            input_channels=int(input_channels),
            output_channels=int(output_channels),
            kernel_size=int(kernel_size),
            stride=int(stride),
            padding=int(padding),
            kernel_size_mp=int(kernel_size_mp),
            stride_mp=int(stride_mp),
        )

        dev = torch.device(device)
        params = ModelParams(
            {
                "batch_size": 1,
                "image_dim": int(image_dim),
                "pixel_size": float(pixel_size_m),
                "dim_in_nonspatial_to_grid": int(time_channels),
                "dense_dim_in_nonspatial": 1,
                "dense_dim_hidden": int(dense_dim_hidden),
                "dense_dim_in_all": int(dense_dim_in_all),
                "input_channels": int(input_channels),
                "output_channels": int(output_channels),
                "kernel_size": int(kernel_size),
                "stride": int(stride),
                "kernel_size_mp": int(kernel_size_mp),
                "stride_mp": int(stride_mp),
                "padding": int(padding),
                "num_movement_params": int(num_movement_params),
                "dropout": float(dropout),
                "device": dev,
            }
        )

        self.model = ConvJointModel(params).to(dev)

        # Ensure deepSSF's internal device attributes and precomputed layers are on the model device.
        self._sync_internal_device(dev)

    def parameter_groups(self) -> dict[str, list[nn.Parameter]]:
        """Parameter groups for separate learning rates."""
        habitat = list(self.model.conv_habitat.parameters())
        movement = list(self.model.conv_movement.parameters()) + list(self.model.fcn_movement_all.parameters())
        if not habitat:
            raise ValueError("deepSSF habitat parameter group is empty")
        if not movement:
            raise ValueError("deepSSF movement parameter group is empty")
        return {"movement": movement, "habitat": habitat}

    def _sync_internal_device(self, device: torch.device) -> None:
        """Keep deepSSF internal tensors/device attributes consistent."""
        mgo = getattr(self.model, "movement_grid_output", None)
        if mgo is not None:
            if hasattr(mgo, "device"):
                mgo.device = device
            for name in ("distance_layer", "bearing_layer"):
                t = getattr(mgo, name, None)
                if isinstance(t, torch.Tensor) and t.device != device:
                    setattr(mgo, name, t.to(device))

        # Other submodules store a `.device` attribute; keep it consistent even if unused.
        for sub_name in ("conv_habitat", "conv_movement", "fcn_movement_all", "scalar_grid_output"):
            sub = getattr(self.model, sub_name, None)
            if sub is not None and hasattr(sub, "device"):
                sub.device = device

    def forward(self, x: tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
        spatial, scalars_to_grid, bearing = x
        dev = spatial.device
        self._sync_internal_device(dev)
        return self.model((spatial, scalars_to_grid, bearing))

