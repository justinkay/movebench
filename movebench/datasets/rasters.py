"""Prepare the existing deepSSF image routes with one shared GPU cache."""

from __future__ import annotations

import torch

from movebench.datasets import DataSplit
from movebench.datasets.deepssf_gpu import (
    AlphaEarthGPUCache,
    build_deepssf_gpu_datasets,
    make_deepssf_gpu_loader,
)
from movebench.datasets.deepssf_wds_cache import (
    make_deepssf_wds_loader,
    steps_per_epoch_from_manifest,
)


class PreparedRasters:
    def __init__(
        self, train: DataSplit, *, batch_size: int, seed: int, device: torch.device
    ) -> None:
        self.inputs = train.inputs
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.output_hw = train.metadata["output_hw"]
        self.channel_indices = list(self.inputs["channel_indices"])
        if self.inputs["kind"] == "gpu":
            self.datasets = build_deepssf_gpu_datasets(
                study_id=train.metadata["study_id"],
                trajectories_path=self.inputs["trajectories_path"],
                plan=self.inputs["plan"],
                splits=list(self.inputs["selected_splits"]),
                catalog=self.inputs["catalog"],
                output_hw=self.output_hw,
            )
            self.pixel_size_m = float(self.datasets["train"].pixel_size_m)
            self.cache = AlphaEarthGPUCache(self.inputs["catalog"], device)
        else:
            for entry in self.inputs["cache_manifests"].values():
                if str(entry["manifest"].get("sample_identity", "")) != "point_id_start":
                    raise ValueError(
                        "deepSSF cache manifest must declare sample_identity='point_id_start'; rebuild the cache"
                    )
            self.pixel_size_m = float(
                self.inputs["cache_manifests"]["train"]["manifest"]["pixel_size_m"]
            )
        self._train_loader = self._make_loader("train", shuffle=True)
        self.steps_per_epoch = (
            len(self._train_loader)
            if self.inputs["kind"] == "gpu"
            else steps_per_epoch_from_manifest(
                self.inputs["cache_manifests"]["train"]["manifest"],
                batch_size=self.batch_size,
            )
        )

    def loader(self, split_key: str, *, shuffle: bool):
        if split_key == "train" and shuffle:
            return self._train_loader
        return self._make_loader(split_key, shuffle=shuffle)

    def _make_loader(self, split_key: str, *, shuffle: bool):
        if self.inputs["kind"] == "gpu":
            return make_deepssf_gpu_loader(
                self.datasets[str(split_key)],
                cache=self.cache,
                batch_size=self.batch_size,
                num_workers=8,
                seed=self.seed,
                shuffle=shuffle,
            )
        return make_deepssf_wds_loader(
            self.inputs["cache_manifests"][str(split_key)]["cache_dir"],
            batch_size=self.batch_size,
            num_workers=8,
            shardshuffle=100 if shuffle else 0,
            shuffle_buffer=self.batch_size * 8 if shuffle else 0,
            seed=self.seed,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=False,
            prefetch_factor=None,
        )


def prepare_rasters(
    train: DataSplit, *, batch_size: int, seed: int, device: torch.device
) -> PreparedRasters:
    return PreparedRasters(train, batch_size=batch_size, seed=seed, device=device)
