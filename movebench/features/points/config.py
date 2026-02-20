from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Literal

SamplerType = Literal["empirical", "parametric"]


def stable_hash_dict(d: dict[str, Any]) -> str:
    payload = json.dumps(d, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def stable_hash(d: dict[str, Any]) -> str:
    """Backwards/ergonomic alias for stable_hash_dict."""
    return stable_hash_dict(d)


@dataclass(frozen=True)
class PointStoreConfig:
    """Configuration that defines a point-store identity.

    If fully specified, this config deterministically maps to a `store_id` and
    an on-disk store directory.

    If partially specified (fields left as None), store resolution may scan the
    filesystem and require a unique match.
    """

    sampler_type: SamplerType | None = None
    num_candidates: int | None = None
    random_seed: int | None = None
    step_multiplier: float | None = None
    step_q: float | None = None
    gap_q: float | None = None
    bounds_buffer_deg: float | None = None

    def is_fully_specified(self) -> bool:
        return all(
            getattr(self, k) is not None
            for k in (
                "sampler_type",
                "num_candidates",
                "random_seed",
                "step_multiplier",
                "step_q",
                "gap_q",
                "bounds_buffer_deg",
            )
        )

    def to_dict(self) -> dict[str, Any]:
        if not self.is_fully_specified():
            missing = [
                k
                for k in (
                    "sampler_type",
                    "num_candidates",
                    "random_seed",
                    "step_multiplier",
                    "step_q",
                    "gap_q",
                    "bounds_buffer_deg",
                )
                if getattr(self, k) is None
            ]
            raise ValueError(f"PointStoreConfig is not fully specified; missing: {missing}")

        return {
            "sampler_type": str(self.sampler_type),
            "num_candidates": int(self.num_candidates),  # type: ignore[arg-type]
            "random_seed": int(self.random_seed),  # type: ignore[arg-type]
            "step_multiplier": float(self.step_multiplier),  # type: ignore[arg-type]
            "step_q": float(self.step_q),  # type: ignore[arg-type]
            "gap_q": float(self.gap_q),  # type: ignore[arg-type]
            "bounds_buffer_deg": float(self.bounds_buffer_deg),  # type: ignore[arg-type]
        }

    def store_id(self) -> str:
        return stable_hash_dict(self.to_dict())
