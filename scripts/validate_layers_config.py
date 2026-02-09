from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from movebench.layers import REGISTRY
from movebench.layers.config import load_layers_config


def main(argv: list[str]) -> int:
    if len(argv) != 2 or argv[1] in ("-h", "--help"):
        print("Usage: python scripts/validate_layers_config.py <path/to/layers.yaml>")
        return 2

    path = Path(argv[1])
    cfg = load_layers_config(path)
    plan = REGISTRY.resolve(cfg)

    print(f"config: {path}")
    print(f"layers: {len(plan.layers_in_order)}")
    print(f"channels: {plan.num_channels()}")

    if plan.chip_groups:
        print("chip_groups:")
        for name, g in plan.chip_groups.items():
            layers = list(g["layers"])
            print(f"  - {name}: {len(layers)} layer(s), target_scale_m={g['target_scale_m']}")
            print(f"    layers={layers}")
    else:
        print("chip_groups: (none)")

    if plan.derived_dependencies:
        print("derived_dependencies:")
        for name, parents in sorted(plan.derived_dependencies.items(), key=lambda kv: kv[0]):
            print(f"  - {name}: {parents}")
        print(f"derived_eval_order: {plan.derived_eval_order()}")
    else:
        print("derived_dependencies: (none)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
