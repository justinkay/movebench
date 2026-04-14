#!/usr/bin/env bash

set -euo pipefail
umask 0002

for arg in "$@"; do
    if [[ "${arg}" == "--no-skip-existing" ]]; then
        echo "download_all_scales_marine.sh refuses --no-skip-existing to avoid overwriting existing chips/tiles." >&2
        exit 2
    fi
done

extra_args=("$@")
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"

layers_configs=(
    "${repo_root}/configs/layers/marine_1h_1h.yaml"
    "${repo_root}/configs/layers/marine_12h_12h.yaml"
    "${repo_root}/configs/layers/marine_1d_1d.yaml"
    "${repo_root}/configs/layers/marine_7d_7d.yaml"
)

for layers_config in "${layers_configs[@]}"; do
    echo "Downloading marine (${layers_config##*/})"
    movebench download --layers-config "${layers_config}" --studies marine "${extra_args[@]}"
done
