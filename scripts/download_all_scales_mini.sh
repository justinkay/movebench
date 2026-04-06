#!/usr/bin/env bash

set -euo pipefail

extra_args=("$@")
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"

layers_configs=(
    "${repo_root}/configs/layers/all_ae_1h_1h.yaml"
    "${repo_root}/configs/layers/all_ae_12h_12h.yaml"
    "${repo_root}/configs/layers/all_ae_1d_1d.yaml"
    "${repo_root}/configs/layers/all_ae_7d_7d.yaml"
)

study_ids=(
    "forrest2024_buffalo"
    "Bellan_48c93576_raw_merged"
    "Garthe_ae7b0b8c_raw_merged"
    "aikens2024_pronghorn"
    "costa2024_elephant_seals"
    "Accio_967b7813_raw_merged"
    "BartlamBrooks_c9b4f9b5_raw_merged"
    "Lago_8b91498b_raw_merged"
    "BastilleRousseau_9eaf99f2_raw_merged_Chelonoidis_porteri"
    "Bildstein_3e431f5b_raw_merged_Cathartes_aura"
)

for layers_config in "${layers_configs[@]}"; do
    for study_id in "${study_ids[@]}"; do
        echo "Downloading ${study_id} (${layers_config##*/})"
        movebench download --layers-config "${layers_config}" --studies "${study_id}" "${extra_args[@]}"
    done
done
