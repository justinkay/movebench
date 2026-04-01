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
    "LaPoint_8d40ab81_raw_merged"
    "WarretRodrigues_5a280544_raw_merged_Vulpes_lagopus"
    "Bellan_48c93576_raw_merged"
    "Wicks_d7a990d0_raw_merged"
    "Fleishman_fb9f260b_raw_merged"
    "PillcoHuarcaya_4d453649_raw_merged"
    "Holland_c32173e9_raw_merged"
    "Serieys_fd28aac2_raw_merged"
    "Garthe_ae7b0b8c_raw_merged"
    "aikens2025_pronghorn"
    "aikens2024_pronghorn"
    "stabach2020_wildebeest"
    "Buitendijk_8318572a_raw_merged_Anser_albifrons"
    "gstsdttir_d4074e60_raw_merged"
    "Harel_84c84bcd_raw_merged"
    "WarwickEvans_8e78d67d_raw_merged"
    "Jennings_f577908f_raw_merged"
    "Irvine_a6f0c2e4_raw_merged"
    "vanToor_1111b046_raw_merged_Anas_platyrhynchos"
    "Spiegel_5935dd02_raw_merged_Gyps_africanus"
    "Matthes_316cff40_raw_merged"
    "costa2024_elephant_seals"
    "Padget_12d8637e_raw_merged"
    "Buitendijk_8318572a_raw_merged_Branta_leucopsis"
    "Cruz_b118f49f_raw_merged"
    "Accio_967b7813_raw_merged"
    "Bildstein_b6a0d056_raw_merged"
    "Byholm_aae3de22_raw_merged"
    "Mskens_08b6b850_raw_merged"
)

for layers_config in "${layers_configs[@]}"; do
    for study_id in "${study_ids[@]}"; do
        echo "Downloading ${study_id} (${layers_config##*/})"
        movebench download --layers-config "${layers_config}" --studies "${study_id}" "${extra_args[@]}"
    done
done
