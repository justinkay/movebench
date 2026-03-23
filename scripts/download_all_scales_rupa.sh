#!/usr/bin/env bash

set -euo pipefail

extra_args=("$@")
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"

download_study() {
    local study_id="$1"
    echo "Downloading ${study_id}"
    movebench download --layers-config "${repo_root}/configs/layers/all_ae_1h_1h.yaml" --studies "${study_id}" "${extra_args[@]}"
    movebench download --layers-config "${repo_root}/configs/layers/all_ae_12h_12h.yaml" --studies "${study_id}" "${extra_args[@]}"
    movebench download --layers-config "${repo_root}/configs/layers/all_ae_1d_1d.yaml" --studies "${study_id}" "${extra_args[@]}"
    movebench download --layers-config "${repo_root}/configs/layers/all_ae_7d_7d.yaml" --studies "${study_id}" "${extra_args[@]}"
}

download_study "LaPoint_8d40ab81_raw_merged"
download_study "WarretRodrigues_5a280544_raw_merged_Vulpes_lagopus"
download_study "Bellan_48c93576_raw_merged"
download_study "Wicks_d7a990d0_raw_merged"
download_study "Fleishman_fb9f260b_raw_merged"
download_study "PillcoHuarcaya_4d453649_raw_merged"
download_study "Holland_c32173e9_raw_merged"
download_study "Serieys_fd28aac2_raw_merged"
download_study "Garthe_ae7b0b8c_raw_merged"
download_study "aikens2025_pronghorn"
download_study "aikens2024_pronghorn"
download_study "stabach2020_wildebeest"
download_study "Buitendijk_8318572a_raw_merged_Anser_albifrons"
download_study "gstsdttir_d4074e60_raw_merged"
download_study "Harel_84c84bcd_raw_merged"
download_study "WarwickEvans_8e78d67d_raw_merged"
download_study "Jennings_f577908f_raw_merged"
download_study "Irvine_a6f0c2e4_raw_merged"
download_study "vanToor_1111b046_raw_merged_Anas_platyrhynchos"
download_study "Spiegel_5935dd02_raw_merged_Gyps_africanus"
download_study "Matthes_316cff40_raw_merged"
download_study "costa2024_elephant_seals"
download_study "Padget_12d8637e_raw_merged"
download_study "Buitendijk_8318572a_raw_merged_Branta_leucopsis"
download_study "Cruz_b118f49f_raw_merged"
download_study "Accio_967b7813_raw_merged"
download_study "Bildstein_b6a0d056_raw_merged"
download_study "Byholm_aae3de22_raw_merged"
download_study "Mskens_08b6b850_raw_merged"
