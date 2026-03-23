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

download_study "Serieys_7de63cc4_raw_merged"
download_study "Graf_e3df6f82_raw_merged"
download_study "Toivonen_338388ea_raw_merged"
download_study "Buitendijk_8318572a_raw_merged_Anser_anser"
download_study "WarretRodrigues_5a280544_raw_merged_Vulpes_vulpes"
download_study "Boyce_3a171441_raw_merged"
download_study "Scasta_2facc53d_raw_merged"
download_study "Hebblewhite_dce434f5_raw_merged"
download_study "Prugh_e7808391_raw_merged_Lynx_rufus"
download_study "Power_6360747f_raw_merged"
download_study "Glass_fa3e16ed_raw_merged"
download_study "Latham_c0de801a_raw_merged"
download_study "Klzsch_a61cb8c5_raw_merged_Branta_leucopsis"
download_study "Setyawan_0425c92c_raw_merged"
download_study "Ronconi_70744563_raw_merged"
download_study "Pezzo_2f7d3d21_raw_merged"
download_study "KiddWeaver_97abc24d_raw_merged"
download_study "dEntremont_831030a3_raw_merged"
download_study "Schweitzer_5075cbbc_raw_merged"
download_study "wilson2018_sea_lion_telemetry"
download_study "McConnell_48f3a4ac_raw_merged"
download_study "Lamb_cec3bcf7_raw_merged"
download_study "krbal_a8224f36_raw_merged"
download_study "Fiedler_lifetrack_stork_all"
download_study "Batbayar_1e31df42_raw_merged"
download_study "klappstein2024_smoothSSF_data_petrel"
download_study "Efrat_9d8e2a5a_raw_merged"
download_study "vanderJeugd_181634a2_raw_merged"
download_study "Batbayar_26c82518_raw_merged"
