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
    "Serieys_7de63cc4_raw_merged"
    "Graf_e3df6f82_raw_merged"
    "Toivonen_338388ea_raw_merged"
    "Buitendijk_8318572a_raw_merged_Anser_anser"
    "WarretRodrigues_5a280544_raw_merged_Vulpes_vulpes"
    "Boyce_3a171441_raw_merged"
    "Scasta_2facc53d_raw_merged"
    "Hebblewhite_dce434f5_raw_merged"
    "Prugh_e7808391_raw_merged_Lynx_rufus"
    "Power_6360747f_raw_merged"
    "Glass_fa3e16ed_raw_merged"
    "Latham_c0de801a_raw_merged"
    "Klzsch_a61cb8c5_raw_merged_Branta_leucopsis"
    "Setyawan_0425c92c_raw_merged"
    "Ronconi_70744563_raw_merged"
    "Pezzo_2f7d3d21_raw_merged"
    "KiddWeaver_97abc24d_raw_merged"
    "dEntremont_831030a3_raw_merged"
    "Schweitzer_5075cbbc_raw_merged"
    "wilson2018_sea_lion_telemetry"
    "McConnell_48f3a4ac_raw_merged"
    "Lamb_cec3bcf7_raw_merged"
    "krbal_a8224f36_raw_merged"
    "Fiedler_lifetrack_stork_all"
    "Batbayar_1e31df42_raw_merged"
    "klappstein2024_smoothSSF_data_petrel"
    "Efrat_9d8e2a5a_raw_merged"
    "vanderJeugd_181634a2_raw_merged"
    "Batbayar_26c82518_raw_merged"
)

for layers_config in "${layers_configs[@]}"; do
    for study_id in "${study_ids[@]}"; do
        echo "Downloading ${study_id} (${layers_config##*/})"
        movebench download --layers-config "${layers_config}" --studies "${study_id}" "${extra_args[@]}"
    done
done
