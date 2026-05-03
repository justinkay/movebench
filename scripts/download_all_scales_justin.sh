#!/usr/bin/env bash

set -euo pipefail
umask 0002

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
    "Medici_e2fe5582_raw_merged_Tapirus_terrestris"
    "Wall_38e05433_raw_merged"
    "Naniwadekar_35a868e5_raw_merged_Rhyticeros_undulatus"
    "Roshier_f7b38cdd_raw_merged"
    "Morato_8fed4a87_raw_merged"
    "forrest2024_buffalo"
    "Kleyheeg_64a804f1_raw_merged"
    "Borrmann_7528b6f8_raw_merged"
    "Cross_4917e970_raw_merged"
    "Beck_0b812dab_raw_merged"
    "hooten2018_mountain_lion_AM80"
    "Slotow_b777279a_raw_merged"
    "Mallory_91804022_raw_merged"
    "Hebblewhite_fb867768_raw_merged"
    "Irvine_4c7988df_raw_merged"
    "Christopher_cd4e676d_raw_merged_Cathartes_melambrotus"
    "Fifield_4362d62f_raw_merged"
    "Lidgard_1a65b7b8_raw_merged"
    "Sherub_5672a811_raw_merged"
    "Spiegel_5935dd02_raw_merged_Torgos_tracheliotus"
    "Bildstein_3e431f5b_raw_merged_Coragyps_atratus"
    "Jeglinski_12a063ce_raw_merged"
    "ydelis_346009bb_raw_merged"
    "Ronconi_4f193d93_raw_merged"
    "vanToor_86f824b7_raw_merged"
    "Ilyashenko_e82ecc51_raw_merged"
    "Friedemann_c92d9814_raw_merged_Circaetus_gallicus"
    "Griffin_644e46c6_raw_merged"
    "Byholm_10bb3544_raw_merged"
)
# removed piironen - was taking forever

for layers_config in "${layers_configs[@]}"; do
    for study_id in "${study_ids[@]}"; do
        echo "Downloading ${study_id} (${layers_config##*/})"
        movebench download gee --layers-config "${layers_config}" --studies "${study_id}" "${extra_args[@]}"
    done
done
