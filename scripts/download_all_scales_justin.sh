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

download_study "Medici_e2fe5582_raw_merged_Tapirus_terrestris"
download_study "Wall_38e05433_raw_merged"
download_study "Naniwadekar_35a868e5_raw_merged_Rhyticeros_undulatus"
download_study "Roshier_f7b38cdd_raw_merged"
download_study "Morato_8fed4a87_raw_merged"
download_study "forrest2024_buffalo"
download_study "Kleyheeg_64a804f1_raw_merged"
download_study "Borrmann_7528b6f8_raw_merged"
download_study "Cross_4917e970_raw_merged"
download_study "Beck_0b812dab_raw_merged"
download_study "hooten2018_mountain_lion_AM80"
download_study "Slotow_b777279a_raw_merged"
download_study "Mallory_91804022_raw_merged"
download_study "Hebblewhite_fb867768_raw_merged"
download_study "Irvine_4c7988df_raw_merged"
download_study "Christopher_cd4e676d_raw_merged_Cathartes_melambrotus"
download_study "Fifield_4362d62f_raw_merged"
download_study "Lidgard_1a65b7b8_raw_merged"
download_study "Sherub_5672a811_raw_merged"
download_study "Spiegel_5935dd02_raw_merged_Torgos_tracheliotus"
download_study "Bildstein_3e431f5b_raw_merged_Coragyps_atratus"
download_study "Jeglinski_12a063ce_raw_merged"
download_study "ydelis_346009bb_raw_merged"
download_study "Ronconi_4f193d93_raw_merged"
download_study "vanToor_86f824b7_raw_merged"
download_study "Ilyashenko_e82ecc51_raw_merged"
download_study "Friedemann_c92d9814_raw_merged_Circaetus_gallicus"
download_study "Piironen_cd62a27e_raw_merged"
download_study "Griffin_644e46c6_raw_merged"
download_study "Byholm_10bb3544_raw_merged"
