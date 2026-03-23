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

download_study "Chynoweth_161b478c_raw_merged"
download_study "amtdeer_wgs84"
download_study "Schloesing_a51b51e0_raw_merged"
download_study "Carter_28708a9c_raw_merged"
download_study "Isbell_5f9fad19_raw_merged"
download_study "Prugh_e7808391_raw_merged_Canis_latrans"
download_study "HernandezPliego_225f9132_raw_merged"
download_study "Naniwadekar_35a868e5_raw_merged_Buceros_bicornis"
download_study "Power_b06f7d4d_raw_merged"
download_study "Patrcio_c2bfb336_raw_merged"
download_study "Choden_e3343be6_raw_merged"
download_study "tsavo_lions"
download_study "cazzi2016_hyena"
download_study "Garthe_335a3479_raw_merged"
download_study "killian_elephants"
download_study "Christopher_cd4e676d_raw_merged_Sarcoramphus_papa"
download_study "BartlamBrooks_c9b4f9b5_raw_merged"
download_study "Geary_a398b445_raw_merged"
download_study "Klzsch_a61cb8c5_raw_merged_Anser_brachyrhynchus"
download_study "Spiegel_77b4dd70_raw_merged_Gyps_africanus"
download_study "Lago_8b91498b_raw_merged"
download_study "Hays_df53968d_raw_merged"
download_study "Friedemann_c92d9814_raw_merged_Buteo_rufinus"
download_study "Ilyashenko_d52006ff_raw_merged"
download_study "Pokrovsky_a1d2d3ab_raw"
download_study "Cabot_2769c15e_raw_merged"
download_study "Bildstein_3e431f5b_raw_merged_Cathartes_aura"
download_study "Klzsch_22d42221_2e9dace9_all"
download_study "Schwemmer_aa260e37_raw_merged"
download_study "Eby_c5d8e38f_raw_merged"
