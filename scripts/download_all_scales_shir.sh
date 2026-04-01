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
    "Chynoweth_161b478c_raw_merged"
    "amtdeer_wgs84"
    "Schloesing_a51b51e0_raw_merged"
    "Carter_28708a9c_raw_merged"
    "Isbell_5f9fad19_raw_merged"
    "Prugh_e7808391_raw_merged_Canis_latrans"
    "HernandezPliego_225f9132_raw_merged"
    "Naniwadekar_35a868e5_raw_merged_Buceros_bicornis"
    "Power_b06f7d4d_raw_merged"
    "Patrcio_c2bfb336_raw_merged"
    "Choden_e3343be6_raw_merged"
    "tsavo_lions"
    "cazzi2016_hyena"
    "Garthe_335a3479_raw_merged"
    "killian_elephants"
    "Christopher_cd4e676d_raw_merged_Sarcoramphus_papa"
    "BartlamBrooks_c9b4f9b5_raw_merged"
    "Geary_a398b445_raw_merged"
    "Klzsch_a61cb8c5_raw_merged_Anser_brachyrhynchus"
    "Spiegel_77b4dd70_raw_merged_Gyps_africanus"
    "Lago_8b91498b_raw_merged"
    "Hays_df53968d_raw_merged"
    "Friedemann_c92d9814_raw_merged_Buteo_rufinus"
    "Ilyashenko_d52006ff_raw_merged"
    "Pokrovsky_a1d2d3ab_raw"
    "Cabot_2769c15e_raw_merged"
    "Bildstein_3e431f5b_raw_merged_Cathartes_aura"
    "Klzsch_22d42221_2e9dace9_all"
    "Schwemmer_aa260e37_raw_merged"
    "Eby_c5d8e38f_raw_merged"
    "Morganti_8371ef00_raw_merged"
    "Newman_e7d1c177_raw_merged"
    "Palomares_5e487e1e_raw_merged"
    "Porteus_86f58f48_raw_merged"
    "Roetman_0a9bcb2a_raw_merged"
    "Serieys_ed70c4c7_raw_Lynx_rufus"
    "Serieys_ed70c4c7_raw_Urocyon_cinereoargenteus"
    "Todd_3cef8e9d_raw_merged"
    "Wagner_6f734b0f_raw_merged"
    "egan2025_deer"
    "potts2022_pig"
)

for layers_config in "${layers_configs[@]}"; do
    for study_id in "${study_ids[@]}"; do
        echo "Downloading ${study_id} (${layers_config##*/})"
        movebench download --layers-config "${layers_config}" --studies "${study_id}" "${extra_args[@]}"
    done
done
