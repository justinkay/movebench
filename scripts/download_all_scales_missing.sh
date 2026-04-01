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
    "BastilleRousseau_12f99da9_raw_merged_Chelonoidis_hoodensis"
    "BastilleRousseau_12f99da9_raw_merged_Chelonoidis_porteri"
    "BastilleRousseau_9eaf99f2_raw_merged_Chelonoidis_donfaustoi"
    "BastilleRousseau_9eaf99f2_raw_merged_Chelonoidis_hoodensis"
    "BastilleRousseau_9eaf99f2_raw_merged_Chelonoidis_porteri"
    "BastilleRousseau_f9b4b052_raw_merged"
    "Fischer_b624221e_raw_merged"
    "Guerra_71fc09c6_raw_merged"
    "Hays_bd9f2614_raw_merged_Eretmochelys_imbricata"
    "Kays_c73c7698_raw_merged_Ateles_geoffroyi"
    "Kays_c73c7698_raw_merged_Cebus_capucinus"
    "Kays_c73c7698_raw_merged_Nasua_narica"
    "Kays_c73c7698_raw_merged_Pecari_tajacu"
    "Kays_c73c7698_raw_merged_Potos_flavus"
    "Kays_c8aac319_raw_merged"
    "Khaewphakdee_bc7febf2_raw_merged"
    "Kikillus_75d6171c_raw_merged"
    "McMahon_0bd4ff73_raw_merged"
    "Mertes_a41015db_raw_merged"
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
