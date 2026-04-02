#!/usr/bin/env bash

set -euo pipefail

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    echo "Usage: $0 [shared_root]"
    echo
    echo "Repairs shared download permissions on files you own."
    echo "Run this without sudo on the NFS-mounted archive."
    exit 0
fi

shared_root="${1:-/archive/vision/beery/movebench/studies}"
current_uid="$(id -u)"
current_user="$(id -un)"

if [[ ! -d "$shared_root" ]]; then
    echo "Missing directory: $shared_root" >&2
    exit 1
fi

count_matches() {
    find "$shared_root" -uid "$current_uid" "$@" | wc -l | tr -d '[:space:]'
}

# echo "Counting files owned by ${current_user} under ${shared_root}"
# owned_bad_files_before="$(count_matches -type f \( ! -perm -g+r -o ! -perm -g+w \))"
# echo "Owned bad files: ${owned_bad_files_before}"

echo "Counting directories owned by ${current_user} under ${shared_root}"
owned_bad_dirs_before="$(count_matches -type d \( ! -perm -g+s -o ! -perm -g+w -o ! -perm -g+x \))"
echo "Owned bad directories: ${owned_bad_dirs_before}"

echo "Repairing files owned by ${current_user} under ${shared_root}"
find "$shared_root" -uid "$current_uid" -type f \( ! -perm -g+r -o ! -perm -g+w \) -exec chmod 664 {} +

echo "Repairing directories owned by ${current_user} under ${shared_root}"
find "$shared_root" -uid "$current_uid" -type d \( ! -perm -g+s -o ! -perm -g+w -o ! -perm -g+x \) -exec chmod 2775 {} +

echo "Scanning for remaining bad files for ${current_user} under ${shared_root}"
owned_bad_files_after="$(count_matches -type f \( ! -perm -g+r -o ! -perm -g+w \))"
owned_bad_dirs_after="$(count_matches -type d \( ! -perm -g+s -o ! -perm -g+w -o ! -perm -g+x \))"
remaining_bad_files="$(find "$shared_root" -type f \( ! -perm -g+r -o ! -perm -g+w \) | wc -l | tr -d '[:space:]')"

echo
# echo "Owned files fixed: ${owned_bad_files_before}"
# echo "Owned directories fixed: ${owned_bad_dirs_before}"
echo "Owned files still missing group read/write: ${owned_bad_files_after}"
echo "Owned directories still missing setgid/group write/execute: ${owned_bad_dirs_after}"

if [[ "$remaining_bad_files" != "0" ]]; then
    echo
    echo "Some files still need repair by their owners or by the NFS server admin:"
    find "$shared_root" -type f \( ! -perm -g+r -o ! -perm -g+w \) -printf '%u %g %m %p\n' | sed -n '1,20p'
fi
