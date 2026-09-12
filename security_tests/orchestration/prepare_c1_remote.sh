#!/usr/bin/env bash
set -euo pipefail

umask 077
base=/home/cc/red_team_sanity_20260809/c1
public_dir=$base/public_attack
private_dir=$base/affine_private
tools_dir=$public_dir/tools

trap 'printf "%s\n" PREPARE_FAILED >&2' ERR

cp -- "$public_dir/c1_final.txt" "$private_dir/c1_final.txt"
cp -- /home/cc/bin/kissat "$tools_dir/kissat"

chmod 700 "$base" "$public_dir" "$private_dir" "$tools_dir"
chmod 600 \
    "$public_dir/c1_final.txt" \
    "$public_dir/challenge.json" \
    "$tools_dir/mpmct1_zero_slice_to_cnf.cpp" \
    "$private_dir/c1_final.txt" \
    "$private_dir/source_c.g57"
chmod 700 \
    "$tools_dir/decode_mpmct1_zero_slice_model.py" \
    "$tools_dir/run_c1_public_attack_remote.py" \
    "$tools_dir/kissat" \
    "$private_dir/hmap_affine" \
    "$private_dir/run_c1_affine_remote.sh"

g++ -O3 -DNDEBUG -std=c++17 -pipe \
    "$tools_dir/mpmct1_zero_slice_to_cnf.cpp" \
    -o "$tools_dir/mpmct1_zero_slice_to_cnf"
chmod 700 "$tools_dir/mpmct1_zero_slice_to_cnf"

check_hash() {
    local expected=$1
    local path=$2
    local actual
    actual=$(sha256sum -- "$path")
    actual=${actual%% *}
    [[ $actual == "$expected" ]]
}

check_hash 13ef54d8128d092f0f01c203270041473aad1337c13c8fa89a54d8f1b3d8c52a "$public_dir/c1_final.txt"
check_hash f1dfa06df65ce2cecc7f2dfd28f93926167e47f16c178e2fbfb5a5b0ed772cad "$public_dir/challenge.json"
check_hash c385c7ccf9dec882d7e829bdbec6f73311fac4e820b4675108fe9ae99dd530db "$tools_dir/mpmct1_zero_slice_to_cnf.cpp"
check_hash 6dba5ac500d5e651aaa23332e4d447fe56d51c7e38622b8891150111fab4d54c "$tools_dir/decode_mpmct1_zero_slice_model.py"
check_hash b437c145e9b21ca922055fb93c3e7d4b8b4bcc1a67e7b8b7e518c6442608c70b "$tools_dir/run_c1_public_attack_remote.py"
check_hash be2038d2cf2e664e91d7d3397347309d522cd7829167b62044a29bc1eaf354db "$tools_dir/kissat"
check_hash 4bfc6d81f3e12f5b336949f2e5952a904762fc91635358fc29cb5251d4bdf153 "$private_dir/source_c.g57"
check_hash 13ef54d8128d092f0f01c203270041473aad1337c13c8fa89a54d8f1b3d8c52a "$private_dir/c1_final.txt"
check_hash 14e160d72b9409a38b3012d3545eb056b5ed940bbe71207e33740b2bdcfd86ad "$private_dir/hmap_affine"
check_hash 600895861c30d288ede799a40171cd6ae8dbaf5398e78495454dec81b47c62fa "$private_dir/run_c1_affine_remote.sh"

public_entries=$(find "$public_dir" -mindepth 1 -maxdepth 1 -printf '%f\n' | LC_ALL=C sort)
[[ $public_entries == $'c1_final.txt\nchallenge.json\ntools' ]]
tool_entries=$(find "$tools_dir" -mindepth 1 -maxdepth 1 -printf '%f\n' | LC_ALL=C sort)
[[ $tool_entries == $'decode_mpmct1_zero_slice_model.py\nkissat\nmpmct1_zero_slice_to_cnf\nmpmct1_zero_slice_to_cnf.cpp\nrun_c1_public_attack_remote.py' ]]
if find "$public_dir" -type f \( -iname '*seed*' -o -iname '*answer*' -o -name 'source_c.g57' \) -print -quit | grep -q .; then
    printf '%s\n' PUBLIC_WHITELIST_FAILED >&2
    exit 1
fi

target=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["target_logical_output_hex"])' "$public_dir/challenge.json")
"$tools_dir/mpmct1_zero_slice_to_cnf" \
    "$public_dir/c1_final.txt" /dev/null 128 "$target" --analyze-only \
    >"$public_dir/encoder_analysis.log" 2>&1
chmod 600 "$public_dir/encoder_analysis.log"
grep -q 'vars=7469493 clauses=58331926$' "$public_dir/encoder_analysis.log"

printf 'public_final_sha256=%s\n' 13ef54d8128d092f0f01c203270041473aad1337c13c8fa89a54d8f1b3d8c52a
printf 'challenge_sha256=%s\n' f1dfa06df65ce2cecc7f2dfd28f93926167e47f16c178e2fbfb5a5b0ed772cad
printf 'private_source_sha256=%s\n' 4bfc6d81f3e12f5b336949f2e5952a904762fc91635358fc29cb5251d4bdf153
printf 'hmap_affine_sha256=%s\n' 14e160d72b9409a38b3012d3545eb056b5ed940bbe71207e33740b2bdcfd86ad
printf 'encoder_source_sha256=%s\n' c385c7ccf9dec882d7e829bdbec6f73311fac4e820b4675108fe9ae99dd530db
printf 'decoder_sha256=%s\n' 6dba5ac500d5e651aaa23332e4d447fe56d51c7e38622b8891150111fab4d54c
printf 'encoder_binary_sha256='
sha256sum -- "$tools_dir/mpmct1_zero_slice_to_cnf" | cut -d' ' -f1
printf '%s\n' 'cnf_expected_header=p cnf 7469493 58331926'
printf '%s\n' PREPARE_OK
