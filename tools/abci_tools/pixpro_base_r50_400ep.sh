#!/bin/bash

set -e
set -x

git_root=$(git rev-parse --show-toplevel | head -1)
date_str=$(date '+%Y%m%d_%H%M%S')
output_dir=${1:-"./output/pixpro_base_r50_400ep/$date_str"}

job_id_base=$JOB_ID
MASTER_ADDR=$(/usr/sbin/ip a show dev bond0 | grep 'inet ' | cut -d " " -f 6 | cut -d "/" -f 1)
MASTER_PORT=$((10000 + ($job_id_base % 50000)))

pushd "$git_root"

./tools/pixpro_base_r50_400ep.sh "$output_dir" $MASTER_ADDR $MASTER_PORT

popd
