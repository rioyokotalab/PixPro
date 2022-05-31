#!/bin/bash

set -e
set -x

date_str=$(date '+%Y%m%d_%H%M%S')
bs=128
# bs=16
dataset_name="bdd100k/images"
# dataset_name="imagenet"
data_dir="./data/$dataset_name"
# aug="BYOL"
aug="myBYOL"
# aug="SimCLR"
# aug="mySimCLR"
output_dir="./output/pixpro_base_r50_100ep/$dataset_name/$bs/$aug/$date_str"
# output_dir="./output/pixpro_base_r50_100ep/20220517_224459"
dataset_type="bdd100k"
# dataset_type="ImageNet"

job_id_base=$JOB_ID
MASTER_ADDR=$(/usr/sbin/ip a show dev bond0 | grep 'inet ' | cut -d " " -f 6 | cut -d "/" -f 1)
MASTER_PORT=$((10000 + ($job_id_base % 50000)))

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --master_addr $MASTER_ADDR --master_port $MASTER_PORT --nproc_per_node=8 \
    main_pretrain.py \
    --data-dir ${data_dir} \
    --output-dir ${output_dir} \
    \
    --cache-mode no \
    --crop 0.08 \
    --aug "$aug" \
    --dataset "$dataset_type" \
    --batch-size $bs \
    \
    --model PixPro \
    --arch resnet50 \
    --head-type early_return \
    \
    --optimizer lars \
    --base-lr 1.0 \
    --weight-decay 1e-5 \
    --warmup-epoch 5 \
    --epochs 100 \
    --amp-opt-level O1 \
    \
    --print-freq 1 \
    --save-freq 1 \
    --auto-resume \
    \
    --pixpro-p 2 \
    --pixpro-momentum 0.99 \
    --pixpro-pos-ratio 0.7 \
    --pixpro-transform-layer 1 \
    --pixpro-ins-loss-weight 0. \
    # --zip \

