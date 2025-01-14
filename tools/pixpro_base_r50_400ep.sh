#!/bin/bash

set -e
set -x

date_str=$(date '+%Y%m%d_%H%M%S')
data_dir="./data/imagenet/"
output_dir=${1:-"./output/pixpro_base_r50_400ep/$date_str"}

MASTER_ADDR=$2
MASTER_PORT=$3

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --master_addr $MASTER_ADDR --master_port $MASTER_PORT --nproc_per_node=8 \
    main_pretrain.py \
    --data-dir ${data_dir} \
    --output-dir ${output_dir} \
    \
    --cache-mode no \
    --crop 0.08 \
    --aug BYOL \
    --dataset ImageNet \
    --batch-size 128 \
    \
    --model PixPro \
    --arch resnet50 \
    --head-type early_return \
    \
    --optimizer lars \
    --base-lr 1.0 \
    --weight-decay 1e-5 \
    --warmup-epoch 5 \
    --epochs 400 \
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

