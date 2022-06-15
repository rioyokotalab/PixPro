#!/bin/bash

set -e
set -x

job_id_base=$PJM_JOBID
nodes=$PJM_NODE
gpus_pernode=${PJM_PROC_BY_NODE}
gpus=${PJM_MPI_PROC}

MASTER_ADDR=$(cat "$PJM_O_NODEINF" | head -1)
MASTER_PORT=$((10000 + ($job_id_base % 50000)))

# date_str=$(date '+%Y%m%d_%H%M%S')
data_dir="./data/imagenet/"
# output_dir="./output/pixpro_base_r50_400ep/$date_str"
output_dir="./output/pixpro_base_r50_400ep/20220516_222227"

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --master_port 12348 --nproc_per_node=8 \
mpirun \
    -machinefile $PJM_O_NODEINF \
    -np $PJM_MPI_PROC \
    -npernode $PJM_PROC_BY_NODE \
    -x MASTER_ADDR=$MASTER_ADDR \
    -x MASTER_PORT=$MASTER_PORT \
    -x NCCL_BUFFSIZE=1048576 \
    python main_pretrain_mpirun.py \
    --data-dir ${data_dir} \
    --output-dir ${output_dir} \
    \
    --cache-mode no \
    --crop 0.08 \
    --aug BYOL \
    --dataset ImageNet \
    --batch-size 16 \
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
    --local_rank 0
    # --zip \

