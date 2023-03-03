#!/bin/bash
#$ -cwd
#$ -l f_node=2
#$ -l h_rt=24:00:00
#$ -N no_raft_2000ep_bdd100k_pretrain_pixpro
#$ -j y
#$ -V


set -x

echo "start scirpt file1 cat"
cat "$0"

set +x

echo "end scirpt file1 cat"

START_TIMESTAMP=$(date '+%s')

# ======== Args ========

epoch=${EPOCH:-0}
bs=${BS:-128}
pixpro_pos_ratio=${POS_RATIO:-0.7}
n_frame=${N_FRAME:-1}
opt_level=${OPT_LEV:-"O1"}
pixpro_frames=${PIXPRO_FRAMES:-1}

cur_rel=${CUR_REL:-"n"}

base_script=${SCRIPT_NAME:-"./pretrain_bdd100k_job_base.sh"}

# previous logfile
pre_logfile=${LOGFILE:-"nframe2_of_bdd100k_pretrain_pixpro.o101010"}

# ======== Variables ========

## read only variables
export ALL_EPOCH=2000
# for raft
export USE_FLOW="n"

## changeable variables
export EPOCH=$epoch
export BS=$bs
export POS_RATIO=$pixpro_pos_ratio
export N_FRAME=$n_frame
export OPT_LEV="$opt_level"
export PIXPRO_FRAMES="$pixpro_frames"

export CUR_REL="$cur_rel"
export LOGFILE="$pre_logfile"


# ======== Scripts ========

bash "$base_script"

END_TIMESTAMP=$(date '+%s')

E_TIME=$(($END_TIMESTAMP-$START_TIMESTAMP))
echo "exec time: $E_TIME s"

