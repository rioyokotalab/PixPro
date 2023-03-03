#!/bin/bash
#$ -cwd
#$ -l f_node=2
#$ -l h_rt=24:00:00
#$ -N nframe2_of_bdd100k_pretrain_pixpro
#$ -j y
#$ -V


set -x

echo "start scirpt file cat"
cat "$0"

set +x

echo "end scirpt file cat"

START_TIMESTAMP=$(date '+%s')

# ======== Args ========

all_epoch=${ALL_EPOCH:-2000} # whole epoch number
epoch=${EPOCH:-0}
bs=${BS:-128}
pixpro_pos_ratio=${POS_RATIO:-0.7}
n_frame=${N_FRAME:-2}
opt_level=${OPT_LEV:-"O1"}
pixpro_frames=${PIXPRO_FRAMES:-1}

# for raft
is_use_flow=${USE_FLOW:-"y"}
raft_name=${RAFT_NAME:-"small"}
is_mask=${USE_MASK:-"y"}
alpha1=${ALPHA1:-0.01}
alpha2=${ALPHA2:-0.5}
flow_bs=${FLOW_BS:-2}
is_use_flow_frames=${FLOW_FRAMES:-"n"}
is_use_flow_files=${FLOW_FILES:-"y"}
flow_up=${FLOW_UP:-"y"}

cur_rel=${CUR_REL:-"n"}

# previous logfile
pre_logfile=${LOGFILE:-"nframe2_of_bdd100k_pretrain_pixpro.o101010"}

# ======== Variables ========

job_id_base=$JOB_ID

git_root=$(git rev-parse --show-toplevel | head -1)
# base_root=$(basename "$git_root")

log_file="$JOB_NAME.o$job_id_base"


# ======== Pyenv ========

export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init --path)"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"
# pipenv property
export PIPENV_VENV_IN_PROJECT=1
export PIPENV_IGNORE_VIRTUALENVS=1
# which python
# pyenv versions

# ======== Modules ========

source /etc/profile.d/modules.sh

module load gcc/8.3.0
module load cmake/3.21.3

module load cuda/10.2.89
module load cudnn/8.1

module load nccl/2.8.4
module load openmpi/3.1.4-opa10.10
# module load openmpi/3.1.4-opa10.10-t3

module list

# ======== MPI ========

nodes=$NHOSTS
gpus_pernode=4
cpus_pernode=28
gpus=$(($nodes * $gpus_pernode))
cpus=$(($nodes * $cpus_pernode))

echo "gpus: $gpus"
echo "gpus per node $gpus_pernode"

MASTER_ADDR=`head -n 1 $SGE_JOB_SPOOL_DIR/pe_hostfile | cut -d " " -f 1`
MASTER_PORT=$((10000 + ($job_id_base % 50000)))

MPI_OPTS="-np $gpus"
MPI_OPTS+=" -npernode $gpus_pernode"
MPI_OPTS+=" -mca btl self,tcp"
MPI_OPTS+=" -x MASTER_ADDR=$MASTER_ADDR"
MPI_OPTS+=" -x MASTER_PORT=$MASTER_PORT"
MPI_OPTS+=" -x NCCL_BUFFSIZE=1048576"
MPI_OPTS+=" -x NCCL_IB_DISABLE=1"
MPI_OPTS+=" -x NCCL_IB_TIMEOUT=14"
MPI_OPTS+=" -x PATH"
MPI_OPTS+=" -x LD_LIBRARY_PATH"
MPI_OPTS+=" -x PSM2_GPUDIRECT=1"
MPI_OPTS+=" -x PSM2_CUDA=1"
# MPI_OPTS+=" $git_root/data/jobs/add_optical_flow/warp.sh"

mpi_backend="nccl"
# mpi_backend="mpi"
# mpi_backend="gloo"

# ======= Model options ========

epoch_str="$all_epoch""ep"

# raft_name="chairs"
# raft_name="kitti"
# raft_name="sintel"
# raft_name="things"
# raft_name=""
# raft_name=${RAFT_NAME:-""}

# flow_bs=2

flow_model_path="$git_root/data/pretrained_flow/models/raft-$raft_name.pth"

raft_str="$raft_name/"

# raft_str+="measure_time/"

raft_opts="--flow_model $flow_model_path"
raft_opts+=" --n-frames $n_frame"
raft_opts+=" --flow_bs $flow_bs"
raft_opts+=" --pixpro-frames $pixpro_frames"

if [ "$is_mask" = "y" ];then
    raft_opts+=" --alpha1 $alpha1"
    raft_opts+=" --alpha2 $alpha2"
    raft_str+="alpha1_$alpha1""_alpha2_""$alpha2/"
else
    raft_str+="no_mask/"
fi

if [ "$is_use_flow_frames" = "y" ];then
    raft_opts+=" --use_flow_frames"
    raft_str+="use_flow_frames/"
fi

if [ "$is_use_flow_files" = "y" ];then
    raft_opts+=" --use_flow_file"
    raft_str+="use_flow_file/"
fi

# flow_up=""

if [ "$flow_up" = "y" ];then
    raft_opts+=" --flow_up"
    raft_str+="up/"
fi

if [ "$is_use_flow" = "y" ]; then
    raft_opts+=" --use_flow"
else
    raft_str="no_raft/"
fi

raft_str+="flow_bs_$flow_bs/n_frame_$n_frame"

raft_str+="/bs_$bs"

is_debug="n"
if [ $epoch -gt 0 ];then
    is_debug="y"
fi
if [ "$is_debug" = "y" ];then
    raft_opts+=" --debug-epochs $epoch"
fi

branch_name=$(git rev-parse --abbrev-ref HEAD)
commit_id=$(git rev-parse --short HEAD)
# commit_id=${commit_id:0:7}

if [ "$branch_name" = "HEAD" ];then
    tag_name=$(git log --oneline --decorate | head -1 | grep "tag:")
    if [ -n "$tag_name" ];then
        tag_name=$(git describe --tags)
        # raft_str+="/$tag_name"
    else
        tag_name="no_tagname"
    fi
    branch_name="$branch_name""_tag_$tag_name""_commit_id_$commit_id"
fi

# pixpro_pos_ratio=2
# pixpro_pos_ratio=$(echo "scale=20; sqrt("$pixpro_pos_ratio")" | bc)
# pixpro_pos_ratio=$(echo "scale=20; 1 / sqrt("$pixpro_pos_ratio")" | bc)

raft_str="pixpro_frames_$pixpro_frames/pos_ratio_$pixpro_pos_ratio/$raft_str"

date_str=$(date '+%Y%m%d_%H%M%S')
cur_root="$git_root/output"

is_resume="n"
if [ "$cur_rel" = "n" ];then
    cur_rel="pixpro_base_r50_$epoch_str/bdd100k/$branch_name/commit_id_$commit_id/$opt_level/ratio_default/image_size_224/$raft_str/$date_str"
else
    is_resume="y"
fi

cur_out="$cur_root/$cur_rel"
# commot_out="$git_root/output/$date_str"
commot_out="$cur_out/$date_str"
git_out="$commot_out/git_out"
script_out="$commot_out/scripts"
config_stash_out="$commot_out/prev_config_stash"

convert_d2_script="$git_root/transfer/detection/convert_pretrain_to_d2.py"

# ======== Scripts ========

pushd "$git_root"

set -x

if [ "$is_resume" = "y" ];then
    if [ -f "$pre_logfile" ];then
        if [ ! -d "$cur_out/job_logs" ];then
            mkdir -p "$cur_out/job_logs"
        fi
        if [ ! -f "$cur_out/job_logs/$pre_logfile" ];then
            mv "$pre_logfile" "$cur_out/job_logs"
        fi
    fi
fi

end_epoch=$all_epoch
if [ "$is_debug" = "y" ];then
    end_epoch=$epoch
fi
if [ -f "$cur_out/ckpt_epoch_$end_epoch.pth" ];then
    exit 0
fi

mkdir -p "$git_out"
mkdir -p "$script_out"
mkdir -p "$config_stash_out"
# cp "data/jobs/of/pretrain_bdd100k_job_100ep.sh" "$script_out"
script_name=$(basename "$0")
cat "$0" > "$script_out/$script_name.sh"

git status | tee "$git_out/git_status.txt"
# git log | tee "$git_out/git_log.txt"
git log > "$git_out/git_log.txt"
git diff HEAD | tee "$git_out/git_diff.txt"
git rev-parse HEAD | tee "$git_out/git_head.txt"

if [ -f "$cur_out/config.json" ];then
    cat "$cur_out/config.json" > "$config_stash_out/config.json"
fi

data_dir="./data/bdd100k/images"
# data_dir="/beegfs/z44812z/bdd100k_root/bdd100k/images"
# if [ "$is_use_flow_files" = "y" ];then
#     flow_root="./data/bdd100k/flow/pth"
#     raft_opts+=" --flow_root $flow_root"
# fi

mpirun ${MPI_OPTS} \
    python main_pretrain_mpirun.py \
    --data-dir ${data_dir} \
    --output-dir ${cur_out} \
    \
    --cache-mode no \
    --crop 0.08 \
    --aug BYOL \
    --dataset bdd100k \
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
    --epochs $all_epoch \
    --amp-opt-level "$opt_level" \
    \
    --save-freq 1 \
    --print-freq 1 \
    --auto-resume \
    \
    --pixpro-p 2 \
    --pixpro-momentum 0.99 \
    --pixpro-pos-ratio $pixpro_pos_ratio \
    --pixpro-transform-layer 1 \
    --pixpro-ins-loss-weight 0. \
    --local_rank 0 \
    \
    ${raft_opts}
    # --zip \
    # \
    # --log_name "$log_file" \


convert_d2_models_path="$cur_out/convert_d2_models"
mkdir -p "$convert_d2_models_path"
c_epoch_list="10 100 200 500 1000 1500 2000"
for c_epoch in ${c_epoch_list};
do
    if [ ! -f "$convert_d2_models_path/ckpt_epoch_$c_epoch.pkl" ];then
        if [ -f "$cur_out/ckpt_epoch_$c_epoch.pth" ];then
            python "$convert_d2_script" "$cur_out/ckpt_epoch_$c_epoch.pth" "$convert_d2_models_path/ckpt_epoch_$c_epoch.pkl"
        fi
    fi
done

popd

END_TIMESTAMP=$(date '+%s')

E_TIME=$(($END_TIMESTAMP-$START_TIMESTAMP))
echo "exec time: $E_TIME s"

# cur_out=$(find "$commot_out/pixpro_base_r50_100ep" -type d -maxdepth 1 -mindepth 1 | sort | tail -1)

mkdir -p "$cur_out/job_logs"
cp "$log_file" "$cur_out/job_logs"

# mv "$git_out" "$cur_out"
# rmdir "$commot_out"
# qstat
