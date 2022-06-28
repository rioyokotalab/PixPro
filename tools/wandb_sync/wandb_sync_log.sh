#!/bin/bash
set -ex

git_root=$(git rev-parse --show-toplevel | head -1)
python_path="$git_root/tools/wandb_sync/wandb_sync_log.py"

out_root=${1:-"$git_root/output"}
out_root=$(
    cd "$out_root" || exit
    pwd
)

target_list=$(find "$out_root" -name "current.pth" | sort)
# target_list=$(find "$out_root" -name "current.pth" | sort | head -1)
wandb_project_name=${2:-"PixPro"}

set +x
num_target=$(echo "$target_list" | wc -l)
set -x

echo "$num_target"

pushd "$out_root"

for t_f_path in ${target_list};
do
    t_path=$(dirname "$t_f_path")
    wandb_id_list=""
    python "$python_path" --project "$wandb_project_name" --target_path "$t_path" > "wandb_id.log"
    cat "wandb_id.log" >> "wandb_id_all.log"

    set +x
    wandb_ids=$(cat "wandb_id.log" | grep "wandb_id:")
    local_ts=$(cat "wandb_id.log" | grep "wandb_dir:")
    local_tf_name=$(cat "wandb_id.log" | grep "tf_name:")
    tf_log_num=$(echo "$wandb_ids" | wc -l)
    set -x

    for i in $(seq 1 ${tf_log_num});
    do
        set +x
        local_wandb_id=$(echo "$wandb_ids" | head -"$i")
        local_t=$(echo "$local_ts" | head -"$i")
        local_tf_log=$(echo "$local_tf_name" | head -"$i")
        set -x
        local_wandb_id=${local_wandb_id##* }
        local_t_path=${local_t##* }
        local_tf_log=${local_tf_log##* }
        local_wandb_id=$(echo ${local_wandb_id} | sed -e "s/[\r\n]\+//g")
        cp "$local_tf_log" "$local_t_path"
        # echo "$local_wandb_id $local_t_path"
        tf_log=$(find "$local_t_path" -name "events.*")
        if [ -f "$tf_log" ];then
            wandb sync -p "$wandb_project_name" --id "$local_wandb_id" "$tf_log" 
        fi
        wandb_id_list+="$local_wandb_id "
    done
    python "$python_path" --project "$wandb_project_name" --target_path "$t_path" --upload --ids $wandb_id_list
done

popd
