import argparse
import os
import sys
import glob
from collections import defaultdict
import json

import wandb


def rename_wandb_name_path(path, remove_str):
    wandb_name = path
    wandb_name = wandb_name.replace(f"{remove_str}", "")
    wandb_name = wandb_name.replace("/", "_")
    wandb_name = wandb_name.replace("pixpro_base_r50_100ep", "")
    wandb_name = wandb_name.replace("convert_d2_models", "")
    wandb_name = wandb_name.rstrip("_")
    wandb_name = wandb_name.lstrip("_")
    wandb_name = wandb_name.replace("__", "_")
    return wandb_name


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path",
                        default="/home/tomo/ssl_pj/pixpro_pj/PixPro/output")
    parser.add_argument("--project", default="detectron2")
    parser.add_argument("--target_path", default="")
    parser.add_argument("--ids", nargs="+", default=None)
    parser.add_argument("--upload", action="store_true")
    args = parser.parse_args()

    root_path = os.path.abspath(args.root_path)
    root = os.path.abspath(args.target_path)

    wandb_name = rename_wandb_name_path(root, root_path)

    file_or_dirs = sorted(glob.glob(f"{root}/**", recursive=True))

    require_files = ["current.pth", ".o", ".txt", "config.json"]

    save_files = []
    tf_logs = []

    common_tf_log_dir = os.path.join(root, "tf_logs")
    for file_or_dir in file_or_dirs:
        if os.path.isfile(file_or_dir):
            is_require_files = any(w in file_or_dir for w in require_files)
            is_tf_logs = "events." in file_or_dir
            if is_tf_logs:
                is_new_tf_logs = common_tf_log_dir in file_or_dir
                if not is_new_tf_logs:
                    tf_logs.append(file_or_dir)
            elif is_require_files:
                save_files.append(file_or_dir)

    tf_logs = sorted(tf_logs)
    for i, tf_log in enumerate(tf_logs):
        wandb_id = None if args.ids is None else args.ids[i]
        local_tf_dirname = f"tf{i+1}"
        local_wandb_name = f"{wandb_name}_{local_tf_dirname}"
        run = wandb.init(entity="tomo",
                         project=args.project,
                         name=local_wandb_name,
                         id=wandb_id)
        wandb_id = run.id
        dirname = os.path.join(common_tf_log_dir, local_tf_dirname)
        if not os.path.isdir(dirname):
            os.makedirs(dirname, exist_ok=True)
        # wandb_id = local_wandb_name
        print("wandb_id:", wandb_id)
        print("wandb_dir:", dirname)
        print("tf_name:", tf_log)
        if args.upload:
            wandb.config.update(args)
            for save_file in save_files:
                if "config.json" in save_file:
                    with open(save_file, "r") as f:
                        config = json.load(f)
                    wandb.config.update(config)
                if i == 0:
                    print("save file:", save_file, file=sys.stderr)
                    wandb.save(save_file, base_path=root)
                    for tf_log in tf_logs:
                        print("save tf file:", tf_log, file=sys.stderr)
                        wandb.save(tf_log, base_path=root)
        run.finish()
