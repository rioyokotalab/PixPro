# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import functools
import logging
import os
import sys
from termcolor import colored

import wandb


class _ColorfulFormatter(logging.Formatter):
    def __init__(self, *args, **kwargs):
        self._root_name = kwargs.pop("root_name") + "."
        self._abbrev_name = kwargs.pop("abbrev_name", "")
        if len(self._abbrev_name):
            self._abbrev_name = self._abbrev_name + "."
        super(_ColorfulFormatter, self).__init__(*args, **kwargs)

    def formatMessage(self, record):
        record.name = record.name.replace(self._root_name, self._abbrev_name)
        log = super(_ColorfulFormatter, self).formatMessage(record)
        if record.levelno == logging.WARNING:
            prefix = colored("WARNING", "red", attrs=["blink"])
        elif record.levelno == logging.ERROR or record.levelno == logging.CRITICAL:
            prefix = colored("ERROR", "red", attrs=["blink", "underline"])
        else:
            return log
        return prefix + " " + log


# so that calling setup_logger multiple times won't add many handlers
@functools.lru_cache()
def setup_logger(
    output=None, distributed_rank=0, *, color=True, name="contrast", abbrev_name=None
):
    """
    Initialize the detectron2 logger and set its verbosity level to "INFO".

    Args:
        output (str): a file name or a directory to save log. If None, will not save log file.
            If ends with ".txt" or ".log", assumed to be a file name.
            Otherwise, logs will be saved to `output/log.txt`.
        name (str): the root module name of this logger

    Returns:
        logging.Logger: a logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    if abbrev_name is None:
        abbrev_name = name

    plain_formatter = logging.Formatter(
        "[%(asctime)s] %(name)s %(levelname)s: %(message)s", datefmt="%m/%d %H:%M:%S"
    )
    # stdout logging: master only
    if distributed_rank == 0:
        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(logging.DEBUG)
        if color:
            formatter = _ColorfulFormatter(
                colored("[%(asctime)s %(name)s]: ", "green") + "%(message)s",
                datefmt="%m/%d %H:%M:%S",
                root_name=name,
                abbrev_name=str(abbrev_name),
            )
        else:
            formatter = plain_formatter
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    # file logging: all workers
    if output is not None:
        if output.endswith(".txt") or output.endswith(".log"):
            filename = output
        else:
            filename = os.path.join(output, "log.txt")
        if distributed_rank > 0:
            filename = filename + f".rank{distributed_rank}"
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        fh = logging.StreamHandler(_cached_log_stream(filename))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(plain_formatter)
        logger.addHandler(fh)

    return logger


# cache the opened file object, so that different calls to `setup_logger`
# with the same file name can safely write to the same file.
@functools.lru_cache(maxsize=None)
def _cached_log_stream(filename):
    return open(filename, "a")


# for wandb log
def get_git_files(root):
    file_or_dirs = sorted(glob.glob(f"{root}/**", recursive=True))

    git_files = []

    for file_or_dir in file_or_dirs:
        if os.path.isfile(file_or_dir):
            if "git" in file_or_dir:
                git_files.append(file_or_dir)

    return git_files


def get_save_files(root, require_files):
    file_or_dirs = sorted(glob.glob(f"{root}/**", recursive=True))

    save_files = []

    for file_or_dir in file_or_dirs:
        if os.path.isfile(file_or_dir):
            is_require_files = any(w in file_or_dir for w in require_files)
            is_model_f = ".pth" in file_or_dir
            is_require_files = is_require_files and (not is_model_f)
            if is_require_files:
                save_files.append(file_or_dir)

    save_files = sorted(save_files)
    return save_files


def get_wandb_name(args):
    wandb_name = "pretrain_"
    wandb_name += f"crop-{args.crop}_"
    wandb_name += f"aug-{args.aug}_"
    wandb_name += f"{args.dataset}_"
    wandb_name += f"image-size-{args.image_size}_"
    wandb_name += f"l-bn-{args.batch_size}_"
    wandb_name += f"epoch-{args.epochs}_"
    if args.flowe_loss:
        wandb_name += "flowe-loss_"
    if args.pixpro_no_headsim:
        wandb_name += "no-headsim_"
    wandb_name = wandb_name.rstrip("_")
    return wandb_name


def init_wandb(args):
    wandb_name = get_wandb_name(args)
    wandb.init(project="PixPro", entity="tomo", name=wandb_name)
    wandb.config.update(args)
    git_files = get_git_files(args.output_dir)
    for f in git_files:
        wandb.save(f, base_path=args.output_dir)

