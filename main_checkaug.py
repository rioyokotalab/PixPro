import json
import os
import time
from shutil import copyfile

import torch
import torch.distributed as dist
from torch.backends import cudnn
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
import torchvision

import wandb
from contrast.logger import init_wandb, get_save_files

from contrast.data import get_loader
from contrast.logger import setup_logger
from contrast.lr_scheduler import get_scheduler
from contrast.option import parse_option
from contrast.util import AverageMeter

from main_pretrain import build_model
from main_pretrain import load_pretrained, load_checkpoint

try:
    # noinspection PyUnresolvedReferences
    from apex import amp
except ImportError:
    amp = None


def dist_setup():
    master_addr = os.getenv("MASTER_ADDR", default="localhost")
    master_port = os.getenv("MASTER_PORT", default="8888")
    method = "tcp://{}:{}".format(master_addr, master_port)
    rank = int(os.getenv("OMPI_COMM_WORLD_RANK", "0"))
    world_size = int(os.getenv("OMPI_COMM_WORLD_SIZE", "1"))
    local_rank = int(os.getenv("OMPI_COMM_WORLD_LOCAL_RANK", "-1"))
    local_size = int(os.getenv("OMPI_COMM_WORLD_LOCAL_SIZE", "-2"))
    node_rank = int(os.getenv("OMPI_COMM_WORLD_NODE_RANK", "-3"))
    host_port_str = f"host: {master_addr}, port: {master_port}"
    print(
        "rank: {}, world_size: {}, local_rank: {}, local_size: {}, node_rank: {}, {}"
        .format(rank, world_size, local_rank, local_size, node_rank, host_port_str))
    dist.init_process_group("nccl", init_method=method, rank=rank,
                            world_size=world_size)
    print("Rank: {}, Size: {}, Host: {} Port: {}".format(dist.get_rank(),
                                                         dist.get_world_size(),
                                                         master_addr, master_port))
    return local_rank


def main(args):
    train_prefix = 'train'
    train_loader = get_loader(
        args.aug, args,
        two_crop=args.model in ['PixPro'],
        prefix=train_prefix,
        return_coord=True,)
    # train_loader.dataset.cuda()
    # train_loader.data.cuda()

    args.num_instances = len(train_loader.dataset)
    logger.info(f"length of training dataset: {args.num_instances}")

    model, optimizer = build_model(args)
    scheduler = get_scheduler(optimizer, len(train_loader), args)

    # optionally resume from a checkpoint
    if args.pretrained_model:
        assert os.path.isfile(args.pretrained_model)
        load_pretrained(model, args.pretrained_model)
    if args.auto_resume:
        resume_file = os.path.join(args.output_dir, "current.pth")
        if os.path.exists(resume_file):
            logger.info(f'auto resume from {resume_file}')
            args.resume = resume_file
        else:
            logger.info(f'no checkpoint found in {args.output_dir}, ignoring auto resume')
    if args.resume:
        assert os.path.isfile(args.resume)
        load_checkpoint(args, model, optimizer, scheduler, sampler=train_loader.sampler)

    # tensorboard
    if dist.get_rank() == 0:
        summary_writer = SummaryWriter(log_dir=args.output_dir)
    else:
        summary_writer = None

    for epoch in range(args.start_epoch, args.epochs + 1):
        if isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(epoch)

        save_img(epoch, train_loader, model, optimizer, scheduler, args, summary_writer)

        if epoch >= args.debug_epochs:
            break


def save_img(epoch, train_loader, model, optimizer, scheduler, args, summary_writer):
    """
    one epoch training
    """
    model.train()

    batch_time = AverageMeter()
    end = time.time()
    train_len = len(train_loader)
    cur_epoch = f"epoch{epoch}"
    out_root = os.path.join(args.output_dir, "check_aug_pair", cur_epoch)
    os.makedirs(out_root, exist_ok=True)
    for idx, data in enumerate(train_loader):
        filename = f"debug_img_{idx}.png"
        filename = os.path.join(out_root, filename)
        data = [item.cuda(non_blocking=True) for item in data]
        im1, im2 = data[0], data[1]
        index = data[4]
        if dist.get_rank() == 0:
            print("coord1", data[2], "coord2", data[3])
            print("im1", im1.shape, "im2", im2.shape)
            print("idx", index)
        bn = im1.shape[0] if im1.ndim > 3 else 1
        c, h, w = im1.shape[-3:]
        save_imgs = []
        if im1.ndim > 3:
            print("in cod")
            im1 = im1.reshape(bn, -1, c, h, w)
            im2 = im2.reshape(bn, -1, c, h, w)
            num_img = im1.shape[1]
            for i in range(num_img):
                for j in range(bn):
                    save_imgs.append(torch.stack([im1[j][i], im2[j][i]]))
        else:
            save_imgs.append(torch.stack([im1, im2]))
        save_imgs = torch.stack(save_imgs)
        print(save_imgs.shape)
        save_imgs = save_imgs.reshape(-1, c, h, w)
        torchvision.utils.save_image(save_imgs,
                                     filename,
                                     nrow=bn * 2,
                                     padding=10)

        batch_time.update(time.time() - end)
        end = time.time()
        logger.info(
            f'Train: [{epoch}/{args.epochs}][{idx}/{train_len}]  '
            f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})  ')


def main_prog(opt):
    # setup logger
    os.makedirs(opt.output_dir, exist_ok=True)
    global logger
    logger = setup_logger(output=opt.output_dir,
                          distributed_rank=dist.get_rank(), name="contrast")
    if dist.get_rank() == 0:
        path = os.path.join(opt.output_dir, "config.json")
        with open(path, 'w') as f:
            json.dump(vars(opt), f, indent=2)
        logger.info("Full config saved to {}".format(path))
        init_wandb(opt, project="ssl_test")
        wandb.save(path, base_path=opt.output_dir)

    # print args
    logger.info(
        "\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(opt)).items()))
    )

    main(opt)

    if dist.get_rank() == 0:
        out_root = opt.output_dir
        log_name = opt.log_name
        if log_name != "" and log_name is not None:
            if os.path.isfile(log_name):
                abs_log_path = os.path.abspath(log_name)
                log_base_name = os.path.basename(abs_log_path)
                ext_log = os.path.splitext(log_base_name)[1]
                if ext_log != ".txt":
                    log_base_name += ".txt"
                new_logname = os.path.join(out_root, log_base_name)
                copyfile(abs_log_path, new_logname)
        require_files = [".o", ".txt", ".sh", "config.json"]
        save_files = get_save_files(out_root, require_files)
        for f in save_files:
            new_f = f
            is_tf_log = "events." in f
            is_log = ".out" in f or ".txt" in f
            if not is_tf_log and is_log:
                base_name = os.path.basename(f)
                ext_log = os.path.splitext(base_name)[1]
                if ext_log != ".txt":
                    new_f = f + ".txt"
                    copyfile(f, new_f)
            wandb.save(new_f, base_path=out_root)


if __name__ == '__main__':
    opt = parse_option(stage='pre-train')

    if opt.amp_opt_level != "O0":
        assert amp is not None, "amp not installed!"

    # mpirun
    local_rank = dist_setup()
    opt.local_rank = local_rank

    torch.cuda.set_device(opt.local_rank)

    # # pytorch distrubuted run
    # torch.distributed.init_process_group(backend='nccl', init_method='env://')

    cudnn.benchmark = True

    main_prog(opt)
