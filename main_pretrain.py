import json
import os
import time
from shutil import copyfile

import torch
import torch.distributed as dist
from torch.backends import cudnn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

import wandb

from contrast.logger import init_wandb

from contrast import models
from contrast import resnet
from contrast.data import get_loader
from contrast.logger import setup_logger
from contrast.lr_scheduler import get_scheduler
from contrast.option import parse_option
from contrast.util import AverageMeter
from contrast.lars import add_weight_decay, LARS

from contrast.flow import RAFT
# from contrast.flow import InputPadder
from contrast.util import apply_optical_flow
from contrast import util

try:
    # noinspection PyUnresolvedReferences
    from apex import amp
except ImportError:
    amp = None


def build_model(args):
    encoder = resnet.__dict__[args.arch]
    model = models.__dict__[args.model](encoder, args).cuda()

    if args.use_flow:
        flow_model = torch.nn.DataParallel(RAFT(args))
        weights = torch.load(args.flow_model, map_location="cpu")
        flow_model.load_state_dict(weights)
        flow_model = flow_model.module.cuda()
        flow_model = DistributedDataParallel(flow_model, device_ids=[args.local_rank],
                                             broadcast_buffers=False)
        flow_model.eval()
        for param in flow_model.parameters():
            param.requires_grad = False

    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.batch_size * dist.get_world_size() / 256 * args.base_learning_rate,
            momentum=args.momentum,
            weight_decay=args.weight_decay,)
    elif args.optimizer == 'lars':
        params = add_weight_decay(model, args.weight_decay)
        optimizer = torch.optim.SGD(
            params,
            lr=args.batch_size * dist.get_world_size() / 256 * args.base_learning_rate,
            momentum=args.momentum,)
        optimizer = LARS(optimizer)
    else:
        raise NotImplementedError

    if args.amp_opt_level != "O0":
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.amp_opt_level)

    model = DistributedDataParallel(model, device_ids=[args.local_rank], broadcast_buffers=False)

    if args.use_flow:
        model = [model, flow_model]

    return model, optimizer


def load_pretrained(model, pretrained_model):
    ckpt = torch.load(pretrained_model, map_location='cpu')
    state_dict = ckpt['model']
    model_dict = model.state_dict()

    model_dict.update(state_dict)
    model.load_state_dict(model_dict)
    logger.info(f"==> loaded checkpoint '{pretrained_model}' (epoch {ckpt['epoch']})")


def load_checkpoint(args, model, optimizer, scheduler, sampler=None):
    logger.info(f"=> loading checkpoint '{args.resume}'")

    checkpoint = torch.load(args.resume, map_location='cpu')
    args.start_epoch = checkpoint['epoch'] + 1
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])

    if args.amp_opt_level != "O0" and checkpoint['opt'].amp_opt_level != "O0":
        amp.load_state_dict(checkpoint['amp'])

    logger.info(f"=> loaded successfully '{args.resume}' (epoch {checkpoint['epoch']})")

    del checkpoint
    torch.cuda.empty_cache()


def save_checkpoint(args, epoch, model, optimizer, scheduler, sampler=None):
    logger.info('==> Saving...')
    state = {
        'opt': args,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'epoch': epoch,
    }
    if args.amp_opt_level != "O0":
        state['amp'] = amp.state_dict()
    file_name = os.path.join(args.output_dir, f'ckpt_epoch_{epoch}.pth')
    torch.save(state, file_name)
    copyfile(file_name, os.path.join(args.output_dir, 'current.pth'))


def main(args):
    train_prefix = 'train'
    train_loader = get_loader(
        args.aug, args,
        two_crop=args.model in ['PixPro'],
        prefix=train_prefix,
        return_coord=True,)

    args.num_instances = len(train_loader.dataset)
    logger.info(f"length of training dataset: {args.num_instances}")

    model, optimizer = build_model(args)
    scheduler = get_scheduler(optimizer, len(train_loader), args)

    if args.use_flow:
        model, flow_model = model

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

    if args.use_flow:
        model = [model, flow_model]
    else:
        model = [model]

    for epoch in range(args.start_epoch, args.epochs + 1):
        if isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(epoch)

        train(epoch, train_loader, model, optimizer, scheduler, args, summary_writer)

        if dist.get_rank() == 0 and (epoch % args.save_freq == 0 or epoch == args.epochs):
            save_checkpoint(args, epoch, model[0], optimizer, scheduler, sampler=train_loader.sampler)

        if epoch >= args.debug_epochs:
            break


def train(epoch, train_loader, model, optimizer, scheduler, args, summary_writer):
    """
    one epoch training
    """
    is_mask_flow = args.alpha1 is not None and args.alpha2 is not None
    is_mask_flow = is_mask_flow and args.use_flow

    if args.use_flow:
        model, flow_model = model
        flow_model.eval()
    else:
        model = model[0]
    model.train()

    batch_time = AverageMeter()
    loss_meter = AverageMeter()

    end = time.time()
    for idx, data in enumerate(train_loader):
        data = [item.cuda(non_blocking=True) for item in data]

        if args.use_flow:
            flow_fwd, flow_bwd = apply_optical_flow(data, flow_model, args)
            data[2] = [data[2], flow_fwd]
            data[3] = [data[3], flow_bwd]

        if is_mask_flow:
            mask_fwd, mask_bwd = flow_fwd[2][0], flow_bwd[2][0]
            r_fwds = util.calc_mask_ratio(mask_fwd)
            r_bwds = util.calc_mask_ratio(mask_bwd)
            with torch.no_grad():
                r_fwd, r_bwd = r_fwds.mean().item(), r_bwds.mean().item()

        if args.debug:
            data[2] = (data[2], [data[6], idx, epoch])
            data[3] = (data[3], [data[7], idx, epoch])

        # In PixPro, data[0] -> im1, data[1] -> im2, data[2] -> coord1, data[3] -> coord2
        loss, pos_num_list = model(data[0], data[1], data[2], data[3])

        # backward
        optimizer.zero_grad()
        if args.amp_opt_level != "O0":
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        optimizer.step()
        scheduler.step()

        # update meters and print info
        loss_meter.update(loss.item(), data[0].size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        train_len = len(train_loader)
        pos_num_list_1, pos_num_list_2 = pos_num_list
        pos_nums_1, pos_means_1 = pos_num_list_1
        pos_nums_2, pos_means_2 = pos_num_list_2
        with torch.no_grad():
            pos_num_1, pos_mean_1 = pos_nums_1.sum().item(), pos_means_1.mean().item()
            pos_num_2, pos_mean_2 = pos_nums_2.sum().item(), pos_means_2.mean().item()
            pos_num = pos_num_1 + pos_num_2
            pos_mean = (pos_mean_1 + pos_mean_2) / 2.0

        if idx % args.print_freq == 0:
            lr = optimizer.param_groups[0]['lr']
            loss_plus = loss_meter.val + 4.0
            mask_ratio_str = ''
            if is_mask_flow:
                mask_ratio_str = f'mask ratio {r_fwd:.3f} {r_bwd:.3f}'

            logger.info(
                f'Train: [{epoch}/{args.epochs}][{idx}/{train_len}]  '
                f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})  '
                f'lr {lr:.3f}  '
                f'loss {loss_meter.val:.3f} ({loss_meter.avg:.3f}) [{loss_plus:.3f}] '
                f'pos_num {pos_num} ({pos_mean}) '
                f'{mask_ratio_str}')

        if args.debug:
            continue

        name_pos_num, name_pos_mean = "positive_pair/num", "positive_pair/avg"
        name_pos_num_1, name_pos_mean_1 = f"{name_pos_num}/1", f"{name_pos_mean}/1"
        name_pos_num_2, name_pos_mean_2 = f"{name_pos_num}/2", f"{name_pos_mean}/2"
        name_fwd, name_bwd = 'mask_ratio/fwd', 'mask_ratio/bwd'

        # tensorboard logger
        is_tensorboard_log = summary_writer is not None
        if is_tensorboard_log:
            step = (epoch - 1) * len(train_loader) + idx
            loss_plus = loss_meter.val + 4.0
            lr = optimizer.param_groups[0]['lr']
            summary_writer.add_scalar('lr', lr, step)
            summary_writer.add_scalar('loss', loss_meter.val, step)
            summary_writer.add_scalar('loss/plus', loss_plus, step)
            summary_writer.add_scalar(name_pos_num, pos_num, step)
            summary_writer.add_scalar(name_pos_mean, pos_mean, step)
            summary_writer.add_scalar(name_pos_num_1, pos_num_1, step)
            summary_writer.add_scalar(name_pos_mean_1, pos_mean_1, step)
            summary_writer.add_scalar(name_pos_num_2, pos_num_2, step)
            summary_writer.add_scalar(name_pos_mean_2, pos_mean_2, step)
            if args.use_flow:
                summary_writer.add_scalar(name_fwd, r_fwd, step)
                summary_writer.add_scalar(name_bwd, r_bwd, step)

        # wandb logger
        is_wandb_log = dist.get_rank() == 0
        if is_wandb_log:
            global_step = (epoch - 1) * train_len + idx
            loss_plus = loss_meter.val + 4.0
            lr = optimizer.param_groups[0]['lr']
            wandb_dict = {"lr": lr, "loss": loss_meter.val, "loss/avg": loss_meter.avg,
                          "loss/plus": loss_plus, "epoch": epoch - 1,
                          "global_step": global_step, "time": batch_time.val,
                          "time/avg": batch_time.avg,
                          name_pos_num: pos_num, name_pos_mean: pos_mean,
                          name_pos_num_1: pos_num_1, name_pos_mean_1: pos_mean_1,
                          name_pos_num_2: pos_num_2, name_pos_mean_2: pos_mean_2}
            if is_mask_flow:
                wandb_dict[name_fwd] = r_fwd
                wandb_dict[name_bwd] = r_bwd

        nb = pos_nums_1.shape[0]
        for pos_i in range(nb):
            tail_batch_str = f"batch_{pos_i}"
            l_name_pos_num_1 = f"{name_pos_num_1}/{tail_batch_str}"
            l_name_pos_mean_1 = f"{name_pos_mean_1}/{tail_batch_str}"
            l_name_pos_num_2 = f"{name_pos_num_2}/{tail_batch_str}"
            l_name_pos_mean_2 = f"{name_pos_mean_2}/{tail_batch_str}"
            l_pos_num_1 = pos_nums_1[pos_i].item()
            l_pos_mean_1 = pos_means_1[pos_i].item()
            l_pos_num_2 = pos_nums_2[pos_i].item()
            l_pos_mean_2 = pos_means_2[pos_i].item()

            if is_mask_flow:
                l_r_fwd, l_r_bwd = r_fwds[pos_i].item(), r_bwds[pos_i].item()
                l_name_fwd = f"{name_fwd}/{tail_batch_str}"
                l_name_bwd = f"{name_bwd}/{tail_batch_str}"

            if is_tensorboard_log:
                summary_writer.add_scalar(l_name_pos_num_1, l_pos_num_1, step)
                summary_writer.add_scalar(l_name_pos_mean_1, l_pos_mean_1, step)
                summary_writer.add_scalar(l_name_pos_num_2, l_pos_num_2, step)
                summary_writer.add_scalar(l_name_pos_mean_2, l_pos_mean_2, step)
                if args.use_flow:
                    summary_writer.add_scalar(f'{l_name_fwd}', l_r_fwd, step)
                    summary_writer.add_scalar(f'{l_name_bwd}', l_r_bwd, step)

            if is_wandb_log:
                wandb_dict[l_name_pos_num_1] = l_pos_num_1
                wandb_dict[l_name_pos_mean_1] = l_pos_mean_1
                wandb_dict[l_name_pos_num_2] = l_pos_num_2
                wandb_dict[l_name_pos_mean_2] = l_pos_mean_2
                if args.use_flow:
                    wandb_dict[l_name_fwd] = l_r_fwd
                    wandb_dict[l_name_bwd] = l_r_bwd

        if is_wandb_log:
            wandb.log(wandb_dict)


def main_prog(opt):
    cudnn.benchmark = not opt.no_benchmark
    # setup logger
    os.makedirs(opt.output_dir, exist_ok=True)
    global logger
    logger = setup_logger(output=opt.output_dir, distributed_rank=dist.get_rank(), name="contrast")
    if dist.get_rank() == 0:
        path = os.path.join(opt.output_dir, "config.json")
        with open(path, 'w') as f:
            json.dump(vars(opt), f, indent=2)
        logger.info("Full config saved to {}".format(path))
        if not opt.debug:
            init_wandb(opt)
            wandb.save(path, base_path=opt.output_dir)

    # print args
    logger.info(
        "\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(opt)).items()))
    )

    main(opt)


if __name__ == '__main__':
    opt = parse_option(stage='pre-train')

    if opt.amp_opt_level != "O0":
        assert amp is not None, "amp not installed!"

    torch.cuda.set_device(opt.local_rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://')

    main_prog(opt)
