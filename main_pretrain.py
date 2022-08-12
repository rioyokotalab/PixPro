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

from contrast.flow import RAFT, InputPadder

try:
    # noinspection PyUnresolvedReferences
    from apex import amp
except ImportError:
    amp = None


@torch.no_grad()
def calc_optical_flow(orig_im1, orig_im2, flow_model):
    flow_model.eval()
    padder = InputPadder(orig_im1.shape)
    padder.pad(orig_im1, orig_im2)
    flow_fwd, _ = flow_model(orig_im1, orig_im2, test_mode=True)
    flow_bwd, _ = flow_model(orig_im2, orig_im1, test_mode=True)
    flow_fwd = flow_fwd.cuda()
    flow_bwd = flow_bwd.cuda()
    return flow_fwd, flow_bwd


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

        with torch.no_grad():
            orig_im1, orig_im2 = data[6], data[7]
            bs = orig_im1.shape[0]
            # to reduce memory usage
            flow_fwds, flow_bwds = [], []
            for i in range(0, bs, 2):
                if i + 2 > bs:
                    break
                l_orig_im1 = orig_im1[i:i+2]
                l_orig_im2 = orig_im2[i:i+2]
                flow_fwd, flow_bwd = calc_optical_flow(l_orig_im1, l_orig_im2,
                                                       flow_model)
                flow_fwds.append(flow_fwd)
                flow_bwds.append(flow_bwd)
            flow_fwd = torch.cat(flow_fwds, dim=0)
            flow_bwd = torch.cat(flow_bwds, dim=0)
            if bs % 2 != 0:
                l_flow_fwd, l_flow_bwd = calc_optical_flow(orig_im1[-1], orig_im2[-1],
                                                           flow_model)
                flow_fwd = torch.cat([flow_fwd, l_flow_bwd], dim=0)
                flow_bwd = torch.cat([flow_bwd, l_flow_bwd], dim=0)
            # flow_fwd, flow_bwd = calc_optical_flow(orig_im1, orig_im2, flow_model)
            flow_fwd = flow_fwd.cuda()
            flow_bwd = flow_bwd.cuda()

            data[2] = [data[2], flow_fwd]
            data[3] = [data[3], flow_bwd]

        # In PixPro, data[0] -> im1, data[1] -> im2, data[2] -> coord1, data[3] -> coord2
        loss = model(data[0], data[1], data[2], data[3])

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
        if idx % args.print_freq == 0:
            lr = optimizer.param_groups[0]['lr']
            logger.info(
                f'Train: [{epoch}/{args.epochs}][{idx}/{train_len}]  '
                f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})  '
                f'lr {lr:.3f}  '
                f'loss {loss_meter.val:.3f} ({loss_meter.avg:.3f})')

            # tensorboard logger
            if summary_writer is not None:
                step = (epoch - 1) * len(train_loader) + idx
                summary_writer.add_scalar('lr', lr, step)
                summary_writer.add_scalar('loss', loss_meter.val, step)

        if dist.get_rank() == 0:
            global_step = (epoch - 1) * train_len + idx
            loss_plus = loss_meter.val + 4.0
            wandb.log({"lr": lr, "loss": loss_meter.val, "loss/avg": loss_meter.avg,
                       "loss/plus": loss_plus, "epoch": epoch - 1,
                       "global_step": global_step, "time": batch_time.val,
                       "time/avg": batch_time.avg})


def main_prog(opt):
    # setup logger
    os.makedirs(opt.output_dir, exist_ok=True)
    global logger
    logger = setup_logger(output=opt.output_dir, distributed_rank=dist.get_rank(), name="contrast")
    if dist.get_rank() == 0:
        path = os.path.join(opt.output_dir, "config.json")
        with open(path, 'w') as f:
            json.dump(vars(opt), f, indent=2)
        logger.info("Full config saved to {}".format(path))
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
    cudnn.benchmark = True

    main_prog(opt)
