import os

import numpy as np
import torch.distributed as dist
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.utils.data.distributed import DistributedSampler

from .transform import get_transform
from .dataset import ImageFolder


def get_loader(aug_type, args, two_crop=False, prefix='train', return_coord=False):
    image_size = args.image_size
    if image_size == 1024:
        image_size = (512, 1024)
    is_corner = not args.is_center
    transform = get_transform(aug_type, args.crop, image_size, two_crop, is_corner,
                              args.same_grid)

    # dataset
    if args.zip:
        if args.dataset == 'ImageNet' or args.dataset == "bdd100k":
            train_ann_file = prefix + "_map.txt"
            train_prefix = prefix + ".zip@/"
        else:
            raise NotImplementedError('Dataset {} is not supported. We only support ImageNet now'.format(args.dataset))

        train_dataset = ImageFolder(
            args.data_dir,
            train_ann_file,
            train_prefix,
            transform,
            two_crop=two_crop,
            cache_mode=args.cache_mode,
            dataset=args.dataset,
            return_coord=return_coord,
            n_frames=args.n_frames)
    else:
        train_folder = os.path.join(args.data_dir, prefix)
        train_dataset = ImageFolder(
            train_folder,
            transform=transform,
            two_crop=two_crop,
            dataset=args.dataset,
            return_coord=return_coord,
            n_frames=args.n_frames)

    # sampler
    indices = np.arange(dist.get_rank(), len(train_dataset), dist.get_world_size())
    if args.zip and args.cache_mode == 'part':
        sampler = SubsetRandomSampler(indices)
    else:
        sampler = DistributedSampler(train_dataset)

    # dataloader
    return DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=True)
