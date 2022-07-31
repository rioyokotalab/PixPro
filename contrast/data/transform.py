import numpy as np
from PIL import ImageFilter, ImageOps
from torchvision import transforms
from .rand_augment import rand_augment_transform
from . import transform_coord
from .augmentations import flow_augmentations
from .augmentations import color_augmentations


class GaussianBlur(object):
    """Gaussian Blur version 2"""

    def __call__(self, x):
        sigma = np.random.uniform(0.1, 2.0)
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


def get_transform(aug_type, crop, image_size=224, two_crop=False,
                  optical_flow_model=None, alpha_1=0.01, alpha_2=0.5):
    # is_use_flow = optical_flow_model is not None
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    if aug_type == "InstDisc":  # used in InstDisc and MoCo v1
        transform = transform_coord.Compose([
            transform_coord.RandomResizedCropCoord(image_size, scale=(crop, 1.)),
            transform_coord.RandomHorizontalFlipCoord(),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            normalize,
        ])
    elif aug_type == 'MoCov2':  # used in MoCov2
        transform = transform_coord.Compose([
            transform_coord.RandomResizedCropCoord(image_size, scale=(crop, 1.)),
            transform_coord.RandomHorizontalFlipCoord(),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur()], p=0.5),
            transforms.ToTensor(),
            normalize
        ])
    elif aug_type == 'SimCLR':  # used in SimCLR and PIC
        transform = transform_coord.Compose([
            transform_coord.RandomResizedCropCoord(image_size, scale=(crop, 1.)),
            transform_coord.RandomHorizontalFlipCoord(),
            transforms.RandomApply([transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur()], p=0.5),
            transforms.ToTensor(),
            normalize,
        ])
    elif aug_type == 'BYOL':
        transform_1 = [
            transform_coord.RandomResizedCropCoord(image_size, scale=(crop, 1.)),
            transform_coord.RandomHorizontalFlipCoord(),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur()], p=1.0),
            transforms.ToTensor(),
            normalize,
        ]
        transform_2 = [
            transform_coord.RandomResizedCropCoord(image_size, scale=(crop, 1.)),
            transform_coord.RandomHorizontalFlipCoord(),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur()], p=0.1),
            transforms.RandomApply([ImageOps.solarize], p=0.2),
            transforms.ToTensor(),
            normalize,
        ]
        transform_tuple = (transform_1, transform_2)
        transform = transform_coord.Compose(transform_tuple, two_crop=two_crop,
                                            optical_flow_model=optical_flow_model,
                                            alpha_1=alpha_1, alpha_2=alpha_2)
    elif aug_type == 'RandAug':  # used in InfoMin
        rgb_mean = (0.485, 0.456, 0.406)
        ra_params = dict(
            translate_const=int(224 * 0.45),
            img_mean=tuple([min(255, round(255 * x)) for x in rgb_mean]),
        )
        transform = transform_coord.Compose([
            transform_coord.RandomResizedCropCoord(image_size, scale=(crop, 1.)),
            transform_coord.RandomHorizontalFlipCoord(),
            transforms.RandomApply([
                transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)
            ], p=0.8),
            transforms.RandomApply([GaussianBlur()], p=0.5),
            rand_augment_transform('rand-n{}-m{}-mstd0.5'.format(2, 10), ra_params),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            normalize,
        ])
    elif aug_type == 'NULL':  # used in linear evaluation
        transform = transform_coord.Compose([
            transform_coord.RandomResizedCropCoord(image_size, scale=(crop, 1.)),
            transform_coord.RandomHorizontalFlipCoord(),
            transforms.ToTensor(),
            normalize,
        ])
    elif aug_type == 'val':  # used in validate
        transform = transforms.Compose([
            transforms.Resize(image_size + 32),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            normalize
        ])
    elif aug_type == 'PreprocessSimCLR':  # used in SimCLR and PIC
        preprocess = transform_coord.Compose([
            transforms.Resize((1080, 1920)),
            transform_coord.RandomRescaleCoord(min_scale=0.75, max_scale=1.25, step_size=0.0),
            transform_coord.RandomResizedCropCoord(image_size, scale=(1., 1.)),
        ], same_two=True)
        transform = transform_coord.Compose([
            preprocess,
            transform_coord.RandomResizedCropCoord(image_size, scale=(crop, 1.)),
            transform_coord.RandomHorizontalFlipCoord(),
            transforms.RandomApply([transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur()], p=0.5),
            transforms.ToTensor(),
            normalize,
        ])
    elif aug_type == 'PreprocessBYOL':
        preprocess = transform_coord.Compose([
            transforms.Resize((1080, 1920)),
            transform_coord.RandomRescaleCoord(min_scale=0.75, max_scale=1.25, step_size=0.0),
            transform_coord.RandomResizedCropCoord(image_size, scale=(1., 1.)),
        ], same_two=True)
        transform_1 = transform_coord.Compose([
            preprocess,
            transform_coord.RandomResizedCropCoord(image_size, scale=(crop, 1.)),
            transform_coord.RandomHorizontalFlipCoord(),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur()], p=1.0),
            transforms.ToTensor(),
            normalize,
        ])
        transform_2 = transform_coord.Compose([
            preprocess,
            transform_coord.RandomResizedCropCoord(image_size, scale=(crop, 1.)),
            transform_coord.RandomHorizontalFlipCoord(),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur()], p=0.1),
            transforms.RandomApply([ImageOps.solarize], p=0.2),
            transforms.ToTensor(),
            normalize,
        ])
        transform = (transform_1, transform_2)
    elif aug_type == 'mySimCLR':  # used in SimCLR and PIC
        transform = transform_coord.Compose([
            transforms.Resize((1080, 1920)),
            transform_coord.RandomRescaleCoord(min_scale=0.75, max_scale=1.25, step_size=0.0),
            transform_coord.RandomResizedCropCoord(image_size, scale=(crop, 1.)),
            # transform_coord.RandomHorizontalFlipCoord(),
            transforms.RandomApply([transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur()], p=0.5),
            transforms.ToTensor(),
            normalize,
        ], same_two=True)
    elif aug_type == 'myBYOL':
        transform_1 = transform_coord.Compose([
            transforms.Resize((1080, 1920)),
            transform_coord.RandomRescaleCoord(min_scale=0.75, max_scale=1.25, step_size=0.0),
            transform_coord.RandomResizedCropCoord(image_size, scale=(crop, 1.)),
            # transform_coord.RandomHorizontalFlipCoord(),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur()], p=1.0),
            transforms.ToTensor(),
            normalize,
        ], same_two=True)
        transform_2 = transform_coord.Compose([
            transforms.Resize((1080, 1920)),
            transform_coord.RandomRescaleCoord(min_scale=0.75, max_scale=1.25, step_size=0.0),
            transform_coord.RandomResizedCropCoord(image_size, scale=(crop, 1.)),
            # transform_coord.RandomHorizontalFlipCoord(),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur()], p=0.1),
            transforms.RandomApply([ImageOps.solarize], p=0.2),
            transforms.ToTensor(),
            normalize,
        ], same_two=True)
        transform = (transform_1, transform_2)
    elif aug_type == 'mySimCLRCoord':
        transform = flow_augmentations.Compose([
            flow_augmentations.FlowAugmentationWrapper(transforms.ToTensor()),
            flow_augmentations.RandomRescale(crop, 1.0, 0),
            flow_augmentations.RandomResizedCrop(image_size),
            flow_augmentations.RandomHorizontalFlip(),
            flow_augmentations.FlowAugmentationWrapper(transforms.RandomApply([transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8)),
            flow_augmentations.FlowAugmentationWrapper(transforms.RandomGrayscale(p=0.2)),
            flow_augmentations.FlowAugmentationWrapper(transforms.RandomApply([GaussianBlur()], p=0.5)),
            flow_augmentations.FlowAugmentationWrapper(normalize),
        ])
    elif aug_type == 'myBYOLCoord':
        transform_1 = flow_augmentations.Compose([
            flow_augmentations.FlowAugmentationWrapper(transforms.ToTensor()),
            flow_augmentations.RandomRescale(crop, 1.0, 0),
            flow_augmentations.RandomResizedCropCoord(image_size, scale=(crop, 1.)),
            flow_augmentations.RandomHorizontalFlipCoord(),
            flow_augmentations.FlowAugmentationWrapper(transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8)),
            flow_augmentations.FlowAugmentationWrapper(transforms.RandomGrayscale(p=0.2)),
            flow_augmentations.FlowAugmentationWrapper(transforms.RandomApply([color_augmentations.GaussianBlur(image_size[0], image_size[1], sigma=(0.1, 2))], p=1.0)),
            flow_augmentations.FlowAugmentationWrapper(normalize)
        ])
        transform_2 = transform_coord.Compose([
            flow_augmentations.FlowAugmentationWrapper(transforms.ToTensor()),
            flow_augmentations.RandomRescale(crop, 1.0, 0),
            flow_augmentations.RandomResizedCropCoord(image_size, scale=(crop, 1.)),
            flow_augmentations.RandomHorizontalFlipCoord(),
            flow_augmentations.FlowAugmentationWrapper(transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8)),
            flow_augmentations.FlowAugmentationWrapper(transforms.RandomGrayscale(p=0.2)),
            flow_augmentations.FlowAugmentationWrapper(transforms.RandomApply([color_augmentations.GaussianBlur(image_size[0], image_size[1], sigma=(0.1, 2))], p=0.1)),
            flow_augmentations.FlowAugmentationWrapper(transforms.RandomSolarize(125, 0.2)),
            flow_augmentations.FlowAugmentationWrapper(normalize)
        ])
        transform = (transform_1, transform_2)
    else:
        supported = '[InstDisc, MoCov2, SimCLR, BYOL, RandAug, NULL, val, mySimCLR, myBYOL, myBYOLCoord, mySimCLRCoord]'
        raise NotImplementedError(f'aug_type "{aug_type}" not supported. Should in {supported}')

    return transform
