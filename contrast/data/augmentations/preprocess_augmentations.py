import torch.nn as nn

import random
import numpy as np
from PIL import Image, ImageOps, ImageFilter


class SSAugmentationWrapper(nn.Module):

    def __init__(self, transform):
        super().__init__()
        self.transform = transform

    def forward(self, images):
        outputs = [self.transform(image) for image in images]
        return outputs


class RandomRescale:
    """rescale an image and label with in target scale
    PIL image version"""

    def __init__(self, min_scale=0.5, max_scale=2.0, step_size=0.25):
        """initialize
        Args:
            min_scale: Min target scale.
            max_scale: Max target scale.
        """
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.step_size = step_size
        # discrete scales
        if (max_scale - min_scale) > step_size and step_size > 0.05:
            self.num_steps = int((max_scale - min_scale) / step_size + 1)
            self.scale_steps = np.linspace(self.min_scale, self.max_scale,
                                           self.num_steps)
        elif (max_scale - min_scale) > step_size and step_size < 0.05:
            self.num_steps = 0
            self.scale_steps = np.array([min_scale])
        else:
            self.num_steps = 1
            self.scale_steps = np.array([min_scale])

    def __call__(self, images):
        """call method"""
        width, height = images[0].size
        # random scale
        if self.num_steps > 0:
            index = random.randint(0, self.num_steps - 1)
            scale_now = self.scale_steps[index]
        else:
            scale_now = random.uniform(self.min_scale, self.max_scale)
        new_width, new_height = int(scale_now * width), int(scale_now * height)
        # resize
        # image = image.resize(self.size, Image.BILINEAR)
        outputs = []
        for image in images:
            outputs.append(image.resize((new_width, new_height), Image.BICUBIC))

        return outputs


class RandomPadOrCrop:
    """Crops and/or pads an image to a target width and height
    PIL image version
    """

    def __init__(self, crop_height, crop_width, mean=(125, 125, 125)):
        """
        Args:
            crop_height: The new height.
            crop_width: The new width.
            ignore_label: Label class to be ignored.
        """
        self.crop_height = crop_height
        self.crop_width = crop_width
        self.mean = mean

    def __call__(self, images):
        """call method"""
        width, height = images[0].size
        pad_width, pad_height = max(width,
                                    self.crop_width), max(height, self.crop_height)
        pad_width = self.crop_width - width if width < self.crop_width else 0
        pad_height = self.crop_height - height if height < self.crop_height else 0
        # pad the image with constant
        outputs = [
            ImageOps.expand(image, border=(0, 0, pad_width, pad_height), fill=self.mean)
            for image in images
        ]
        # random crop image to crop_size
        new_w, new_h = outputs[0].size
        x1 = random.randint(0, new_w - self.crop_width)
        y1 = random.randint(0, new_h - self.crop_height)

        outputs = [
            image.crop((x1, y1, x1 + self.crop_width, y1 + self.crop_height))
            for image in outputs
        ]

        return outputs


class RandomHorizontalFlip:
    """Randomly flip an image and label horizontally (left to right).
    PIL image version"""

    def __call__(self, images):
        """call method"""
        outputs = images
        if random.random() < 0.5:
            outputs = [image.transpose(Image.FLIP_LEFT_RIGHT) for image in images]

        return outputs


class RandomRotate:

    def __init__(self, degree):
        self.degree = degree

    def __call__(self, images):
        rotate_degree = random.uniform(-1 * self.degree, self.degree)
        outputs = []
        for image in images:
            outputs.append(image.rotate(rotate_degree, Image.BILINEAR))

        return outputs


class RandomGaussianBlur:

    def __call__(self, images):
        outputs = images
        if random.random() < 0.5:
            outputs = [
                image.filter(ImageFilter.GaussianBlur(radius=random.random()))
                for image in images
            ]

        return outputs


class FixScaleCrop:

    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, images):
        w, h = images[0].size
        if w > h:
            oh = self.crop_size
            ow = int(1.0 * w * oh / h)
        else:
            ow = self.crop_size
            oh = int(1.0 * h * ow / w)
        outputs = [image.resize((ow, oh), Image.BILINEAR) for image in images]
        # center crop
        w, h = outputs[0].size
        x1 = int(round((w - self.crop_size) / 2.))
        y1 = int(round((h - self.crop_size) / 2.))
        outputs = [
            image.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
            for image in outputs
        ]

        return outputs
