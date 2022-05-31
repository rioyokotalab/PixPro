from __future__ import division
import torch
import math
import random
import numpy as np
from PIL import Image
try:
    import accimage
except ImportError:
    accimage = None
import warnings

from torchvision.transforms import functional as F


_pil_interpolation_to_str = {
    Image.NEAREST: 'PIL.Image.NEAREST',
    Image.BILINEAR: 'PIL.Image.BILINEAR',
    Image.BICUBIC: 'PIL.Image.BICUBIC',
    Image.LANCZOS: 'PIL.Image.LANCZOS',
    Image.HAMMING: 'PIL.Image.HAMMING',
    Image.BOX: 'PIL.Image.BOX',
}


def _get_image_size(img):
    if F._is_pil_image(img):
        return img.size
    elif isinstance(img, torch.Tensor) and img.dim() > 2:
        return img.shape[-2:][::-1]
    else:
        raise TypeError("Unexpected type {}".format(type(img)))


class Compose(object):
    """Composes several transforms together.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.

    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms, same_two=False):
        self.transforms = transforms
        self.same_two = same_two

    def __call__(self, img, coord=None):
        params, in_params = None, None
        tmp_len_key = {}
        if self.same_two and coord is not None:
            coord, params = coord

        for t in self.transforms:
            if t.__class__.__name__ in tmp_len_key.keys():
                tmp_len_key[t.__class__.__name__] += 1
            else:
                tmp_len_key[t.__class__.__name__] = 1
            if self.same_two and params is not None:
                key = t.__class__.__name__
                len_key = tmp_len_key[key]
                if len_key > 1:
                    key += f"{len_key}"
                in_params = params.get(key, None)

            if 'RandomResizedCropCoord' in t.__class__.__name__:
                if self.same_two:
                    coord = [coord, in_params]
                in_img = [img, coord] if self.same_two else img
                img, coord = t(in_img, same_two=self.same_two)
            elif 'RandomRescaleCoord' in t.__class__.__name__:
                if self.same_two:
                    coord = [coord, in_params]
                in_img = [img, coord] if self.same_two else img
                img, coord = t(in_img, same_two=self.same_two)
            elif 'FlipCoord' in t.__class__.__name__:
                img, coord = t(img, coord)
                if self.same_two and coord is not None:
                    coord = [coord, None]
            else:
                img = t(img)
                if self.same_two and coord is not None:
                    coord = [coord, None]

            if self.same_two and coord is not None:
                coord, in_params = coord
                if in_params is not None:
                    key = t.__class__.__name__
                    len_key = tmp_len_key[key]
                    if params is None:
                        params = {}
                    if len_key > 1:
                        c_key = key + f"{len_key}"
                        params[c_key] = in_params[key]
                    else:
                        params.update(in_params)

        if self.same_two:
            coord = [coord, params]

        return img, coord

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class RandomRescaleCoord(object):
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

    def __call__(self, img, same_two=False):
        """call method"""
        if same_two:
            img, coord = img
            coord, params = coord
        width, height = img.size

        if not same_two or params is None:
            # random scale
            if self.num_steps > 0:
                index = random.randint(0, self.num_steps - 1)
                scale_now = self.scale_steps[index]
            else:
                scale_now = random.uniform(self.min_scale, self.max_scale)
        else:
            scale_now = params
        new_width, new_height = int(scale_now * width), int(scale_now * height)
        # resize
        img = img.resize((new_width, new_height), Image.BICUBIC)
        if same_two:
            params = {self.__class__.__name__: scale_now}
            coord = [coord, params]

        return img, coord


class RandomHorizontalFlipCoord(object):
    """Horizontally flip the given PIL Image randomly with a given probability.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, coord):
        """
        Args:
            img (PIL Image): Image to be flipped.

        Returns:
            PIL Image: Randomly flipped image.
        """
        if random.random() < self.p:
            coord_new = coord.clone()
            coord_new[0] = coord[2]
            coord_new[2] = coord[0]
            return F.hflip(img), coord_new
        return img, coord

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class RandomVerticalFlipCoord(object):
    """Vertically flip the given PIL Image randomly with a given probability.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, coord):
        """
        Args:
            img (PIL Image): Image to be flipped.

        Returns:
            PIL Image: Randomly flipped image.
        """
        if random.random() < self.p:
            coord_new = coord.clone()
            coord_new[1] = coord[3]
            coord_new[3] = coord[1]
            return F.vflip(img), coord_new
        return img, coord

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class RandomResizedCropCoord(object):
    """Crop the given PIL Image to random size and aspect ratio.

    A crop of random size (default: of 0.08 to 1.0) of the original size and a random
    aspect ratio (default: of 3/4 to 4/3) of the original aspect ratio is made. This crop
    is finally resized to given size.
    This is popularly used to train the Inception networks.

    Args:
        size: expected output size of each edge
        scale: range of size of the origin size cropped
        ratio: range of aspect ratio of the origin aspect ratio cropped
        interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.), interpolation=Image.BILINEAR):
        if isinstance(size, (tuple, list)):
            self.size = size
        else:
            self.size = (size, size)
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            warnings.warn("range should be of kind (min, max)")

        self.interpolation = interpolation
        self.scale = scale
        self.ratio = ratio

    @staticmethod
    def get_params(img, scale, ratio):
        """Get parameters for ``crop`` for a random sized crop.

        Args:
            img (PIL Image): Image to be cropped.
            scale (tuple): range of size of the origin size cropped
            ratio (tuple): range of aspect ratio of the origin aspect ratio cropped

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
                sized crop.
        """
        width, height = _get_image_size(img)
        area = height * width

        for attempt in range(10):
            target_area = random.uniform(*scale) * area
            log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
            aspect_ratio = math.exp(random.uniform(*log_ratio))

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < w <= width and 0 < h <= height:
                i = random.randint(0, height - h)
                j = random.randint(0, width - w)
                return i, j, h, w, height, width

        # Fallback to central crop
        in_ratio = float(width) / float(height)
        if (in_ratio < min(ratio)):
            w = width
            h = int(round(w / min(ratio)))
        elif (in_ratio > max(ratio)):
            h = height
            w = int(round(h * max(ratio)))
        else:  # whole image
            w = width
            h = height
        i = (height - h) // 2
        j = (width - w) // 2
        return i, j, h, w, height, width

    def __call__(self, img, same_two=False):
        """
        Args:
            img (PIL Image): Image to be cropped and resized.

        Returns:
            PIL Image: Randomly cropped and resized image.
        """
        # rank = torch.distributed.get_rank()
        if same_two:
            img, coord = img
            coord, params = coord
            # if rank == 0:
            #     print(self.__class__.__name__, "params:", params)
        if not same_two or params is None:
            i, j, h, w, height, width = self.get_params(img, self.scale, self.ratio)
        else:
            i, j, h, w, height, width = params
        # if same_two:
        #     if rank == 0:
        #         print(self.__class__.__name__, "params:", params)
        params = [i, j, h, w, height, width]
        coord = torch.Tensor([float(j) / (width - 1), float(i) / (height - 1),
                              float(j + w - 1) / (width - 1), float(i + h - 1) / (height - 1)])
        if same_two:
            params = {self.__class__.__name__: params}
            coord = [coord, params]
        return F.resized_crop(img, i, j, h, w, self.size, self.interpolation), coord

    def __repr__(self):
        interpolate_str = _pil_interpolation_to_str[self.interpolation]
        format_string = self.__class__.__name__ + '(size={0}'.format(self.size)
        format_string += ', scale={0}'.format(tuple(round(s, 4) for s in self.scale))
        format_string += ', ratio={0}'.format(tuple(round(r, 4) for r in self.ratio))
        format_string += ', interpolation={0})'.format(interpolate_str)
        return format_string
