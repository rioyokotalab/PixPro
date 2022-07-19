# import numpy as np
# import random
# import math

from PIL import Image
import torch
import torch.nn as nn
# import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms import functional as trF


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

    def __call__(self, img, ratio=8):
        nb, _, ht, wd = img.shape

        if isinstance(ratio, int):
            ratio = (ratio, ratio)
        assert len(ratio) == 2

        new_ht, new_wd = ht // ratio[0], wd // ratio[1]
        coords0 = torch.meshgrid(torch.arange(new_ht), torch.arange(new_wd))
        coords0 = normalize_coord(
            torch.stack(coords0[::-1], dim=0).float().repeat(nb, 1, 1,
                                                             1)).to(img0.device)
        coords1 = coords0.clone()
        mask = torch.ones(nb, new_ht, new_wd, dtype=bool).to(img0.device)


        grid = torch.meshgrid(torch.arange(ht), torch.arange(wd))
        # grid = normalize_grid(torch.stack(grid[::-1], dim=0).float()).float()
        grid = centerize_grid(torch.stack(grid[::-1], dim=0).float()).float()
        grid = grid.repeat(nb, 1, 1, 1).to(img.device)

        coords0_cent = denormalize_cenetrize_coord(coords0)
        coords1_cent = denormalize_cenetrize_coord(coords1)

        sample0 = self.aug({'image': img0, "grid": grid0, 'coord': coords0_cent})
        sample1 = self.aug({'image': img1, "grid": grid1, 'coord': coords1_cent})

        img0, img1 = sample0['image'], sample1['image']
        grid0 = normalize_grid_ceterized(sample0["grid"]).detach()
        grid1 = normalize_grid_ceterized(sample1["grid"]).detach()

        img0 = F.grid_sample(img0, grid0.permute(0, 2, 3, 1), align_corners=True)
        img1 = F.grid_sample(img1, grid1.permute(0, 2, 3, 1), align_corners=True)

        coords0 = normalize_coord_centrized(sample0['coord']).detach()
        coords1 = normalize_coord_centrized(sample1['coord']).detach()
        # mask = (sample0['mask'] & sample1['mask']).detach()
        mask = mask.detach()
        mask = mask & (torch.abs(coords0[:, 0]) < 1) & (torch.abs(coords0[:, 1]) < 1)
        mask = mask & (torch.abs(coords1[:, 0]) < 1) & (torch.abs(coords1[:, 1]) < 1)

        for t in self.transforms:
            sample = t(sample)
        img = sample["image"]
        coord = sample["grid"]
        coord_inv = sample["coord"]
        return img, [coord, coord_inv]

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class FlowAugmentationWrapper(nn.Module):

    def __init__(self, transform):
        super().__init__()
        self.transform = transform

    def forward(self, sample):
        # image, coord, mask = sample["image"], sample["coord"], sample["mask"]
        image, grid, coord = sample["image"], sample["grid"], sample["coord"]
        image = self.transform(image)
        # return {"image": image, "coord": coord, "mask": mask}
        return {"image": image, "grid": grid, "coord": coord}


class ColorJitter(transforms.ColorJitter):

    def __init__(self,
                 brightness=0,
                 contrast=0,
                 saturation=0,
                 hue=0,
                 is_random=True,
                 is_switch=True):
        super().__init__(brightness, contrast, saturation, hue)
        self.is_random, self.is_switch = is_random, is_switch
        self.num = 0

    def get_params(self, brightness, contrast, saturation, hue):
        if self.is_random:
            return transforms.ColorJitter.get_params(self.brightness, self.contrast,
                                                     self.saturation, self.hue)
        fn_idx = torch.arange(4)

        self.num = (self.num + 1) % 2 if self.is_switch else 0
        b = None if brightness is None else float(brightness[self.num])
        c = None if contrast is None else float(contrast[self.num])
        s = None if saturation is None else float(saturation[self.num])
        h = None if hue is None else float(hue[self.num])

        return fn_idx, b, c, s, h

    def __repr__(self):
        format_string = super().__repr__()[:-1]
        format_string += ", is_random={0}".format(self.is_random)
        format_string += ", is_switch={0})".format(self.is_switch)
        return format_string


class Resize(transforms.Resize):

    def __init__(self, size, interpolation=Image.BILINEAR):
        super().__init__(size, interpolation)

    def forward(self, sample):
        image, coord, mask = sample["image"], sample["coord"], sample["mask"]
        image = super().forward(image)
        return {"image": image, "coord": coord, "mask": mask}


class RandomRescale(nn.Module):

    def __init__(self, min_scale=0.5, max_scale=1.5, step_size=0.25):
        super().__init__()
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.step_size = step_size
        # discrete scales
        if (max_scale - min_scale) > step_size and step_size > 0.05:
            self.num_steps = int((max_scale - min_scale) / step_size + 1)
            self.scale_steps = torch.linspace(self.min_scale, self.max_scale,
                                              self.num_steps)
        elif (max_scale - min_scale) > step_size and step_size < 0.05:
            self.num_steps = 0
            self.scale_steps = torch.tensor([min_scale])
        else:
            self.num_steps = 1
            self.scale_steps = torch.tensor([min_scale])

    def get_index(self, nb):
        return torch.randint(0, self.num_steps, (nb, ))

    def get_scale_coef(self, nb):
        return torch.rand(nb)

    def forward(self, sample):
        # image, coord, mask = sample["image"], sample["coord"], sample["mask"]
        image, grid, coord = sample["image"], sample["grid"], sample["coord"]
        # nb, ht, wd = image.shape[0], image.shape[2], image.shape[3]
        nb = image.shape[0]
        # random scale
        if self.num_steps > 0:
            index = self.get_index(nb)
            scale_now = self.scale_steps[index].to(image.device)
        else:
            scale_now = (self.get_scale_coef(nb).to(image.device) *
                         (self.max_scale - self.min_scale) + self.min_scale)
        # grid = torch.meshgrid(torch.arange(ht), torch.arange(wd))
        # grid = normalize_grid(torch.stack(grid[::-1], dim=0).float()).to(image.device)
        # grid_scaled = torch.einsum("b,nhw->bnhw", 1 / scale_now, grid)
        grid_scaled = torch.einsum("b,bnhw->bnhw", 1 / scale_now, grid)
        # image = F.grid_sample(image,
        #                       grid_scaled.permute(0, 2, 3, 1),
        #                       align_corners=True)

        coord = torch.einsum("b,bnhw->bnhw", scale_now, coord)
        # mask = mask & (torch.abs(coord[:, 0]) < 1) & (torch.abs(coord[:, 1]) < 1)

        # return {"image": image, "coord": coord, "mask": mask}
        return {"image": image, "grid": grid_scaled, "coord": coord}

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        format_string += "min_scale={0}".format(self.min_scale)
        format_string += ", max_scale={0}".format(self.max_scale)
        format_string += ", num_steps={0}".format(self.num_steps)
        format_string += ", step_size={0})".format(self.step_size)
        return format_string


class RandomHorizontalFlip(nn.Module):

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, sample):
        # image, coord, mask = sample["image"], sample["coord"], sample["mask"]
        image, grid, coord = sample["image"], sample["grid"], sample["coord"]
        nb = image.shape[0]
        ind = torch.rand(nb) < self.p
        # image[ind] = trF.hflip(image[ind])
        grid[ind] = trF.hflip(grid[ind])
        coord[ind] = trF.hflip(coord[ind])

        # return {"image": image, "coord": coord, "mask": mask}
        return {"image": image, "grid": grid, "coord": coord}

    def __repr__(self):
        format_string = self.__class__.__name__ + "(p={0})".format(self.p)
        return format_string


class RandomRotation(nn.Module):

    def __init__(self, degree=30):
        super().__init__()
        self.degree = degree

    def get_rotate_coef(self, nb):
        return torch.rand(nb)

    def forward(self, sample):
        # image, coord, mask = sample["image"], sample["coord"], sample["mask"]
        image, grid, coord = sample["image"], sample["grid"], sample["coord"]
        # nb, ht, wd = image.shape[0], image.shape[2], image.shape[3]
        nb = image.shape[0]
        rotate_degree = (2 * self.get_rotate_coef(nb) - 1) * self.degree
        matrix, inv_matrix = _get_rotation_matrix(rotate_degree)
        matrix, inv_matrix = matrix.to(image.device), inv_matrix.to(image.device)
        # grid = torch.meshgrid(torch.arange(ht), torch.arange(wd))
        # grid = normalize_grid(torch.stack(grid[::-1], dim=0).float()).to(image.device)
        # grid_rotated = torch.einsum("bmn,nhw->bmhw", inv_matrix, grid)
        grid_rotated = torch.einsum("bmn,bnhw->bmhw", inv_matrix, grid)
        # image_rotated = F.grid_sample(image,
        #                               grid_rotated.permute(0, 2, 3, 1),
        #                               align_corners=True)
        coord_rotated = torch.einsum("bmn,bnhw->bmhw", matrix, coord)
        # mask = mask & (torch.abs(coord[:, 0]) < 1) & (torch.abs(coord[:, 1]) < 1)

        # return {"image": image_rotated, "coord": coord_rotated, "mask": mask}
        return {"image": image, "grid": grid_rotated, "coord": coord_rotated}

    def __repr__(self):
        format_string = self.__class__.__name__ + "(degree={0})".format(self.degree)
        return format_string


class Rescale(RandomRescale):

    def __init__(self, min_scale=0.5, max_scale=1.5, step_size=0.25, is_switch=True):
        super().__init__(min_scale, max_scale, step_size)
        self.num, self.is_switch = 0, is_switch
        self.len_scale_steps = len(self.scale_steps)

    def get_index(self, nb):
        if self.is_switch:
            self.num = (self.num + 1) % self.len_scale_steps
        index = [self.num for _ in range(nb)]
        return torch.tensor(index)

    def get_rotate_coef(self, nb):
        self.num = (self.num + 1) % 2 if self.is_switch else 1
        rotate_coef = torch.ones(nb) if self.num == 1 else torch.zeros(nb)
        return rotate_coef

    def __repr__(self):
        format_string = super().__repr__()[:-1]
        format_string += ", is_switch={0})".format(self.is_switch)
        return format_string


class HorizontalFlip(RandomHorizontalFlip):

    def __init__(self):
        super().__init__(p=1.0)


class Rotation(RandomRotation):

    def __init__(self, degree=30, is_switch=True):
        super().__init__(degree)
        self.num, self.is_switch = 0, is_switch

    def get_rotate_coef(self, nb):
        self.num = (self.num + 1) % 2 if self.is_switch else 1
        rotate_coef = torch.ones(nb) if self.num == 1 else torch.zeros(nb)
        return rotate_coef

    def __repr__(self):
        format_string = super().__repr__()[:-1]
        format_string += ", is_switch={0})".format(self.is_switch)
        return format_string


def _get_rotation_matrix(angles):
    rots = torch.deg2rad(angles)
    a = torch.cos(rots)
    b = -torch.sin(rots)
    c = -b
    d = a
    matrix = torch.stack([a, b, c, d], dim=-1).reshape(-1, 2, 2)
    inv_matrix = torch.stack([d, -b, -c, a], dim=-1).reshape(-1, 2, 2)
    return matrix, inv_matrix


@torch.no_grad()
def normalize_grid(grid):
    _, ht, wd = grid.shape
    grid_norm = grid.clone()
    grid_norm[0] = 2 * grid_norm[0] / (wd - 1) - 1
    grid_norm[1] = 2 * grid_norm[1] / (ht - 1) - 1
    return grid_norm


@torch.no_grad()
def normalize_grid_ceterized(grid_cent):
    _, ht, wd = grid_cent.shape
    grid_norm = grid_cent.clone()
    grid_norm[0] = grid_norm[0] / (wd - 1)
    grid_norm[1] = grid_norm[1] / (ht - 1)
    return grid_norm


@torch.no_grad()
def centerize_grid(grid):
    _, ht, wd = grid.shape
    grid_cent = grid.clone()
    grid_cent[0] = 2 * grid_cent[0] - (wd - 1)
    grid_cent[1] = 2 * grid_cent[1] - (ht - 1)
    return grid_cent
