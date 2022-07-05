# import numpy as np
# import random
# import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms import functional as trF

from contrast.data import transform


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

    def __init__(self,
                 transforms,
                 same_two=False,
                 optical_flow_model=None,
                 *,
                 alpha_1=0.01,
                 alpha_2=0.5):
        self.transforms = transforms
        self.same_two = same_two
        self.optical_flow_model = optical_flow_model
        self.alpha_1 = alpha_1
        self.alpha_2 = alpha_2
        self.get_crop_size()

    def get_crop_size(self):
        for t in self.transforms:
            if 'RandomResizedCrop' in t.__class__.__name__:
                self.crop_size = t.size

    def get_coord(self, imgs, size=(32, 32), return_two=True):
        if self.optical_flow_model:
            self.optical_flow_model.eval()
            flow_fwds = torch.stack([
                self.optical_flow_model(img1 * 255,
                                        img2 * 255,
                                        upsample=False,
                                        test_mode=True)[0]
                for img1, img2 in zip(imgs[:-1], imgs[1:])
            ])
            flow_bwds = torch.stack([
                self.optical_flow_model(img1 * 255,
                                        img2 * 255,
                                        upsample=False,
                                        test_mode=True)[0]
                for img1, img2 in zip(imgs[1:][::-1], imgs[:-1][::-1])
            ])
            flow_fwd = concat_flow(flow_fwds)
            flow_bwd = concat_flow(flow_bwds)
            coord1, coord2, mask = forward_backward_consistency(
                flow_fwd, flow_bwd, alpha_1=self.alpha_1, alpha_2=self.alpha_2)
            coord = [coord1, coord2]
        else:
            img1 = imgs[0]
            nb, _, ht, wd = imgs[0].shape
            new_ht, new_wd = ht // size[0], wd // size[1]
            coords1 = torch.meshgrid(torch.arange(new_ht),
                                     torch.arange(new_wd))
            coords1 = normalize_coord(
                torch.stack(coords1[::-1], dim=0).float()).to(img1.device)
            coord = coords1.clone()
            if return_two:
                coords2 = coords1.clone()
                coord = [coord1, coords2]
            mask = torch.ones(nb, new_ht, new_wd, dtype=bool).to(img1.device)
        return coord, mask

    def two_transform(self, image, device):
        assert isinstance(image, list) or isinstance(image, tuple)
        img1, img2 = image[0], image[-1]
        if F._is_pil_image(img1) and F._is_pil_image(img2):
            to_tensor_transform = transforms.ToTensor()
            img1 = to_tensor_transform(img1).to(device)
            img2 = to_tensor_transform(img2).to(device)

        coord1, coord2, mask = self.get_coord(image, (32, 32))

        _, height, width = img1.shape
        grid1 = torch.meshgrid(torch.arange(height), torch.arange(width))
        grid1 = centerize_grid(torch.stack(grid1[::-1], dim=0).float()).float()
        grid1 = grid1.to(img1.device)
        grid2 = grid1.clone()

        coord1_cent = denormalize_cenetrize_coord(coord1)
        coord2_cent = denormalize_cenetrize_coord(coord2)

        is_two_transform = isinstance(self.transforms, tuple)
        if is_two_transform:
            sample1 = self.transforms[0]({
                'image': img1.unsqueeze(0),
                "grid": grid1.unsqueeze(0),
                'coord': coord1_cent.unsqueeze(0)
            })
            sample2 = self.transforms[-1]({
                'image': img2.unsqueeze(0),
                "grid": grid2.unsqueeze(0),
                'coord': coord2_cent.unsqueeze(0)
            })
        else:
            sample1 = self.transforms({
                'image': img1.unsqueeze(0),
                "grid": grid1.unsqueeze(0),
                'coord': coord1_cent.unsqueeze(0)
            })
            sample2 = self.transforms({
                'image': img2.unsqueeze(0),
                "grid": grid2.unsqueeze(0),
                'coord': coord2_cent.unsqueeze(0)
            })


        img1, img2 = sample1['image'], sample2['image']
        grid1 = normalize_grid_ceterized(sample1["grid"]).detach()
        grid2 = normalize_grid_ceterized(sample2["grid"]).detach()

        img1 = F.grid_sample(img1,
                             grid1.permute(0, 2, 3, 1),
                             align_corners=True)
        img2 = F.grid_sample(img2,
                             grid2.permute(0, 2, 3, 1),
                             align_corners=True)
        # img1 = F.interpolate(img1,
        #                      self.crop_size,
        #                      mode="bilinear",
        #                      align_corners=False)
        # img2 = F.interpolate(img2,
        #                      self.crop_size,
        #                      mode="bilinear",
        #                      align_corners=False)

        coords1 = normalize_coord_centrized(sample1['coord'][0]).detach()
        coords2 = normalize_coord_centrized(sample2['coord'][0]).detach()
        # mask = (sample0['mask'] & sample1['mask']).detach()
        mask = mask.detach()
        mask = mask & (torch.abs(coords1[0]) < 1) & (torch.abs(coords1[1]) < 1)
        mask = mask & (torch.abs(coords2[0]) < 1) & (torch.abs(coords2[1]) < 1)
        return [img1[0], img2[0]], [[coords1, coords2], mask]

    def __call__(self, img, coord=None, device=None):
        """
        Augument images and get pixel correspondence between them.
        """
        if device is None:
            device_name = "cuda" if torch.cuda.is_available() else "cpu"
            if device_name == "cuda":
                rank = 0
                if torch.distributed.is_initialized():
                    rank = torch.distributed.get_rank()
                ngpus = torch.cuda.device_count()
                local_rank = rank % ngpus
                # torch.cuda.set_device(local_rank)
                device = torch.device(device_name, local_rank)
            else:
                device = torch.device(device_name)
        if isinstance(img, list) or isinstance(img, tuple):
            if not isinstance(coord, list) and not isinstance(coord, tuple):
                coord = [coord, coord]
            return self.two_transform(img, device)

        img = img.to(device)
        self.optical_flow_model = None
        coord, mask = self.get_coord([img], (32, 32), False)
        _, height, width = img.shape

        grid = torch.meshgrid(torch.arange(height), torch.arange(width))
        grid = centerize_grid(torch.stack(grid[::-1], dim=0).float()).float()
        grid = grid.to(img.device)

        coord_cent = denormalize_cenetrize_coord(coord)

        sample = self.transforms({
            'image': img.unsqueeze(0),
            "grid": grid.unsqueeze(0),
            'coord': coord_cent.unsqueeze(0)
        })

        img = sample['image']
        grid = normalize_grid_ceterized(sample["grid"]).detach()

        img = F.grid_sample(img, grid.permute(0, 2, 3, 1), align_corners=True)
        # img = F.interpolate(img,
        #                     self.crop_size,
        #                     mode="bilinear",
        #                     align_corners=False)

        coord = normalize_coord_centrized(sample['coord'][0]).detach()
        mask = mask.detach()
        mask = mask & (torch.abs(coord[0]) < 1) & (torch.abs(coord[1]) < 1)

        return img[0], [coord, mask]

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
        # nb, height, width = image.shape[0], image.shape[2], image.shape[3]
        nb = image.shape[0]
        # random scale
        if self.num_steps > 0:
            index = self.get_index(nb)
            scale_now = self.scale_steps[index].to(image.device)
        else:
            scale_now = (self.get_scale_coef(nb).to(image.device) *
                         (self.max_scale - self.min_scale) + self.min_scale)
        grid_scaled = torch.einsum("b,bnhw->bnhw", 1 / scale_now, grid)
        # image = F.grid_sample(image,
        #                       grid_scaled.permute(0, 2, 3, 1),
        #                       align_corners=True)

        coord = torch.einsum("b,bnhw->bnhw", scale_now, coord)
        return {"image": image, "grid": grid_scaled, "coord": coord}

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        format_string += "min_scale={0}".format(self.min_scale)
        format_string += ", max_scale={0}".format(self.max_scale)
        format_string += ", num_steps={0}".format(self.num_steps)
        format_string += ", step_size={0})".format(self.step_size)
        return format_string


class RandomResizedCrop(nn.Module):

    def __init__(self, size):
        super().__init__()
        if isinstance(size, int):
            size = (size, size)
        self.size = size
        self.crop_height = size[0]
        self.crop_width = size[1]

    def forward(self, sample):
        image, grid, coord = sample["image"], sample["grid"], sample["coord"]
        nb = image.shape[0]
        h, w = image.shape[-2:]
        scale_now_h = h / self.crop_height
        scale_now_w = w / self.crop_width
        scale_now = torch.tensor([scale_now_w, scale_now_h]).repeat(nb, 1)
        scale_now = scale_now.to(grid.device)
        grid_scaled = torch.einsum("bn,bnhw->bnhw", 1 / scale_now, grid)

        coord = torch.einsum("bn,bnhw->bnhw", scale_now, coord)

        pad_width, pad_height = max(w,
                                    self.crop_width), max(h, self.crop_height)
        pad_width = self.crop_width - w if w < self.crop_width else 0
        pad_height = self.crop_height - h if h < self.crop_height else 0
        new_h, new_w = h + pad_height, w + pad_width

        x1 = torch.randint(-(new_w - self.crop_width) // 2,
                           (new_w - self.crop_width) // 2, (nb, 1, 1))
        y1 = torch.randint(-(new_h - self.crop_height) // 2,
                           (new_h - self.crop_height) // 2, (nb, 1, 1))
        # rank = torch.distributed.get_rank()
        # print(f"rank: {rank}", x1, y1)
        x1 = x1 / (new_w - self.crop_width)
        y1 = y1 / (new_h - self.crop_height)
        # print(f"rank: {rank}", x1, y1)
        x1 = x1.to(grid.device)
        y1 = y1.to(grid.device)
        # x1 = torch.randint(0, new_w - self.crop_width)

        grid[:, 0] = grid_scaled[:, 0] - x1
        grid[:, 1] = grid_scaled[:, 1] - y1

        coord[:, 0] = coord[:, 0] + x1
        coord[:, 1] = coord[:, 1] + y1

        return {"image": image, "grid": grid, "coord": coord}

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        format_string += "size={0})".format(self.size)
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
        # nb, height, width = image.shape[0], image.shape[2], image.shape[3]
        nb = image.shape[0]
        rotate_degree = (2 * self.get_rotate_coef(nb) - 1) * self.degree
        matrix, inv_matrix = _get_rotation_matrix(rotate_degree)
        matrix, inv_matrix = matrix.to(image.device), inv_matrix.to(
            image.device)
        grid_rotated = torch.einsum("bmn,bnhw->bmhw", inv_matrix, grid)
        # image_rotated = F.grid_sample(image,
        #                               grid_rotated.permute(0, 2, 3, 1),
        #                               align_corners=True)
        coord_rotated = torch.einsum("bmn,bnhw->bmhw", matrix, coord)
        return {"image": image, "grid": grid_rotated, "coord": coord_rotated}

    def __repr__(self):
        format_string = self.__class__.__name__ + "(degree={0})".format(
            self.degree)
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


@torch.no_grad()
def concat_flow(flows):
    _, _, ht, wd = flows.shape
    coords1 = torch.meshgrid(torch.arange(ht), torch.arange(wd))
    coords1 = normalize_coord(torch.stack(coords1[::-1],
                                          dim=0).float()).to(flows.device)
    coords2 = coords1.clone()
    for flow in flows:
        flow_interpolate = grid_sample_flow(normalize_flow(flow), coords2)
        coords2 = coords2 + flow_interpolate
    return coords2 - coords1


@torch.no_grad()
def normalize_coord(coords):
    _, ht, wd = coords.shape
    coords_norm = coords.clone()
    coords_norm[0] = 2 * coords_norm[0] / (wd - 1) - 1
    coords_norm[1] = 2 * coords_norm[1] / (ht - 1) - 1
    return coords_norm


@torch.no_grad()
def normalize_flow(flow):
    _, ht, wd = flow.shape
    flow_norm = flow.clone()
    flow_norm[0] = 2 * flow_norm[0] / (wd - 1)
    flow_norm[1] = 2 * flow_norm[1] / (ht - 1)
    return flow_norm


@torch.no_grad()
def denormalize_cenetrize_coord(coords_norm):
    _, ht, wd = coords_norm.shape
    coords_cent = coords_norm.clone()
    coords_cent[0] = coords_cent[0] * (wd - 1)
    coords_cent[1] = coords_cent[1] * (ht - 1)
    return coords_cent


@torch.no_grad()
def normalize_coord_centrized(coords_cent):
    _, ht, wd = coords_cent.shape
    coords_norm = coords_cent.clone()
    coords_norm[0] = coords_norm[0] / (wd - 1)
    coords_norm[1] = coords_norm[1] / (ht - 1)
    return coords_norm


@torch.no_grad()
def grid_sample_flow(flow, coords_norm):
    flow_interpolate = F.grid_sample(flow,
                                     coords_norm.permute(0, 2, 3, 1),
                                     align_corners=True)
    return flow_interpolate


# implement: https://arxiv.org/pdf/1711.07837.pdf
@torch.no_grad()
def forward_backward_consistency(flow_fwd,
                                 flow_bwd,
                                 alpha_1=0.01,
                                 alpha_2=0.5):
    flow_fwd = flow_fwd.detach()
    flow_bwd = flow_bwd.detach()

    _, ht, wd = flow_fwd.shape
    coord1 = torch.meshgrid(torch.arange(ht), torch.arange(wd))
    coord1 = normalize_coord(torch.stack(coord1[::-1],
                                         dim=0).float()).to(flow_fwd.device)

    coord2 = coord1 + flow_fwd
    mask = (torch.abs(coord1[0]) < 1) & (torch.abs(coord2[1]) < 1)

    flow_bwd_interpolate = grid_sample_flow(flow_bwd.unsqueeze(0),
                                            coord2.unsqueeze(0))[0]
    flow_cycle = flow_fwd + flow_bwd_interpolate

    flow_cycle_norm = (flow_cycle**2).sum(1)
    eps = alpha_1 * ((flow_fwd**2).sum(1) +
                     (flow_bwd_interpolate**2).sum(1)) + alpha_2

    mask = mask & ((flow_cycle_norm - eps) <= 0)
    return coord1, coord2, mask
