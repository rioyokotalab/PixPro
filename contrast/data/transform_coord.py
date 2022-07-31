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

from typing import List, Tuple, Union

from torchvision.transforms import functional as F
import torch.nn.functional as nnF

from contrast.flow import InputPadder

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


def get_init_grid(grid_size, normalize_type=None, is_corner=True):
    height, width = grid_size
    grid = torch.meshgrid(torch.arange(height), torch.arange(width))
    grid = torch.stack(grid[::-1], dim=0).float()
    if not is_corner:
        grid = grid + 0.5
    if normalize_type is not None:
        if normalize_type == "norm":
            grid = normalize_grid(grid)
        elif normalize_type == "center":
            grid = centerize_grid(grid)
    return grid


def get_crop_size(transforms: List):
    crop_size = None
    for t in transforms:
        if 'RandomResizedCrop' in t.__class__.__name__:
            crop_size = t.size
    return crop_size


def load_img_for_raft(img: Image):
    img = np.array(img).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None]
    # return img[None].cuda()


def get_coord(size, coord, is_corner=True):
    H, W = size
    # C, H, W = coord.shape
    array = torch.meshgrid(torch.arange(H), torch.arange(W))
    array = torch.stack(array[::-1], dim=0).float()
    # x_array = torch.arange(0., float(W), dtype=coord.dtype, device=coord.device).view(1, 1, -1).repeat(1, H, 1)
    # y_array = torch.arange(0., float(H), dtype=coord.dtype, device=coord.device).view(1, -1, 1).repeat(1, 1, W)
    # [bs, 1, 1]
    bin_width = ((coord[2] - coord[0]) / W).view(1, 1)
    bin_height = ((coord[3] - coord[1]) / H).view(1, 1)
    # [bs, 1, 1]
    start_x = coord[0].view(1, 1)
    start_y = coord[1].view(1, 1)

    # [bs, 7, 7]
    if is_corner:
        array[0] = array[0] * bin_width + start_x
        array[1] = array[1] * bin_height + start_y
    else:
        array[0] = (array[0] + 0.5) * bin_width + start_x
        array[1] = (array[1] + 0.5) * bin_height + start_y

    array = 2 * array - 1
    return array


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

    def __init__(self, transforms: Union[Tuple[List], List],
                 same_two=False, two_crop=False,
                 optical_flow_model=None,
                 *,
                 alpha_1=0.01, alpha_2=0.5):
        if isinstance(transforms, tuple):
            self.transforms: Tuple[List] = transforms
        else:
            self.transforms = (transforms,)
        self.same_two = same_two
        self.two_crop = two_crop
        num_transform = len(self.transforms)
        if num_transform > 2:
            raise Exception(f"Unsupport for # of transforms is {num_transform}")

        self.flow_model = optical_flow_model
        self.alpha_1, self.alpha_2 = alpha_1, alpha_2
        self.use_flow = self.flow_model is not None

        crop_size_list = []
        for transforms in self.transforms:
            tmp_crop_size = get_crop_size(transforms)
            crop_size_list.append(tmp_crop_size)

        tmp_crop_size = crop_size_list[0]
        for crop_size in crop_size_list[1:]:
            if tmp_crop_size is not None:
                assert tmp_crop_size == crop_size
        self.crop_size = tmp_crop_size
        # self.crop_size = None

    def calc_optical_flow(self, pil_imgs, out_size=None, is_norm=True):
        assert self.flow_model is not None

        imgs = [load_img_for_raft(img) for img in pil_imgs]
        padder = InputPadder(imgs[0].shape)
        imgs = padder.pad(*imgs)

        self.flow_model.eval()
        flow_fwds = torch.stack([
            self.flow_model(img1, img2, test_mode=True)[0]
            for img1, img2 in zip(imgs[:-1], imgs[1:])
        ])
        flow_bwds = torch.stack([
            self.flow_model(img1, img2, test_mode=True)[0]
            for img1, img2 in zip(imgs[1:][::-1], imgs[:-1][::-1])
        ])
        flow_fwd = concat_flow(flow_fwds, is_norm)
        flow_bwd = concat_flow(flow_bwds, is_norm)
        coord1, coord2, mask = self.forward_backward_consistency(
            flow_fwd, flow_bwd, is_norm)

        return [coord1, coord2], [flow_fwd, flow_bwd], mask

    # implement: https://arxiv.org/pdf/1711.07837.pdf
    @torch.no_grad()
    def forward_backward_consistency(self, flow_fwd_s, flow_bwd_s, is_normed=True):
        flow_fwd = flow_fwd_s.clone().detach()
        flow_bwd = flow_bwd_s.clone().detach()
        alpha_1, alpha_2 = self.alpha_1, self.alpha_2

        ht, wd = flow_fwd.shape[-2:]
        coord1 = torch.meshgrid(torch.arange(ht), torch.arange(wd))
        coord1 = torch.stack(coord1[::-1], dim=0).float()
        if is_normed:
            coord1 = normalize_grid(coord1)
        coord1 = coord1.to(flow_fwd.device)

        coord2 = coord1 + flow_fwd
        coord2 = coord2.squeeze(0)
        if is_normed:
            coord2_norm = coord2.clone()
        else:
            coord2_norm = normalize_grid(coord2)
        ndim = coord2_norm.ndim
        if ndim == 3:
            coord2_norm = coord2_norm.unsqueeze(0)
        ndim = flow_fwd.ndim
        if ndim == 3:
            flow_fwd = flow_fwd.unsqueeze(0)
            flow_bwd = flow_bwd.unsqueeze(0)

        mask = (torch.abs(coord2_norm[:, 0]) < 1) & (torch.abs(coord2_norm[:, 1]) < 1)
        flow_bwd_interpolate = grid_sample_flow(flow_bwd, coord2_norm)
        flow_cycle = flow_fwd + flow_bwd_interpolate

        flow_cycle_norm = (flow_cycle**2).sum(1)
        eps = alpha_1 * ((flow_fwd**2).sum(1) +
                         (flow_bwd_interpolate**2).sum(1)) + alpha_2

        mask = mask & ((flow_cycle_norm - eps) <= 0)
        return coord1, coord2, mask

    def __call__(self, imgs, coord=None):
        is_list = isinstance(imgs, list) or isinstance(imgs, tuple)
        if is_list:
            image1, image2 = imgs[0], imgs[-1]
        else:
            image1, image2 = imgs, imgs

        if self.use_flow:
            flow_outs = self.calc_optical_flow(imgs, is_norm=False)
            flow_grids, flow_fwd_bwd, mask = flow_outs
            flow_grid1, flow_grid2 = flow_grids
            flow_fwd, flow_bwd = flow_fwd_bwd

        grid_size = self.crop_size
        if grid_size is None:
            w, h = _get_image_size(image1)
            grid_size = (h, w)
        # w, h = _get_image_size(image1)
        # grid_size = (h, w)
        coord_size = (grid_size[0] // 8, grid_size[1] // 8)

        is_corner = False
        # normalize_type = None
        # normalize_type = "norm"
        normalize_type = "center"
        init_grid = get_init_grid(grid_size, normalize_type, is_corner)
        init_coord = get_init_grid(coord_size, normalize_type, is_corner)
        # init_mask = torch.ones(coord_size, dtype=bool)
        if isinstance(coord, list):
            coord, params = coord
            coord = [(init_grid.clone(), init_coord.clone()), coord]
            coord = [coord, params]
        else:
            coord = [(init_grid.clone(), init_coord.clone()), coord]
            coord = [coord, None]

        img, coord = self.main_call(image1, coord, self.transforms[0])
        if self.two_crop:
            in_coord = coord if self.same_two else [coord, None]
            in_coord, params = in_coord
            in_coord = [(init_grid.clone(), init_coord.clone()), in_coord]
            # in_coord = [(init_grid.clone(), flow_grid2.clone()), in_coord]
            in_coord = [in_coord, params]
            img2, coord2 = self.main_call(image2, in_coord, self.transforms[-1])
        if self.same_two:
            coord, _ = coord
            if self.two_crop:
                coord2, _ = coord2

        # official coord
        grids, coord = coord
        calc_coord = get_coord(grid_size, coord, is_corner)
        if self.two_crop:
            grids2, coord2 = coord2
            calc_coord2 = get_coord(grid_size, coord2, is_corner)

        # my coord
        grid, mycoord = grids
        mycoord = normalize_grid_ceterized(mycoord)
        # mycoord = F.resize(mycoord, self.crop_size)
        grid = normalize_grid_ceterized(grid)
        mask = (torch.abs(mycoord[0]) < 1) & (torch.abs(mycoord[1]) < 1)
        coord = [coord, grid.clone()]
        # coord = mycoord.clone()
        if self.two_crop:
            grid2, mycoord2 = grids2
            mycoord2 = normalize_grid_ceterized(mycoord2)
            # mycoord2 = F.resize(mycoord2, self.crop_size)

            if self.use_flow:
                grid_norm2 = normalize_grid_ceterized(grid2)
                if flow_fwd.ndim == 3:
                    flow_fwd = flow_fwd.unsqueeze(0)
                flow_t = grid_sample_flow(flow_fwd, grid_norm2.unsqueeze(0))
                grid2_flow = grid2 + flow_t
                grid2_flow = normalize_grid_ceterized(grid2_flow)
                grid2_flow = grid2_flow.squeeze(0)

            grid2 = normalize_grid_ceterized(grid2)
            mask2 = (torch.abs(mycoord2[0]) < 1) & (torch.abs(mycoord2[1]) < 1)
            mask = mask & mask2
            coord2 = [coord2, grid2.clone()]
            # coord2 = mycoord2.clone()

        # if torch.distributed.get_rank() == 0:
        #     print(calc_coord.shape, mycoord.shape, grid.shape)
        #     print(calc_coord, mycoord, grid, coord, calc_coord==grid, calc_coord == mycoord)
        #     print(calc_coord2, mycoord2, grid2, coord2, calc_coord2==grid2, calc_coord2 == mycoord2)
        #     print(calc_coord, grid, coord)
        #     print(calc_coord2, grid2, coord2)
        #     print(mycoord[0][mask], mycoord[1][mask])
        #     print(mycoord2[0][mask], mycoord2[1][mask])
        #     print(grid[0][mask], grid[1][mask])
        #     print(grid2[0][mask], grid2[1][mask])

        img_tmp = img.unsqueeze(0)
        grid_tmp = grid.unsqueeze(0).permute(0, 2, 3, 1)
        # img_tmp = nnF.grid_sample(img_tmp, grid_tmp, align_corners=True)
        img_tmp = nnF.grid_sample(img_tmp, grid_tmp, align_corners=is_corner)
        # img_tmp = F.resize(img_tmp[0], list(self.crop_size))
        img = img_tmp[0].clone()
        if self.two_crop:
            img2_tmp = img2.unsqueeze(0)
            grid2_tmp = grid2.unsqueeze(0).permute(0, 2, 3, 1)
            # img2_tmp = nnF.grid_sample(img2_tmp, grid2_tmp, align_corners=True)
            img2_tmp = nnF.grid_sample(img2_tmp, grid2_tmp, align_corners=is_corner)
            # img2_tmp = F.resize(img2_tmp[0], list(self.crop_size))
            img2 = img2_tmp[0].clone()

        if self.two_crop:
            return (img, coord), (img2, coord2)
        return img, coord

    def main_call(self, img, coord=None, transforms=None):

        if transforms is None:
            transforms = self.transforms[0]

        params, in_params = None, None
        tmp_len_key = {}
        is_same_two_coord = isinstance(coord, list)
        if is_same_two_coord:
            coord, params = coord

        for t in transforms:
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

            if isinstance(t, Compose):
                t_same_two = t.same_two
                if self.same_two:
                    coord = [coord, in_params]
                elif t_same_two:
                    coord = [coord, params]
                img, coord = t(img, coord=coord)
            elif 'RandomResizedCropCoord' in t.__class__.__name__:
                if self.same_two:
                    coord = [coord, in_params]
                # in_img = [img, coord] if self.same_two else img
                in_img = [img, coord]
                img, coord = t(in_img, same_two=self.same_two)
            elif 'RandomRescaleCoord' in t.__class__.__name__:
                if self.same_two:
                    coord = [coord, in_params]
                # in_img = [img, coord] if self.same_two else img
                in_img = [img, coord]
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
        is_coord = isinstance(img, list)
        if is_coord:
            img, coord = img
        if same_two:
            coord, params = coord

        is_calc_coord = False
        if is_coord:
            is_calc_coord = isinstance(coord, list)

        if is_calc_coord:
            grids, coord = coord
            grid, mycoord = grids
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
        # resize
        new_width, new_height = int(scale_now * width), int(scale_now * height)
        i, j = (new_width - width) / 2, (new_height - height) / 2
        coord = torch.Tensor([j / (width - 1), i / (height - 1),
                              (j + width - 1) / (width - 1), (i + height - 1) / (height - 1)])

        if is_calc_coord:
            grid = grid / scale_now
            mycoord = mycoord * scale_now
            coord = [(grid, mycoord), coord]
        # img = img.resize((new_width, new_height), Image.BICUBIC)
        # fill = [0.0] * F._get_image_num_channels(img)
        # img = F.affine(img, 0.0, (0, 0), scale_now, (0.0, 0.0), fill=fill)
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
            is_calc_coord = isinstance(coord, list)
            if is_calc_coord:
                grids, coord = coord
                grid, mycoord = grids

            coord_new = coord.clone()
            coord_new[0] = coord[2]
            coord_new[2] = coord[0]

            if is_calc_coord:
                grid = F.hflip(grid)
                mycoord = F.hflip(mycoord)
                coord_new = [(grid, mycoord), coord_new]

            # return F.hflip(img), coord_new
            return img, coord_new
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
            is_calc_coord = isinstance(coord, list)
            if is_calc_coord:
                grids, coord = coord
                grid, mycoord = grids

            coord_new = coord.clone()
            coord_new[1] = coord[3]
            coord_new[3] = coord[1]

            if is_calc_coord:
                grid = F.vflip(grid)
                mycoord = F.vflip(mycoord)
                coord_new = [(grid, mycoord), coord_new]

            # return F.vflip(img), coord_new
            return img, coord_new
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
        is_coord = isinstance(img, list)
        if is_coord:
            img, coord = img
        if same_two and is_coord:
            coord, params = coord
            # if rank == 0:
            #     print(self.__class__.__name__, "params:", params)
        is_calc_coord = False
        if is_coord:
            is_calc_coord = isinstance(coord, list)
        if is_calc_coord:
            grids, coord = coord
            grid, mycoord = grids
            grid_h, grid_w = grid.shape[-2:]
            coord_h, coord_w = mycoord.shape[-2:]

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
        if is_calc_coord:
            scale_now_h = height / h
            scale_now_w = width / w
            # diff_x1 = 2 * j / (grid_w - 1)
            # diff_y1 = 2 * i / (grid_h - 1)
            diff_x1 = (2 * j + w - width + 1) / (width - 1)
            diff_y1 = (2 * i + h - height + 1) / (height - 1)
            grid[0] = grid[0] / scale_now_w
            grid[1] = grid[1] / scale_now_h
            grid[0] = grid[0] + (diff_x1 * grid_w)
            grid[1] = grid[1] + (diff_y1 * grid_h)
            mycoord[0] = mycoord[0] - (diff_x1 * coord_w)
            mycoord[1] = mycoord[1] - (diff_x1 * coord_h)
            mycoord[0] = mycoord[0] * scale_now_w
            mycoord[1] = mycoord[1] * scale_now_h
            # mycoord[0] = mycoord[0] / scale_now_w
            # mycoord[1] = mycoord[1] / scale_now_h
            # mycoord[0] = mycoord[0] + (diff_x1 * coord_w)
            # mycoord[1] = mycoord[1] + (diff_x1 * coord_h)
            coord = [(grid, mycoord), coord]

        if same_two:
            params = {self.__class__.__name__: params}
            coord = [coord, params]
        # return F.resized_crop(img, i, j, h, w, self.size, self.interpolation), coord
        return img, coord

    def __repr__(self):
        interpolate_str = _pil_interpolation_to_str[self.interpolation]
        format_string = self.__class__.__name__ + '(size={0}'.format(self.size)
        format_string += ', scale={0}'.format(tuple(round(s, 4) for s in self.scale))
        format_string += ', ratio={0}'.format(tuple(round(r, 4) for r in self.ratio))
        format_string += ', interpolation={0})'.format(interpolate_str)
        return format_string


@torch.no_grad()
def normalize_grid(grid):
    ndim = grid.ndim
    assert ndim == 3 or ndim == 4
    ht, wd = grid.shape[-2:]
    grid_norm = grid.clone()
    if ndim == 3:
        grid_norm[0] = 2 * grid_norm[0] / (wd - 1) - 1
        grid_norm[1] = 2 * grid_norm[1] / (ht - 1) - 1
    elif ndim == 4:
        grid_norm[:, 0] = 2 * grid_norm[:, 0] / (wd - 1) - 1
        grid_norm[:, 1] = 2 * grid_norm[:, 1] / (ht - 1) - 1
    return grid_norm


@torch.no_grad()
def normalize_grid_ceterized(grid_cent):
    ndim = grid_cent.ndim
    assert ndim == 3 or ndim == 4
    ht, wd = grid_cent.shape[-2:]
    grid_norm = grid_cent.clone()
    if ndim == 3:
        grid_norm[0] = grid_norm[0] / (wd - 1)
        grid_norm[1] = grid_norm[1] / (ht - 1)
    elif ndim == 4:
        grid_norm[:, 0] = grid_norm[:, 0] / (wd - 1)
        grid_norm[:, 1] = grid_norm[:, 1] / (ht - 1)
    return grid_norm


@torch.no_grad()
def centerize_grid(grid):
    ndim = grid.ndim
    assert ndim == 3 or ndim == 4
    ht, wd = grid.shape[-2:]
    grid_cent = grid.clone()
    if ndim == 3:
        grid_cent[0] = 2 * grid_cent[0] - (wd - 1)
        grid_cent[1] = 2 * grid_cent[1] - (ht - 1)
    elif ndim == 4:
        grid_cent[:, 0] = 2 * grid_cent[:, 0] - (wd - 1)
        grid_cent[:, 1] = 2 * grid_cent[:, 1] - (ht - 1)
    return grid_cent


@torch.no_grad()
def denormalize_cenetrize_grid(grid_norm):
    ndim = grid_norm.ndim
    assert ndim == 3 or ndim == 4
    ht, wd = grid_norm.shape[-2:]
    grid_cent = grid_norm.clone()
    if ndim == 3:
        grid_cent[0] = grid_cent[0] * (wd - 1)
        grid_cent[1] = grid_cent[1] * (ht - 1)
    elif ndim == 4:
        grid_cent[:, 0] = grid_cent[:, 0] * (wd - 1)
        grid_cent[:, 1] = grid_cent[:, 1] * (ht - 1)
    return grid_cent


@torch.no_grad()
def concat_flow(flows, is_norm=True):
    ndim = flows.ndim
    ht, wd = flows.shape[-2:]
    coord1 = torch.meshgrid(torch.arange(ht), torch.arange(wd))
    coord1 = torch.stack(coord1[::-1], dim=0).float()
    if is_norm:
        coord1 = normalize_grid(coord1)
    coord1 = coord1.to(flows.device)

    coord2 = coord1.clone()
    for flow in flows:
        flow_tmp = flow.clone()
        if is_norm:
            coord2_norm = coord2.clone().unsqueeze(0)
            flow_tmp = normalize_flow(flow_tmp)
            if ndim < 4:
                flow_tmp = flow_tmp.unsqueeze(0)
        else:
            coord2_norm = normalize_grid(coord2).unsqueeze(0)
            if ndim < 4:
                flow_tmp = flow_tmp.unsqueeze(0)
        flow_interpolate = grid_sample_flow(flow_tmp, coord2_norm)
        coord2 = coord2 + flow_interpolate
    return coord2 - coord1


@torch.no_grad()
def normalize_flow(flow):
    ht, wd = flow.shape[-2:]
    flow_norm = flow.clone()
    flow_norm[0] = 2 * flow_norm[0] / (wd - 1)
    flow_norm[1] = 2 * flow_norm[1] / (ht - 1)
    return flow_norm


@torch.no_grad()
def grid_sample_flow(flow, coord_norm, is_corner=True):
    flow_interpolate = nnF.grid_sample(flow,
                                       coord_norm.permute(0, 2, 3, 1),
                                       align_corners=is_corner)
    return flow_interpolate


def down_flow(flow, times: int, mode='bilinear'):
    new_size = (flow.shape[-2] // times, flow.shape[-1] // times)
    return nnF.interpolate(flow, size=new_size, mode=mode, align_corners=True) / times


def up_flow(flow, times: int, mode='bilinear'):
    new_size = (times * flow.shape[-2], times * flow.shape[-1])
    return times * nnF.interpolate(flow, size=new_size, mode=mode, align_corners=True)
