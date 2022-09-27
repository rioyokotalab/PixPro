import os
import itertools

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
# from torchvision.utils import save_image
from PIL import ImageDraw
from PIL import Image


rank = torch.distributed.get_rank()


def prepare_imgs(coord_q, coord_k):
    if isinstance(coord_q, tuple):
        idx = None
        coord_q, test_imgs = coord_q
        coord_k, test_imgs2 = coord_k
        if isinstance(test_imgs, list):
            test_imgs, idx = test_imgs
            test_imgs2, _ = test_imgs2
        ndim = test_imgs.ndim
        img1, img2 = None, None
        if ndim > 4:
            img1 = test_imgs[:, 1]
            img2 = test_imgs[:, 2]
            test_imgs = test_imgs[:, 0]
            test_imgs2 = None
    return coord_q, coord_k, test_imgs, test_imgs2, img1, img2, idx


def prepare_dirs(out_root, test_imgs, test_imgs2, coord_q, coord_k, idx, img1, img2, is_calc_flow, is_pos):
    base_name = os.path.basename(out_root)
    is_reverse = base_name == "2"
    color = [(255, 165, 0), (0, 0, 255)]
    if is_reverse:
        color_tmp = color[0]
        color[0] = color[1]
        color[1] = color_tmp
    out_path_center = f"{out_root}/center"
    out_path = f"{out_root}/no_center"
    os.makedirs(out_path_center, exist_ok=True)
    os.makedirs(out_path, exist_ok=True)

    if test_imgs2 is not None:
        out_path_crop = f"{out_root}/crop"
        os.makedirs(out_path_crop, exist_ok=True)
        rec_imgs, crop_imgs = draw_rects(test_imgs, test_imgs2, coord_q, coord_k, color, idx, out_path_crop)
        test_imgs, img1, img2 = rec_imgs

    color = [(255, 0, 0), (0, 255, 0)]
    if is_reverse:
        color_tmp = color[0]
        color[0] = color[1]
        color[1] = color_tmp

    calc_flow_list = []

    if is_calc_flow:
        out_path_center_flo = f"{out_root}/center_flow"
        out_path_flo = f"{out_root}/no_center_flow"
        os.makedirs(out_path_center_flo, exist_ok=True)
        os.makedirs(out_path_flo, exist_ok=True)
        calc_flow_list = [out_path_flo, out_path_center_flo]

    out_path_pos = None
    if is_pos:
        out_path_pos = f"{out_root}/positive_pair"
        os.makedirs(out_path_pos, exist_ok=True)

    return out_path, out_path_center, color, test_imgs, img1, img2, calc_flow_list, out_path_pos


def calc_flow_grid_crop_size(coord_q, coord_k):
    q_grids, k_grids = [], []
    for (q_w, q_h), (k_w, k_h) in zip(coord_q[:, 6:8], coord_k[:, 6:8]):
        q_crop_w, q_crop_h = int(q_w.item()), int(q_h.item())
        k_crop_w, k_crop_h = int(k_w.item()), int(k_h.item())
        q_grid_tmp = torch.meshgrid(torch.arange(q_crop_h), torch.arange(q_crop_w))
        q_grid_tmp = torch.stack(q_grid_tmp[::-1]).to(coord_q.dtype).to(coord_q.device)
        q_grids.append(q_grid_tmp)
        k_grid_tmp = torch.meshgrid(torch.arange(k_crop_h), torch.arange(k_crop_w))
        k_grid_tmp = torch.stack(k_grid_tmp[::-1]).to(coord_k.dtype).to(coord_k.device)
        k_grids.append(k_grid_tmp)
    return q_grids, k_grids


def _debug_print(*args, flag="None"):
    if rank == 0:
        if flag != "None":
            print(*args)


def debug_print(q_start_x="None", q_start_y="None", k_start_x="None", k_start_y="None",
                q_bin_width="None", q_bin_height="None", k_bin_width="None",
                k_bin_height="None", q_grids="None", k_grids="None", q_bin_diag="None",
                k_bin_diag="None", max_bin_diag="None", center_q_x="None",
                center_q_y="None", center_k_x="None", center_k_y="None",
                center_q_x_o="None", center_q_y_o="None", center_k_x_o="None",
                center_k_y_o="None", q_flip_x="None", q_flip_y="None", k_flip_x="None",
                k_flip_y="None"):
    _debug_print(f"q_flip_x: {q_flip_x.shape}", q_flip_x.tolist(), flag=q_flip_x)
    _debug_print(f"q_flip_y: {q_flip_y.shape}", q_flip_y.tolist(), flag=q_flip_y)
    _debug_print(f"k_flip_x: {k_flip_x.shape}", k_flip_x.tolist(), flag=k_flip_x)
    _debug_print(f"k_flip_y: {k_flip_y.shape}", k_flip_y.tolist(), flag=k_flip_y)
    _debug_print(f"q_start_x: {q_start_x.shape}", q_start_x.tolist(), flag=q_start_x)
    _debug_print(f"q_start_y: {q_start_y.shape}", q_start_y.tolist(), flag=q_start_y)
    _debug_print(f"k_start_x: {k_start_x.shape}", k_start_x.tolist(), flag=k_start_x)
    _debug_print(f"k_start_y: {k_start_y.shape}", k_start_y.tolist(), flag=k_start_y)
    _debug_print(f"q_bin_width: {q_bin_width.shape}", q_bin_width.tolist(), flag=q_bin_width)
    _debug_print(f"q_bin_height: {q_bin_height.shape}", q_bin_height.tolist(), flag=q_bin_height)
    _debug_print(f"k_bin_width: {k_bin_width.shape}", k_bin_width.tolist(), flag=k_bin_width)
    _debug_print(f"k_bin_height: {k_bin_height.shape}", k_bin_height.tolist(), flag=k_bin_height)
    if q_grids != "None" and k_grids != "None":
        for ii, (q_grid, k_grid) in enumerate(zip(q_grids, k_grids)):
            _debug_print(f"{ii} q_grid: {q_grid.shape}", flag=q_grid)
            _debug_print(f"{ii} k_grid: {k_grid.shape}", flag=k_grid)
            # _debug_print(f"{ii} q_grid: {q_grid.shape}", q_grid.tolist(), flag=q_grid)
            # _debug_print(f"{ii} k_grid: {k_grid.shape}", k_grid.tolist(), flag=k_grid)
    _debug_print(f"q_bin_diag: {q_bin_diag.shape}", q_bin_diag.tolist(), flag=q_bin_diag)
    _debug_print(f"k_bin_diag: {k_bin_diag.shape}", k_bin_diag.tolist(), flag=k_bin_diag)
    _debug_print(f"max_bin_diag: {max_bin_diag.shape}", max_bin_diag.tolist(), flag=max_bin_diag)
    _debug_print(f"center_q_x: {center_q_x.shape}", center_q_x.tolist(), flag=center_q_x)
    _debug_print(f"center_q_y: {center_q_y.shape}", center_q_y.tolist(), flag=center_q_y)
    _debug_print(f"center_k_x: {center_k_x.shape}", center_k_x.tolist(), flag=center_k_x)
    _debug_print(f"center_k_y: {center_k_y.shape}", center_k_y.tolist(), flag=center_k_y)
    _debug_print(f"center_q_x_o: {center_q_x_o.shape}", center_q_x_o.tolist(), flag=center_q_x_o)
    _debug_print(f"center_q_y_o: {center_q_y_o.shape}", center_q_y_o.tolist(), flag=center_q_y_o)
    _debug_print(f"center_k_x_o: {center_k_x_o.shape}", center_k_x_o.tolist(), flag=center_k_x_o)
    _debug_print(f"center_k_y_o: {center_k_y_o.shape}", center_k_y_o.tolist(), flag=center_k_y_o)


def calc_grid_no_center(x_array, y_array, q_bin_width, q_bin_height, k_bin_width,
                        k_bin_height, q_start_x, q_start_y, k_start_x, k_start_y,
                        W_orig, H_orig):
    q_x = x_array * q_bin_width + q_start_x
    q_y = y_array * q_bin_height + q_start_y
    k_x = x_array * k_bin_width + k_start_x
    k_y = y_array * k_bin_height + k_start_y
    q_x = q_x * (W_orig - 1)
    q_y = q_y * (H_orig - 1)
    k_x = k_x * (W_orig - 1)
    k_y = k_y * (H_orig - 1)
    return q_x, q_y, k_x, k_y


def load_img_from_pil(img):
    img = np.array(img).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1)
    return img


# concat width
def get_concat_h(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst


# concat height
def get_concat_v(im1, im2):
    dst = Image.new('RGB', (im1.width, im1.height + im2.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (0, im1.height))
    return dst


def draw_rect_simple(img_src, rects, colors):
    if isinstance(img_src, torch.Tensor):
        img = img_src.clone()
        img = transforms.ToPILImage(mode="RGB")(img)
    else:
        img = img_src
    # rectcolor = (255, 0, 0)  # red
    linewidth = 4  # 線の太さ

    if len(rects) > 0:
        draw = ImageDraw.Draw(img)  # 準備

    for i, (rect, rectcolor) in enumerate(zip(rects, colors)):
        rect_tup = [(rect[0], rect[1]), (rect[2], rect[3])]
        draw.rectangle(rect_tup, outline=rectcolor, width=linewidth)

    return img


def draw_rects(img1, img2, coord_q, coord_k, colors, idx=0, out_root="./"):
    rank = torch.distributed.get_rank()
    view1 = coord_q.clone()
    view1 = view1[:, 4:8].to(torch.int)
    view1[:, 2] = view1[:, 2] + view1[:, 0]
    view1[:, 3] = view1[:, 3] + view1[:, 1]
    view1 = view1.tolist()
    view2 = coord_k.clone()
    view2 = view2[:, 4:8].to(torch.int)
    view2[:, 2] = view2[:, 2] + view2[:, 0]
    view2[:, 3] = view2[:, 3] + view2[:, 1]
    view2 = view2.tolist()
    orig_im1 = img1.clone().to(torch.uint8)
    orig_im2 = img2.clone().to(torch.uint8)
    out_img1_img2, out_img1, out_img2 = [], [], []
    crop_imgs = []
    for b_idx, (im1, im2, v1, v2) in enumerate(zip(orig_im1, orig_im2, view1, view2)):
        pil_im1 = draw_rect_simple(im1, [], [])
        pil_im2 = draw_rect_simple(im2, [], [])
        img = draw_rect_simple(im2, [v1, v2], colors)
        img1 = draw_rect_simple(im1, [v1], [colors[0]])
        img2 = draw_rect_simple(im2, [v2], [colors[1]])
        img.save(os.path.join(out_root, f"crop_rec_img_{idx}_{b_idx}_{rank}.png"))
        img1.save(os.path.join(out_root, f"crop_rec_img_1_{idx}_{b_idx}_{rank}.png"))
        img2.save(os.path.join(out_root, f"crop_rec_img_2_{idx}_{b_idx}_{rank}.png"))
        out_img1_img2.append(load_img_from_pil(img))
        out_img1.append(load_img_from_pil(img1))
        out_img2.append(load_img_from_pil(img2))
        im1_crop = pil_im1.crop(v1)
        im2_crop = pil_im2.crop(v2)
        im1_crop.save(os.path.join(out_root, f"crop_img_1_{idx}_{b_idx}_{rank}.png"))
        im2_crop.save(os.path.join(out_root, f"crop_img_2_{idx}_{b_idx}_{rank}.png"))
        crop_imgs.append([load_img_from_pil(im1_crop), load_img_from_pil(im2_crop)])

    out_img1_img2 = torch.stack(out_img1_img2)
    out_img1 = torch.stack(out_img1)
    out_img2 = torch.stack(out_img2)

    # return [out_img1_img2, out_img1, out_img2], crop_imgs
    return [out_img1_img2, out_img1_img2.clone(), out_img1_img2.clone()], crop_imgs


def create_colors(color_src, num=None):
    color_list = []
    s_color = list(color_src)
    max_idx, min_idx = 0, 1
    max_val = 255 - 50
    min_c, max_c = s_color[min_idx], s_color[max_idx]
    for i, c in enumerate(s_color):
        if min_c > c:
            min_idx = i
            min_c = c
        if max_c < c:
            max_idx = i
            max_c = c

    if min_idx == max_idx:
        min_idx = (max_idx + 1) % 3

    if min_idx == 0:
        if max_idx == 1:
            mid_idx = 2
        elif max_idx == 2:
            mid_idx = 1
    elif min_idx == 1:
        if max_idx == 0:
            mid_idx = 2
        elif max_idx == 2:
            mid_idx = 0
    elif min_idx == 2:
        if max_idx == 0:
            mid_idx = 1
        elif max_idx == 1:
            mid_idx = 0

    cnt_idx, i = 0, 0
    while True:
        while True:
            while True:
                # print(s_color, cnt_idx)
                if s_color[mid_idx] > max_val:
                    color_list.append(tuple(s_color))
                    cnt_idx += 1
                    break
                color_list.append(tuple(s_color))
                if cnt_idx == 0:
                    s_color[mid_idx] = 100
                s_color[mid_idx] += 50
                s_color[mid_idx] %= 256
                cnt_idx += 1
                if num is not None and cnt_idx >= num:
                    break
            if s_color[min_idx] > max_val:
                break
            # s_color[mid_idx] = 0
            s_color[mid_idx] = 100
            s_color[min_idx] += 50
            s_color[min_idx] %= 256
            if num is not None and cnt_idx >= num:
                break
        s_color[min_idx] = 0
        if i == 0:
            s_color[max_idx] = 0
        elif s_color[max_idx] > max_val:
            break
        # s_color[mid_idx] = 0
        s_color[mid_idx] = 100
        s_color[max_idx] += 50
        s_color[max_idx] %= 256
        i += 1
        if num is not None and cnt_idx >= num:
            break

    return color_list


def draw_point_simple(img_src, points_src, color_list, width=4, bin_width=None, bin_height=None):
    if isinstance(img_src, torch.Tensor):
        img = img_src.clone()
        img = transforms.ToPILImage(mode="RGB")(img)
    else:
        img = img_src

    if isinstance(points_src, torch.Tensor):
        points = points_src.clone()
        ndim = points.ndim
        if ndim >= 3:
            points = list(itertools.chain.from_iterable(points.tolist()))
            if width <= 0:
                points = list(itertools.chain.from_iterable(points))
        elif ndim == 1:
            points = points.unsqueeze(0).tolist()
    else:
        points = points_src

    # rectcolor = (255, 0, 0)  # red
    # width = 4  # 線の太さ

    if len(points) > 0:
        draw = ImageDraw.Draw(img)  # 準備

    if width <= 0:
        draw.point(points, fill=color_list)
        return img

    for i, point in enumerate(points):
        if isinstance(color_list, tuple):
            l_color = color_list
        else:
            l_color = tuple(color_list[i])
        point_tmp = [(point[0] - (width / 2), point[1] - (width / 2)), (point[0] + (width / 2), point[1] + (width / 2))]
        draw.ellipse(point_tmp, fill=l_color, width=width)
        if bin_width is not None and bin_height is not None:
            rect_1 = (int(point[0] - (bin_width.item() / 2)), int(point[1] - (bin_height.item() / 2)))
            rect_2 = (int(point[0] + (bin_width.item() / 2)), int(point[1] + (bin_height.item() / 2)))
            rect_tup = [rect_1, rect_2]
            draw.rectangle(rect_tup, outline=l_color, width=width)

    return img


def make_grid(x, y):
    ndim = x.ndim
    assert ndim == 3 or ndim == 2
    if ndim == 3:
        grids = torch.stack([x, y]).permute(1, 0, 2, 3)
    elif ndim == 2:
        grids = torch.stack([x, y]).unsqueeze(0)

    return grids


def adjust_img_dim(img_src):
    img_ndim = img_src.ndim
    assert img_ndim == 4 or img_ndim == 3
    img = img_src.clone()
    if img_ndim == 3:
        img = img.unsqueeze(0)
    return img


def draw_points_onegrid(x, y, img_src, out_path, color, name="plot_point", width=4):
    # print(color, "color in one grid")
    if isinstance(color, list):
        color = color[0]
    # print(color, "color in one grid")

    grids = make_grid(x, y)
    test_imgs = adjust_img_dim(img_src)

    for idx, (orig_im, grid) in enumerate(zip(test_imgs, grids)):
        img = draw_point_simple(orig_im, grid.permute(1, 2, 0), color, width)
        img.save(os.path.join(out_path, f"{name}_{idx}.png"))


def draw_points(q_x, q_y, k_x, k_y, img_src, out_path, color, name="plot_point", width=4):
    q_grids = make_grid(q_x, q_y)
    k_grids = make_grid(k_x, k_y)
    test_imgs = adjust_img_dim(img_src)

    for idx, (orig_im, q_grid, k_grid) in enumerate(zip(test_imgs, q_grids, k_grids)):
        img = draw_point_simple(orig_im, q_grid.permute(1, 2, 0), color[0], width)
        img = draw_point_simple(img, k_grid.permute(1, 2, 0), color[1], width)
        img.save(os.path.join(out_path, f"{name}_{idx}.png"))


def draw_point_positive_pair(q_x, q_y, k_x, k_y, img1_src, img2_src, out_path, color_s,
                             pos_masks, name="plot_point_positive", width=4,
                             q_bin_width=None, k_bin_width=None, q_bin_height=None,
                             k_bin_height=None):
    q_grids = make_grid(q_x, q_y)
    k_grids = make_grid(k_x, k_y)
    im1 = adjust_img_dim(img1_src)
    im2 = adjust_img_dim(img2_src)
    nb, c, h, w = q_grids.shape
    q_grids = q_grids.permute(0, 2, 3, 1)
    k_grids = k_grids.permute(0, 2, 3, 1)
    q_grids = q_grids.view(nb, h * w, 1, c).repeat(1, 1, h * w, 1)
    k_grids = k_grids.view(nb, 1, h * w, c).repeat(1, h * w, 1, 1)
    color_list = create_colors(color_s[0])

    for idx, (orig_im1, orig_im2, q_grid, k_grid, pos_mask) in enumerate(zip(im1, im2, q_grids, k_grids, pos_masks)):
        l_q_grid = q_grid[pos_mask]
        l_k_grid = k_grid[pos_mask]
        img1 = draw_point_simple(orig_im1, l_q_grid, color_list, width, True, q_bin_width[idx], q_bin_height[idx])
        img2 = draw_point_simple(orig_im2, l_k_grid, color_list, width, True, k_bin_width[idx], k_bin_height[idx])
        # img1 = draw_point_simple(orig_im1, l_q_grid[0], color_s[0], width, True, q_bin_width[idx], q_bin_height[idx])
        # img2 = draw_point_simple(orig_im2, l_k_grid[0], color_s[0], width, True, k_bin_width[idx], k_bin_height[idx])
        img = get_concat_h(img1, img2)
        img.save(os.path.join(out_path, f"{name}_{idx}.png"))


def draw_warp_img(grid_x, grid_y, W_orig, H_orig, img_src, name, out_path, mask_src=None):
    # rank = torch.distributed.get_rank()
    grid = torch.stack([grid_x, grid_y]).unsqueeze(0)
    grid[:, 0] = grid[:, 0] / (W_orig - 1)
    grid[:, 1] = grid[:, 1] / (H_orig - 1)
    grid[:, 0] = 2 * grid[:, 0] - 1
    grid[:, 1] = 2 * grid[:, 1] - 1
    # warp img2 -> img1
    orig_img_copy = img_src.clone()
    orig_img_copy = orig_img_copy.to(grid.device).to(grid.dtype)
    orig_img_copy = orig_img_copy.unsqueeze(0)
    of_img = F.grid_sample(orig_img_copy, grid.permute(0, 2, 3, 1),
                           align_corners=True)
    of_img = of_img.detach().cpu()
    orig_img_copy = orig_img_copy.detach().cpu()
    orig_img_copy = orig_img_copy.to(torch.uint8)
    of_img = of_img.to(torch.uint8)
    # pil_of_img = transforms.ToPILImage(mode="RGB")(of_img.squeeze(0))
    # pil_of_img = draw_rect_simple(of_img.squeeze(0), [], [])
    # pil_of_img.save(os.path.join(out_path, f"{name}.png"))
    # save_image(orig_img_copy, os.path.join(out_path, f"{name}0.png"))
    pil_orig_img = transforms.ToPILImage(mode="RGB")(orig_img_copy.squeeze(0))
    pil_orig_img.save(os.path.join(out_path, f"{name}_orig.png"))
    pil_of_img = transforms.ToPILImage(mode="RGB")(of_img.squeeze(0))
    # save_image(of_img, os.path.join(out_path, f"{name}.png"))
    pil_of_img.save(os.path.join(out_path, f"{name}.png"))

    if mask_src is not None:
        mask = mask_src.clone()
        mask = torch.logical_not(mask.unsqueeze(0))
        # orig_img_copy = orig_img_copy.permute(0, 2, 3, 1)
        # orig_img_copy[mask] = 0.0
        # orig_img_copy = orig_img_copy.permute(0, 3, 1, 2)
        # pil_orig_img = transforms.ToPILImage(mode="RGB")(orig_img_copy.squeeze(0))
        # pil_orig_img.save(os.path.join(out_path, f"{name}_orig_mask.png"))
        of_img = of_img.permute(0, 2, 3, 1)
        of_img[mask] = 0.0
        of_img = of_img.permute(0, 3, 1, 2)
        pil_of_img = transforms.ToPILImage(mode="RGB")(of_img.squeeze(0))
        pil_of_img.save(os.path.join(out_path, f"{name}_mask.png"))
    # save_image((of_img * 255), os.path.join(out_path, f"{name}2.png"))
    # if rank == 0:
    #     print(f"{i} orig_img_copy: {orig_img_copy.shape}", orig_img_copy.tolist())
    #     print(f"{i} of_q_grid: {of_q_grid.shape}", of_q_grid.tolist())
    #     print(f"{i} of_img: {of_img.shape}", of_img.tolist())


def draw_points_all(q_grids, k_grids, q_bin_width, q_bin_height, k_bin_width,
                    k_bin_height, q_start_x, q_start_y, k_start_x, k_start_y,
                    W, H, W_orig, H_orig, test_imgs, img1, img2, out_path, color,
                    name, is_center=False, flow_fwd=None, out_path_flo="",
                    add_optical_flow=None):
    is_calc_flow = flow_fwd is not None and add_optical_flow is not None
    size = (H_orig, W_orig)
    # rank = torch.distributed.get_rank()
    for i, (q_grid, k_grid) in enumerate(zip(q_grids, k_grids)):
        q_crop_h, q_crop_w = q_grid.shape[-2:]
        k_crop_h, k_crop_w = k_grid.shape[-2:]
        l_q_bin_width = q_bin_width[i] * W / (q_crop_w)
        l_q_bin_height = q_bin_height[i] * H / (q_crop_h)
        l_k_bin_width = k_bin_width[i] * W / (k_crop_w)
        l_k_bin_height = k_bin_height[i] * H / (k_crop_h)
        # l_q_bin_width = q_bin_width[i] * W / (q_crop_w - 1)
        # l_q_bin_height = q_bin_height[i] * H / (q_crop_h - 1)
        # l_k_bin_width = k_bin_width[i] * W / (k_crop_w - 1)
        # l_k_bin_height = k_bin_height[i] * H / (k_crop_h - 1)
        if is_center:
            center_q_x_tmp = (q_grid[0] + 0.5) * l_q_bin_width + q_start_x[i]
            center_q_y_tmp = (q_grid[1] + 0.5) * l_q_bin_height + q_start_y[i]
            # center_q_x_tmp = (q_grid[0] + 0.5) * (q_bin_width[i].item() * W * (W_orig - 1) / (q_crop_w)) + (q_start_x[i] * (W_orig - 1))
            # center_q_y_tmp = (q_grid[1] + 0.5) * (q_bin_height[i].item() * H * (H_orig - 1) / (q_crop_h)) + (q_start_y[i] * (H_orig - 1))
            # center_q_x_tmp = (q_grid[0] + 0.5) * (q_bin_width[i].item() * W * (W_orig - 1) / (q_crop_w - 1)) + (q_start_x[i] * (W_orig - 1))
            # center_q_y_tmp = (q_grid[1] + 0.5) * (q_bin_height[i].item() * H * (H_orig - 1) / (q_crop_h - 1)) + (q_start_y[i] * (H_orig - 1))
            center_k_x_tmp = (k_grid[0] + 0.5) * l_k_bin_width + k_start_x[i]
            center_k_y_tmp = (k_grid[1] + 0.5) * l_k_bin_height + k_start_y[i]
            # center_k_x_tmp = (k_grid[0] + 0.5) * (k_bin_width[i].item() * W * (W_orig - 1) / (k_crop_w)) + (k_start_x[i] * (W_orig - 1))
            # center_k_y_tmp = (k_grid[1] + 0.5) * (k_bin_height[i].item() * H * (H_orig - 1) / (k_crop_h)) + (k_start_y[i] * (H_orig - 1))
            # center_k_x_tmp = (k_grid[0] + 0.5) * (k_bin_width[i].item() * W * (W_orig - 1) / (k_crop_w - 1)) + (k_start_x[i] * (W_orig - 1))
            # center_k_y_tmp = (q_grid[1] + 0.5) * (q_bin_height[i].item() * H * (H_orig - 1) / (k_crop_h - 1)) + (k_start_y[i] * (H_orig - 1))
        else:
            center_q_x_tmp = q_grid[0] * l_q_bin_width + q_start_x[i]
            center_q_y_tmp = q_grid[1] * l_q_bin_height + q_start_y[i]
            # center_q_x_tmp = q_grid[0] * (q_bin_width[i].item() * W * (W_orig - 1) / (q_crop_w)) + (q_start_x[i] * (W_orig - 1))
            # center_q_y_tmp = q_grid[1] * (q_bin_height[i].item() * H * (H_orig - 1) / (q_crop_h)) + (q_start_y[i] * (H_orig - 1))
            # center_q_x_tmp = q_grid[0] * (q_bin_width[i].item() * W * (W_orig - 1) / (q_crop_w - 1)) + (q_start_x[i] * (W_orig - 1))
            # center_q_y_tmp = q_grid[1] * (q_bin_height[i].item() * H * (H_orig - 1) / (q_crop_h - 1)) + (q_start_y[i] * (H_orig - 1))
            center_k_x_tmp = k_grid[0] * l_k_bin_width + k_start_x[i]
            center_k_y_tmp = k_grid[1] * l_k_bin_height + k_start_y[i]
            # center_k_x_tmp = k_grid[0] * (k_bin_width[i].item() * W * (W_orig - 1) / (k_crop_w)) + (k_start_x[i] * (W_orig - 1))
            # center_k_y_tmp = k_grid[1] * (k_bin_height[i].item() * H * (H_orig - 1) / (k_crop_h)) + (k_start_y[i] * (H_orig - 1))
            # center_k_x_tmp = k_grid[0] * (k_bin_width[i].item() * W * (W_orig - 1) / (k_crop_w - 1)) + (k_start_x[i] * (W_orig - 1))
            # center_k_y_tmp = k_grid[1] * (k_bin_height[i].item() * H * (H_orig - 1) / (k_crop_h - 1)) + (k_start_y[i] * (H_orig - 1))

        center_q_x_tmp = center_q_x_tmp * (W_orig - 1)
        center_q_y_tmp = center_q_y_tmp * (H_orig - 1)
        center_k_x_tmp = center_k_x_tmp * (W_orig - 1)
        center_k_y_tmp = center_k_y_tmp * (H_orig - 1)

        draw_points(center_q_x_tmp, center_q_y_tmp, center_k_x_tmp, center_k_y_tmp, test_imgs[i], out_path, color, f"{name}_{i}", 0)
        draw_points_onegrid(center_k_x_tmp, center_k_y_tmp, test_imgs[i], out_path, color[1], f"{name}_2grid_on_1frame_{i}", 0)
        draw_warp_img(center_q_x_tmp, center_q_y_tmp, W_orig, H_orig, test_imgs[i],
                      f"{name}_2frame_aug_img_q_{i}", out_path_flo)
        draw_warp_img(center_k_x_tmp, center_k_y_tmp, W_orig, H_orig, test_imgs[i],
                      f"{name}_2frame_aug_img_k_{i}", out_path_flo)
        if img1 is not None:
            # draw_points(center_q_x_tmp, center_q_y_tmp, center_k_x_tmp, center_k_y_tmp, img1[i], out_path, color, f"{name}_1frame_{i}", 0)
            draw_points_onegrid(center_q_x_tmp, center_q_y_tmp, img1[i], out_path, color[0], f"{name}_1frame_{i}", 0)
            draw_warp_img(center_q_x_tmp, center_q_y_tmp, W_orig, H_orig, img1[i],
                          f"{name}_1frame_aug_img_{i}", out_path_flo)
        if img2 is not None:
            # draw_points(center_q_x_tmp, center_q_y_tmp, center_k_x_tmp, center_k_y_tmp, img2[i], out_path, color, f"{name}_2frame_{i}", 0)
            draw_points_onegrid(center_k_x_tmp, center_k_y_tmp, img2[i], out_path, color[1], f"{name}_2frame_{i}", 0)
            draw_warp_img(center_k_x_tmp, center_k_y_tmp, W_orig, H_orig, img2[i],
                          f"{name}_2frame_aug_img_{i}", out_path_flo)
        # if rank == 0:
        #     print(f"{i} center_q_x_tmp: {center_q_x_tmp.shape}", center_q_x_tmp.tolist())
        #     print(f"{i} center_q_y_tmp: {center_q_y_tmp.shape}", center_q_x_tmp.tolist())
        #     print(f"{i} center_k_x_tmp: {center_k_x_tmp.shape}", center_k_x_tmp.tolist())
        #     print(f"{i} center_k_y_tmp: {center_k_y_tmp.shape}", center_k_x_tmp.tolist())
        #     print(f"{i} q_grid: {q_grid.shape}", q_grid.tolist())
        #     print(f"{i} k_grid: {k_grid.shape}", k_grid.tolist())

        if is_calc_flow:
            l_flow_fwd = flow_fwd[i].unsqueeze(0)
            l_center_q_x_tmp = center_q_x_tmp.unsqueeze(0)
            l_center_q_y_tmp = center_q_y_tmp.unsqueeze(0)
            q_x_tmp, q_y_tmp = add_optical_flow(l_flow_fwd, l_center_q_x_tmp,
                                                l_center_q_y_tmp, size)
            q_x_tmp = q_x_tmp.squeeze(0)
            q_y_tmp = q_y_tmp.squeeze(0)
            draw_points(q_x_tmp, q_y_tmp, center_k_x_tmp, center_k_y_tmp, test_imgs[i], out_path_flo, color, f"{name}_{i}", 0)
            draw_points_onegrid(center_k_x_tmp, center_k_y_tmp, test_imgs[i], out_path_flo, color[1], f"{name}_2frame_on_1frame_{i}", 0)
            if img1 is not None:
                # draw_points(q_x_tmp, q_y_tmp, center_k_x_tmp, center_k_y_tmp, img1[i], out_path_flo, color, f"{name}_1frame_{i}", 0)
                draw_points_onegrid(q_x_tmp, q_y_tmp, img1[i], out_path_flo, color[0], f"{name}_1frame_{i}", 0)
            if img2 is not None:
                # draw_points(q_x_tmp, q_y_tmp, center_k_x_tmp, center_k_y_tmp, img2[i],
                #             out_path_flo, color, f"{name}_2frame_{i}", 0)
                draw_points_onegrid(center_k_x_tmp, center_k_y_tmp, img2[i], out_path_flo, color[1], f"{name}_2frame_{i}", 0)
            # if rank == 0:
            #     print(f"{i} q_x_tmp: {q_x_tmp.shape}", q_x_tmp.tolist())
            #     print(f"{i} q_y_tmp: {q_y_tmp.shape}", q_x_tmp.tolist())
            #     print(f"{i} center_k_x_tmp: {center_k_x_tmp.shape}", center_k_x_tmp.tolist())
            #     print(f"{i} center_k_y_tmp: {center_k_y_tmp.shape}", center_k_x_tmp.tolist())
            mask = None
            draw_warp_img(q_x_tmp, q_y_tmp, W_orig, H_orig, test_imgs[i],
                          f"{name}_2frame_to_1frame_img_{i}_flo", out_path_flo, mask)


def debug_calc_grid(x_array, y_array, q_start_x, q_start_y, k_start_x, k_start_y,
                    q_bin_width, q_bin_height, k_bin_width, k_bin_height, q_grids,
                    k_grids, q_x, q_y, k_x, k_y, test_imgs, img1, img2,
                    out_path, out_path_center, color, W_orig, H_orig,
                    center_q_x=None, center_q_y=None, center_k_x=None, center_k_y=None,
                    flow_fwd=None, out_path_flo="", out_path_center_flo="",
                    add_optical_flow=None):
    H, W = x_array.shape[-2:]
    is_plot_flow = center_q_x is not None
    is_calc_flow = add_optical_flow is not None

    # debug
    draw_points(q_x, q_y, k_x, k_y, test_imgs, out_path_center, color)
    if img1 is not None:
        draw_points(q_x, q_y, k_x, k_y, img1, out_path_center, color, "plot_point_1frame")
    if img2 is not None:
        draw_points(q_x, q_y, k_x, k_y, img2, out_path_center, color, "plot_point_2frame")

    if is_plot_flow:
        draw_points(center_q_x, center_q_y, center_k_x, center_k_y, test_imgs, out_path_center_flo, color)
        # draw_points(center_q_x[mask], center_q_y[mask], center_k_x, center_k_y, test_imgs, out_path_center_flo, color)
        if img1 is not None:
            draw_points(center_q_x, center_q_y, center_k_x, center_k_y, img1, out_path_center_flo, color, "plot_point_1frame")
        if img2 is not None:
            draw_points(center_q_x, center_q_y, center_k_x, center_k_y, img2, out_path_center_flo, color, "plot_point_2frame")

    # check optical flow
    # draw_points_all(q_grids, k_grids, q_bin_width, q_bin_height, k_bin_width,
    #                 k_bin_height, q_start_x, q_start_y, k_start_x, k_start_y,
    #                 W, H, W_orig, H_orig, test_imgs, img1, img2, out_path_center,
    #                 color, "of_plot_point", True)
    draw_points_all(q_grids, k_grids, q_bin_width, q_bin_height, k_bin_width,
                    k_bin_height, q_start_x, q_start_y, k_start_x, k_start_y,
                    W, H, W_orig, H_orig, test_imgs, img1, img2, out_path_center,
                    color, "of_plot_point", True, flow_fwd, out_path_center_flo,
                    add_optical_flow)

    # debug no center
    out_grids = calc_grid_no_center(x_array, y_array, q_bin_width, q_bin_height,
                                    k_bin_width, k_bin_height, q_start_x, q_start_y,
                                    k_start_x, k_start_y, W_orig, H_orig)
    q_x_n, q_y_n, k_x_n, k_y_n = out_grids
    draw_points(q_x_n, q_y_n, k_x_n, k_y_n, test_imgs, out_path, color)
    if img1 is not None:
        draw_points(q_x_n, q_y_n, k_x_n, k_y_n, img1, out_path, color, "plot_point_1frame")
    if img2 is not None:
        draw_points(q_x_n, q_y_n, k_x_n, k_y_n, img2, out_path, color, "plot_point_2frame")

    if is_calc_flow:
        size = (H_orig, W_orig)
        center_q_x_n, center_q_y_n = add_optical_flow(flow_fwd, q_x_n, q_y_n, size)
        center_k_x_n, center_k_y_n = k_x_n.clone(), k_y_n.clone()
        draw_points(center_q_x_n, center_q_y_n, center_k_x_n, center_k_y_n, test_imgs, out_path_flo, color)
        # draw_points(center_q_x[mask], center_q_y[mask], center_k_x, center_k_y, test_imgs, out_path_flo, color)
        if img1 is not None:
            draw_points(center_q_x_n, center_q_y_n, center_k_x_n, center_k_y_n, img1, out_path_flo, color, "plot_point_1frame")
            # draw_points(center_q_x[mask], center_q_y[mask], center_k_x, center_k_y, img1, out_path_flo, color, "plot_point_1frame")
        if img2 is not None:
            draw_points(center_q_x_n, center_q_y_n, center_k_x_n, center_k_y_n, img2, out_path_flo, color, "plot_point_2frame")
            # draw_points(center_q_x[mask], center_q_y[mask], center_k_x, center_k_y, img2, out_path_flo, color, "plot_point_2frame")

    # check optical flow
    # draw_points_all(q_grids, k_grids, q_bin_width, q_bin_height, k_bin_width,
    #                 k_bin_height, q_start_x, q_start_y, k_start_x, k_start_y,
    #                 W, H, W_orig, H_orig, test_imgs, img1, img2, out_path,
    #                 color, "of_plot_point", False)
    draw_points_all(q_grids, k_grids, q_bin_width, q_bin_height, k_bin_width,
                    k_bin_height, q_start_x, q_start_y, k_start_x, k_start_y,
                    W, H, W_orig, H_orig, test_imgs, img1, img2, out_path,
                    color, "of_plot_point", False, flow_fwd, out_path_flo,
                    add_optical_flow)
