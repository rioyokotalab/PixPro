from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

import torch
from torchvision.transforms import functional as F
import torch.nn.functional as nnF


def debug_img_save(img_tensor, orig_img_pil, out_coord, out_grid, idx=1,
                   calc_coord=None, is_corner=True):
    frame_name = f"{idx} frame"
    orig_img = F.to_tensor(orig_img_pil).clone()
    grid = out_grid.clone()
    coord = out_coord.clone()
    img = img_tensor.clone()
    is_use_calc_coord = calc_coord is not None
    if is_use_calc_coord:
        calc_grid = calc_coord.clone()
    out_img_list = []

    h, w = img.shape[-2:]
    # h, w = h // 8, w // 8
    to_pad_resize = Resize_with_pad(w, h)

    img = to_pad_resize(img)
    img_str = add_str_img(f"official data aug img on {frame_name}", img)

    orig_img = orig_img.unsqueeze(0)

    # grid_h, grid_w = grid.shape[-2:]
    # resize_grid = F.resize(grid, (grid_h // 8, grid_w // 8))
    # resize_grid = resize_grid.unsqueeze(0).permute(0, 2, 3, 1)
    # img_tmp_resize = nnF.grid_sample(orig_img, resize_grid, align_corners=is_corner)
    # img_tmp_resize = to_pad_resize(img_tmp_resize[0])
    # str_down = f"down size my resized data aug img on {idx} frame by pad"
    # img_tmp_resize = add_str_img(str_down, img_tmp_resize)

    grid_tmp = grid.unsqueeze(0).permute(0, 2, 3, 1)
    coord_tmp = coord.unsqueeze(0).permute(0, 2, 3, 1)
    img_tmp = nnF.grid_sample(orig_img, grid_tmp, align_corners=is_corner)
    if is_use_calc_coord:
        calc_grid_tmp = calc_grid.unsqueeze(0).permute(0, 2, 3, 1)
        img_calc_tmp = nnF.grid_sample(orig_img, calc_grid_tmp, align_corners=is_corner)
        img_calc_tmp = to_pad_resize(img_calc_tmp)
        add_str = f"data aug by using official grid img on {frame_name}"
        img_calc_tmp = add_str_img(add_str, img_calc_tmp[0])
        inv_img_calc_tmp = nnF.grid_sample(img.unsqueeze(0), calc_grid_tmp,
                                           align_corners=is_corner)
        inv_img_calc_tmp = to_pad_resize(inv_img_calc_tmp)
        add_str = "inv data aug by using official grid img from officical data aug on"
        add_str += f" {frame_name}"
        inv_img_calc_tmp = add_str_img(add_str, inv_img_calc_tmp[0])

    down_size_orig_img = to_pad_resize(orig_img[0])
    # down_size_orig_img = to_pad_resize(orig_img)
    add_str = f"down size orginal img on {frame_name}"
    down_size_orig_img = add_str_img(add_str, down_size_orig_img)

    down_size_img_tmp = to_pad_resize(img_tmp[0])
    # down_size_img_tmp = to_pad_resize(img_tmp)
    # down_size_img_tmp = down_size_img_tmp[0]
    add_str = f"down size my data aug img on {frame_name} by pad"
    down_size_img_tmp = add_str_img(add_str, down_size_img_tmp)

    img_tmp_crop = F.resize(img_tmp[0], [h, w])
    # img_tmp_crop = nnF.interpolate(img_tmp, (h, w),
    #                                mode='bilinear',
    #                                align_corners=is_corner)
    # img_tmp_crop = img_tmp_crop[0]
    add_str = f"down size my data aug img on {frame_name} by resize"
    img_tmp_crop = add_str_img(add_str, img_tmp_crop)

    inv_img = nnF.grid_sample(img.unsqueeze(0), coord_tmp, align_corners=is_corner)
    add_str = f"inv my data aug img from offical data aug img on {frame_name}"
    inv_img = add_str_img(add_str, inv_img[0])
    inv_img_tmp = nnF.grid_sample(img_tmp, coord_tmp, align_corners=is_corner)
    add_str = "inv my data aug img from my data aug img which no resize on "
    add_str += f"{frame_name}"
    inv_img_tmp = add_str_img(add_str, inv_img_tmp[0])
    inv_orig_img_tmp = nnF.grid_sample(orig_img, coord_tmp, align_corners=is_corner)
    add_str = "inv orig img my data aug img from my data aug img which no resize on "
    add_str += f"{frame_name}"
    inv_orig_img_tmp = add_str_img(add_str, inv_orig_img_tmp[0])
    # inv_img_tmp_crop = nnF.grid_sample(img_tmp_crop.unsqueeze(0), coord_tmp,
    #                                    align_corners=is_corner)
    # add_str = "inv my data aug img from my data aug img which has resize "
    # add_str += f"on {frame_name}"
    # inv_img_tmp_crop = add_str_img(add_str, inv_img_tmp_crop[0])
    # out_img_list.append(orig_img[0])
    out_img_list.append(down_size_orig_img)
    out_img_list.append(img_str)
    out_img_list.append(down_size_img_tmp)
    out_img_list.append(img_tmp_crop)
    # out_img_list.append(img_tmp_resize)
    # out_img_list.append(img_tmp[0])
    if is_use_calc_coord:
        out_img_list.append(img_calc_tmp)
        out_img_list.append(inv_img_calc_tmp)
    # out_img_list.append(down_size_orig_img)
    out_img_list.append(inv_img)
    out_img_list.append(inv_img_tmp)
    out_img_list.append(inv_orig_img_tmp)
    # out_img_list.append(inv_img_tmp_crop)
    out_img = torch.stack(out_img_list)

    return out_img


@torch.no_grad()
def add_str_img(text: str, img: torch.Tensor):
    device = img.device
    s_img = img.clone()
    dims = s_img.ndim - 1
    wd = s_img.shape[-1]

    # ttfontname = "NotoSansCJK-Regular.ttc"
    ttfontname = "DejaVuSans.ttf"
    fontsize = 20

    canvasSize = (wd, 30)
    backgroundRGB = (255, 255, 255)
    textRGB = (0, 0, 0)

    img = Image.new("RGB", canvasSize, backgroundRGB)
    draw = ImageDraw.Draw(img)

    font = ImageFont.truetype(ttfontname, fontsize)
    textWidth, textHeight = draw.textsize(text, font=font)
    textTopLeft = (0, canvasSize[1] // 2 - textHeight // 2)
    draw.text(textTopLeft, text, fill=textRGB, font=font)

    img_tensor = F.to_tensor(img)
    img_tensor = img_tensor.to(device)

    cat_img = torch.cat([img_tensor, s_img], dim=dims - 1)
    return cat_img


class Resize_with_pad:
    def __init__(self, w=1024, h=768):
        self.w = w
        self.h = h

    def __call__(self, image):
        h_1, w_1 = image.shape[-2:]
        ratio_f = self.w / self.h
        ratio_1 = w_1 / h_1

        # check if the original and final aspect ratios are the same within a margin
        if round(ratio_1, 2) != round(ratio_f, 2):
            # padding to preserve aspect ratio
            hp = int(w_1 / ratio_f - h_1)
            wp = int(ratio_f * h_1 - w_1)
            # wp = int(w_1 / ratio_f - h_1)
            # hp = int(ratio_f * h_1 - w_1)
            if hp > 0 and wp < 0:
                # hp = hp // 2
                # image = nnF.pad(image, (0, 0, 0, hp), "constant", 0)
                image = F.pad(image, (0, 0, 0, hp), 0, "constant")
            elif hp < 0 and wp > 0:
                # wp = wp // 2
                # image = nnF.pad(image, (0, 0, wp, 0), "constant", 0)
                image = F.pad(image, (0, 0, wp, 0), 0, "constant")

        # return nnF.interpolate(image, (self.h, self.w),
        #                      mode='bilinear',
        #                      align_corners=True)
        return F.resize(image, [self.h, self.w])
