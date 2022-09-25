import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed import get_world_size

from .base import BaseModel


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return input


def conv1x1(in_planes, out_planes):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=True)


class MLP2d(nn.Module):
    def __init__(self, in_dim, inner_dim=4096, out_dim=256):
        super(MLP2d, self).__init__()

        self.linear1 = conv1x1(in_dim, inner_dim)
        self.bn1 = nn.BatchNorm2d(inner_dim)
        self.relu1 = nn.ReLU(inplace=True)

        self.linear2 = conv1x1(inner_dim, out_dim)

    def forward(self, x):
        x = self.linear1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.linear2(x)

        return x


def add_optical_flow(flow, x_grid, y_grid, size, verbose=False):
    H, W = x_grid.shape[-2:]
    H_in, W_in = flow.shape[-2:]
    H_orig, W_orig = size
    is_diff_size = H_in != H_orig or W_in != W_orig
    if is_diff_size:
        ratio_h, ratio_w = H_in / H_orig, W_in / W_orig
    # get crop size before resize
    if verbose:
        rank = torch.distributed.get_rank()
        print(f"rank: {rank} orig size: ({H_orig}, {W_orig})")
        print(f"rank: {rank} cur size: ({H}, {W})")

    x = x_grid.clone()
    y = y_grid.clone()
    x = 2 * (x / (W_orig - 1)) - 1
    y = 2 * (y / (H_orig - 1)) - 1
    grid = torch.stack([x, y]).permute(1, 0, 2, 3)
    flow_grid = F.grid_sample(flow, grid.permute(0, 2, 3, 1), align_corners=True)

    out_x = x_grid.clone()
    out_y = y_grid.clone()
    if is_diff_size:
        out_x = out_x * ratio_w + flow_grid[:, 0]
        out_y = out_y * ratio_h + flow_grid[:, 1]
        out_x = out_x / ratio_w
        out_y = out_y / ratio_h
    else:
        out_x = out_x + flow_grid[:, 0]
        out_y = out_y + flow_grid[:, 1]

    # out_x = 2 * (out_x / (W_orig - 1)) - 1
    # out_y = 2 * (out_y / (H_orig - 1)) - 1
    # out_x = out_x / (W_orig - 1)
    # out_y = out_y / (H_orig - 1)
    return out_x, out_y


def regression_loss(q, k, coord_q, coord_k, pos_ratio=0.5):
    """ q, k: N * C * H * W
        coord_q, coord_k: N * 4 (x_upper_left, y_upper_left, x_lower_right, y_lower_right)
    """
    N, C, H, W = q.shape

    is_calc_flow = isinstance(coord_q, list)
    if is_calc_flow:
        coord_q, flow_fwd = coord_q
        coord_k, flow_bwd = coord_k
        if isinstance(flow_fwd, list):
            flow_fwd, size = flow_fwd
            flow_bwd, _ = flow_bwd
            if isinstance(size, torch.Tensor):
                H_orig, W_orig = int(size[0].item()), int(size[1].item())
                size = (H_orig, W_orig)
        else:
            size = flow_fwd.shape[-2:]
    else:
        size = (coord_q[0][9].item(), coord_q[0][8].item())
    H_orig, W_orig = size

    # [bs, feat_dim, 49]
    q = q.view(N, C, -1)
    k = k.view(N, C, -1)

    # generate center_coord, width, height
    # [1, 7, 7]
    x_array = torch.arange(0., float(W), dtype=coord_q.dtype, device=coord_q.device).view(1, 1, -1).repeat(1, H, 1)
    y_array = torch.arange(0., float(H), dtype=coord_q.dtype, device=coord_q.device).view(1, -1, 1).repeat(1, 1, W)
    # [bs, 1, 1]
    q_bin_width = ((coord_q[:, 2] - coord_q[:, 0]) / W).view(-1, 1, 1)
    q_bin_height = ((coord_q[:, 3] - coord_q[:, 1]) / H).view(-1, 1, 1)
    k_bin_width = ((coord_k[:, 2] - coord_k[:, 0]) / W).view(-1, 1, 1)
    k_bin_height = ((coord_k[:, 3] - coord_k[:, 1]) / H).view(-1, 1, 1)
    # [bs, 1, 1]
    q_start_x = coord_q[:, 0].view(-1, 1, 1)
    q_start_y = coord_q[:, 1].view(-1, 1, 1)
    k_start_x = coord_k[:, 0].view(-1, 1, 1)
    k_start_y = coord_k[:, 1].view(-1, 1, 1)

    # [bs, 1, 1]
    # q_flip_x = (coord_q[:, 0] > coord_q[:, 2]).view(-1, 1, 1)
    # q_flip_y = (coord_q[:, 1] > coord_q[:, 3]).view(-1, 1, 1)
    # q_flip_x = (2 * q_flip_x - 1).to(torch.int)
    # q_flip_y = (2 * q_flip_y - 1).to(torch.int)
    # q_bin_width = ((coord_q[:, 6] - 1) / W).view(-1, 1, 1)
    # q_bin_height = ((coord_q[:, 7] - 1) / H).view(-1, 1, 1)
    # q_bin_width = q_bin_width * q_flip_x
    # q_bin_height = q_bin_width * q_flip_y
    # k_flip_x = (coord_k[:, 0] > coord_k[:, 2]).view(-1, 1, 1)
    # k_flip_y = (coord_k[:, 1] > coord_k[:, 3]).view(-1, 1, 1)
    # k_flip_x = (2 * k_flip_x - 1).to(torch.int)
    # k_flip_y = (2 * k_flip_y - 1).to(torch.int)
    # k_bin_width = ((coord_k[:, 6] - 1) / W).view(-1, 1, 1)
    # k_bin_height = ((coord_k[:, 7] - 1) / H).view(-1, 1, 1)
    # k_bin_width = k_bin_width * k_flip_x
    # k_bin_height = k_bin_width * k_flip_y
    # # q_width = ((coord_q[:, 6] - 1).to(torch.int)).view(-1, 1, 1)
    # # q_height = ((coord_q[:, 7] - 1).to(torch.int)).view(-1, 1, 1)
    # # k_width = ((coord_k[:, 6] - 1).to(torch.int)).view(-1, 1, 1)
    # # k_height = ((coord_k[:, 7] - 1).to(torch.int)).view(-1, 1, 1)
    # # [bs, 1, 1]
    # q_start_x = coord_q[:, 4].view(-1, 1, 1)
    # q_start_y = coord_q[:, 5].view(-1, 1, 1)
    # k_start_x = coord_k[:, 4].view(-1, 1, 1)
    # k_start_y = coord_k[:, 5].view(-1, 1, 1)

    # [bs, 1, 1]
    # q_bin_diag = torch.sqrt((H * q_width) ** 2 + (W * q_height) ** 2) / torch.sqrt(W ** 2 + H ** 2)
    # k_bin_diag = torch.sqrt((H * k_width) ** 2 + (W * k_height) ** 2) / torch.sqrt(W ** 2 + H ** 2)
    # q_bin_diag = torch.sqrt(q_bin_width ** 2 + q_bin_height ** 2)
    # k_bin_diag = torch.sqrt(k_bin_width ** 2 + k_bin_height ** 2)
    q_bin_diag = torch.sqrt((q_bin_width * (W_orig - 1)) ** 2 + (q_bin_height * (H_orig - 1)) ** 2)
    k_bin_diag = torch.sqrt((k_bin_width * (W_orig - 1)) ** 2 + (k_bin_height * (H_orig - 1)) ** 2)
    max_bin_diag = torch.max(q_bin_diag, k_bin_diag)

    if not is_calc_flow:
        # [bs, 7, 7]
        center_q_x = (x_array + 0.5) * q_bin_width + q_start_x
        center_q_y = (y_array + 0.5) * q_bin_height + q_start_y
        center_k_x = (x_array + 0.5) * k_bin_width + k_start_x
        center_k_y = (y_array + 0.5) * k_bin_height + k_start_y
        # center_q_x = x_array * q_bin_width + q_start_x
        # center_q_y = y_array * q_bin_height + q_start_y
        # center_k_x = x_array * k_bin_width + k_start_x
        # center_k_y = y_array * k_bin_height + k_start_y
        center_q_x = center_q_x * (W_orig - 1)
        center_q_y = center_q_y * (H_orig - 1)
        center_k_x = center_k_x * (W_orig - 1)
        center_k_y = center_k_y * (H_orig - 1)
    else:
        q_x = (x_array + 0.5) * q_bin_width + q_start_x
        q_y = (y_array + 0.5) * q_bin_height + q_start_y
        k_x = (x_array + 0.5) * k_bin_width + k_start_x
        k_y = (y_array + 0.5) * k_bin_height + k_start_y
        # q_x = x_array * q_bin_width + q_start_x
        # q_y = y_array * q_bin_height + q_start_y
        # k_x = x_array * k_bin_width + k_start_x
        # k_y = y_array * k_bin_height + k_start_y
        q_x = q_x * (W_orig - 1)
        q_y = q_y * (H_orig - 1)
        k_x = k_x * (W_orig - 1)
        k_y = k_y * (H_orig - 1)
        center_q_x, center_q_y = add_optical_flow(flow_fwd, q_x, q_y, size)
        center_k_x, center_k_y = k_x.clone(), k_y.clone()
        # center_q_x, center_q_y = q_x.clone(), q_y.clone()
        # center_k_x, center_k_y = add_optical_flow(flow_fwd, q_x, q_y, size)

    # [bs, 49, 49]
    dist_center = torch.sqrt((center_q_x.view(-1, H * W, 1) - center_k_x.view(-1, 1, H * W)) ** 2
                             + (center_q_y.view(-1, H * W, 1) - center_k_y.view(-1, 1, H * W)) ** 2) / max_bin_diag
    pos_mask = (dist_center < pos_ratio).float().detach()

    # [bs, 49, 49]
    logit = torch.bmm(q.transpose(1, 2), k)

    loss = (logit * pos_mask).sum(-1).sum(-1) / (pos_mask.sum(-1).sum(-1) + 1e-6)

    return -2 * loss.mean()


def Proj_Head(in_dim=2048, inner_dim=4096, out_dim=256):
    return MLP2d(in_dim, inner_dim, out_dim)


def Pred_Head(in_dim=256, inner_dim=4096, out_dim=256):
    return MLP2d(in_dim, inner_dim, out_dim)


class PixPro(BaseModel):
    def __init__(self, base_encoder, args):
        super(PixPro, self).__init__(base_encoder, args)

        # parse arguments
        self.pixpro_p               = args.pixpro_p
        self.pixpro_momentum        = args.pixpro_momentum
        self.pixpro_pos_ratio       = args.pixpro_pos_ratio
        self.pixpro_clamp_value     = args.pixpro_clamp_value
        self.pixpro_transform_layer = args.pixpro_transform_layer
        self.pixpro_ins_loss_weight = args.pixpro_ins_loss_weight

        # create the encoder
        self.encoder = base_encoder(head_type='early_return')
        self.projector = Proj_Head()

        # create the encoder_k
        self.encoder_k = base_encoder(head_type='early_return')
        self.projector_k = Proj_Head()

        for param_q, param_k in zip(self.encoder.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        for param_q, param_k in zip(self.projector.parameters(), self.projector_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        nn.SyncBatchNorm.convert_sync_batchnorm(self.encoder)
        nn.SyncBatchNorm.convert_sync_batchnorm(self.encoder_k)
        nn.SyncBatchNorm.convert_sync_batchnorm(self.projector)
        nn.SyncBatchNorm.convert_sync_batchnorm(self.projector_k)

        self.K = int(args.num_instances * 1. / get_world_size() / args.batch_size * args.epochs)
        self.k = int(args.num_instances * 1. / get_world_size() / args.batch_size * (args.start_epoch - 1))

        if self.pixpro_transform_layer == 0:
            self.value_transform = Identity()
        elif self.pixpro_transform_layer == 1:
            self.value_transform = conv1x1(in_planes=256, out_planes=256)
        elif self.pixpro_transform_layer == 2:
            self.value_transform = MLP2d(in_dim=256, inner_dim=256, out_dim=256)
        else:
            raise NotImplementedError

        if self.pixpro_ins_loss_weight > 0.:
            self.projector_instance = Proj_Head()
            self.projector_instance_k = Proj_Head()
            self.predictor = Pred_Head()

            for param_q, param_k in zip(self.projector_instance.parameters(), self.projector_instance_k.parameters()):
                param_k.data.copy_(param_q.data)
                param_k.requires_grad = False

            nn.SyncBatchNorm.convert_sync_batchnorm(self.projector_instance)
            nn.SyncBatchNorm.convert_sync_batchnorm(self.projector_instance_k)
            nn.SyncBatchNorm.convert_sync_batchnorm(self.predictor)

            self.avgpool = nn.AvgPool2d(7, stride=1)

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        _contrast_momentum = 1. - (1. - self.pixpro_momentum) * (np.cos(np.pi * self.k / self.K) + 1) / 2.
        self.k = self.k + 1

        for param_q, param_k in zip(self.encoder.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * _contrast_momentum + param_q.data * (1. - _contrast_momentum)

        for param_q, param_k in zip(self.projector.parameters(), self.projector_k.parameters()):
            param_k.data = param_k.data * _contrast_momentum + param_q.data * (1. - _contrast_momentum)

        if self.pixpro_ins_loss_weight > 0.:
            for param_q, param_k in zip(self.projector_instance.parameters(), self.projector_instance_k.parameters()):
                param_k.data = param_k.data * _contrast_momentum + param_q.data * (1. - _contrast_momentum)

    def featprop(self, feat):
        N, C, H, W = feat.shape

        # Value transformation
        feat_value = self.value_transform(feat)
        feat_value = F.normalize(feat_value, dim=1)
        feat_value = feat_value.view(N, C, -1)

        # Similarity calculation
        feat = F.normalize(feat, dim=1)

        # [N, C, H * W]
        feat = feat.view(N, C, -1)

        # [N, H * W, H * W]
        attention = torch.bmm(feat.transpose(1, 2), feat)
        attention = torch.clamp(attention, min=self.pixpro_clamp_value)
        if self.pixpro_p < 1.:
            attention = attention + 1e-6
        attention = attention ** self.pixpro_p

        # [N, C, H * W]
        feat = torch.bmm(feat_value, attention.transpose(1, 2))

        return feat.view(N, C, H, W)

    def regression_loss(self, x, y):
        return -2. * torch.einsum('nc, nc->n', [x, y]).mean()

    def forward(self, im_1, im_2, coord1, coord2):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """
        # compute query features
        feat_1 = self.encoder(im_1)  # queries: NxC
        proj_1 = self.projector(feat_1)
        pred_1 = self.featprop(proj_1)
        pred_1 = F.normalize(pred_1, dim=1)

        feat_2 = self.encoder(im_2)
        proj_2 = self.projector(feat_2)
        pred_2 = self.featprop(proj_2)
        pred_2 = F.normalize(pred_2, dim=1)

        if self.pixpro_ins_loss_weight > 0.:
            proj_instance_1 = self.projector_instance(feat_1)
            pred_instacne_1 = self.predictor(proj_instance_1)
            pred_instance_1 = F.normalize(self.avgpool(pred_instacne_1).view(pred_instacne_1.size(0), -1), dim=1)

            proj_instance_2 = self.projector_instance(feat_2)
            pred_instance_2 = self.predictor(proj_instance_2)
            pred_instance_2 = F.normalize(self.avgpool(pred_instance_2).view(pred_instance_2.size(0), -1), dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            feat_1_ng = self.encoder_k(im_1)  # keys: NxC
            proj_1_ng = self.projector_k(feat_1_ng)
            proj_1_ng = F.normalize(proj_1_ng, dim=1)

            feat_2_ng = self.encoder_k(im_2)
            proj_2_ng = self.projector_k(feat_2_ng)
            proj_2_ng = F.normalize(proj_2_ng, dim=1)

            if self.pixpro_ins_loss_weight > 0.:
                proj_instance_1_ng = self.projector_instance_k(feat_1_ng)
                proj_instance_1_ng = F.normalize(self.avgpool(proj_instance_1_ng).view(proj_instance_1_ng.size(0), -1),
                                                 dim=1)

                proj_instance_2_ng = self.projector_instance_k(feat_2_ng)
                proj_instance_2_ng = F.normalize(self.avgpool(proj_instance_2_ng).view(proj_instance_2_ng.size(0), -1),
                                                 dim=1)

        # compute loss
        loss = regression_loss(pred_1, proj_2_ng, coord1, coord2, self.pixpro_pos_ratio) \
            + regression_loss(pred_2, proj_1_ng, coord2, coord1, self.pixpro_pos_ratio)

        if self.pixpro_ins_loss_weight > 0.:
            loss_instance = self.regression_loss(pred_instance_1, proj_instance_2_ng) + \
                         self.regression_loss(pred_instance_2, proj_instance_1_ng)
            loss = loss + self.pixpro_ins_loss_weight * loss_instance

        return loss
