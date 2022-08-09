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


def regression_loss_same(q, k):
    """ q, k: N * C * H * W
        coord_q, coord_k: N * 4 (x_upper_left, y_upper_left, x_lower_right, y_lower_right)
    """
    # if torch.distributed.get_rank() == 0:
    #     print("in same loss")

    N, C, H, W = q.shape
    device = q.device
    # [bs, feat_dim, 49]
    q = q.view(N, C, -1)
    k = k.view(N, C, -1)

    diag_mask = torch.diag(torch.ones(H * W, dtype=torch.bool)).to(device)
    pos_mask = diag_mask.float()
    # if torch.distributed.get_rank() == 0:
    #     print("pos_mask_sum_shape:", pos_mask.sum(-1).sum(-1).shape, pos_mask.sum(-1).shape, pos_mask.shape)
    #     print("pos_mask_sum:", pos_mask.sum(-1).sum(-1), pos_mask.sum(-1))

    # [bs, 49, 49]
    logit = torch.bmm(q.transpose(1, 2), k)

    loss = (logit * pos_mask).sum(-1).sum(-1) / (pos_mask.sum(-1).sum(-1) + 1e-6)

    return -2 * loss.mean()


def flow_loss_dataset(q, k, coord_q, coord_k, mask):
    q_valid = F.grid_sample(q, coord_q.permute(0, 2, 3, 1),
                            align_corners=True).permute(0, 2, 3, 1)[mask]
    k_valid = F.grid_sample(k, coord_k.permute(0, 2, 3, 1),
                            align_corners=True).permute(0, 2, 3, 1)[mask]
    loss = F.cosine_similarity(q_valid, k_valid, -1, eps=1e-6)
    return loss.mean()


def flowe_loss(q, k, coord_q, coord_k):
    """ q, k: N * C * H * W
        coord_q, coord_k: N * 4 (x_upper_left, y_upper_left, x_lower_right, y_lower_right)
    """
    N, C, H, W = q.shape

    if isinstance(coord_q, list):
        coord_q, _ = coord_q
        coord_k, _ = coord_k

    # generate center_coord, width, height
    # [1, 7, 7]
    x_array = torch.arange(0., float(W), dtype=coord_q.dtype, device=coord_q.device).view(1, 1, -1).repeat(1, H, 1)
    y_array = torch.arange(0., float(H), dtype=coord_q.dtype, device=coord_q.device).view(1, -1, 1).repeat(1, 1, W)
    # [bs, 1, 1]
    q_bin_width = ((coord_q[:, 2] - coord_q[:, 0]) / W).view(-1, 1, 1)
    q_bin_height = ((coord_q[:, 3] - coord_q[:, 1]) / H).view(-1, 1, 1)
    # k_bin_width = ((coord_k[:, 2] - coord_k[:, 0]) / W).view(-1, 1, 1)
    # k_bin_height = ((coord_k[:, 3] - coord_k[:, 1]) / H).view(-1, 1, 1)
    k_bin_width = ((coord_k[:, 2] - coord_k[:, 0])).view(-1, 1, 1)
    k_bin_height = ((coord_k[:, 3] - coord_k[:, 1])).view(-1, 1, 1)
    # [bs, 1, 1]
    q_start_x = coord_q[:, 0].view(-1, 1, 1)
    q_start_y = coord_q[:, 1].view(-1, 1, 1)
    k_start_x = coord_k[:, 0].view(-1, 1, 1)
    k_start_y = coord_k[:, 1].view(-1, 1, 1)

    # [bs, 7, 7]
    center_q_x = (x_array + 0.5) * q_bin_width + q_start_x
    center_q_y = (y_array + 0.5) * q_bin_height + q_start_y
    # if torch.distributed.get_rank() == 0:
    #     print("qx", center_q_x[:, 0, -1], coord_q[:, 2])
    #     print("qy", center_q_y[:, -1, 0], coord_q[:, 3])
    #     print("ky", center_k_y[:, -1, 0], coord_k[:, 3])
    #     print("kx", center_k_x[:, 0, -1], coord_k[:, 2])

    # [bs, 7, 7]
    relative_center_q_x = (center_q_x - k_start_x) / k_bin_width
    relative_center_q_y = (center_q_y - k_start_y) / k_bin_height
    relative_center_q_x_norm = relative_center_q_x * 2 - 1
    relative_center_q_y_norm = relative_center_q_y * 2 - 1

    pos_mask_x = torch.abs(relative_center_q_x_norm) < 1
    pos_mask_y = torch.abs(relative_center_q_y_norm) < 1
    pos_mask = pos_mask_x & pos_mask_y

    # [2, bs, 7, 7]
    grid_k = torch.stack([relative_center_q_x_norm, relative_center_q_y_norm])
    # [bs, 7, 7, 2]
    grid_k = grid_k.permute(1, 2, 3, 0)

    q_mask = q.permute(0, 2, 3, 1)[pos_mask]
    # [bs, feat_dim, 7, 7]
    k_mask = F.grid_sample(k, grid_k, align_corners=True)

    k_mask = k_mask.permute(0, 2, 3, 1)[pos_mask]
    # if torch.distributed.get_rank() == 0:
    #     print("q_mask", q_mask.shape)
    #     print("k_mask", k_mask.shape)

    loss = F.cosine_similarity(q_mask, k_mask, -1, 1e-6)

    return -2 * loss.mean()


def regression_loss(q, k, coord_q, coord_k, pos_ratio=0.5, is_flowe=False, same_loss=False):
    """ q, k: N * C * H * W
        coord_q, coord_k: N * 4 (x_upper_left, y_upper_left, x_lower_right, y_lower_right)
    """
    N, C, H, W = q.shape
    if same_loss:
        return regression_loss_same(q, k)

    if is_flowe:
        if isinstance(coord_q, list):
            return flow_loss_dataset(q, k, coord_q, coord_k)
        # max_norm_diag = (1 / H) ** 2 + (1 / W) ** 2
        # pos_ratio = torch.sqrt(torch.tensor(max_norm_diag)) / 2
        return flowe_loss(q, k, coord_q, coord_k)

    if isinstance(coord_q, list):
        coord_q, flow_fwd = coord_q
        coord_k, flow_bwd = coord_k

    # [bs, feat_dim, 49]
    q = q.view(N, C, -1)
    k = k.view(N, C, -1)

    # generate center_coord, width, height
    # [1, 7, 7]
    x_array = torch.arange(0., float(W), dtype=coord_q.dtype, device=coord_q.device).view(1, 1, -1).repeat(1, H, 1)
    y_array = torch.arange(0., float(H), dtype=coord_q.dtype, device=coord_q.device).view(1, -1, 1).repeat(1, 1, W)

    q_bins, k_bins, max_bin_diag = calc_diag(coord_q, coord_k, H, W)
    # [bs, 1, 1]
    q_bin_width, q_bin_height = q_bins
    k_bin_width, k_bin_height = k_bins

    # [bs, 1, 1]
    q_start_x = coord_q[:, 0].view(-1, 1, 1)
    q_start_y = coord_q[:, 1].view(-1, 1, 1)
    k_start_x = coord_k[:, 0].view(-1, 1, 1)
    k_start_y = coord_k[:, 1].view(-1, 1, 1)
    # if torch.distributed.get_rank() == 0:
    #     print("q_bin_width:", q_bin_width, "q_bin_height:", q_bin_height)
    #     print("q_start_x:", q_start_x, "q_start_y:", q_start_y)
    #     print("k_bin_width:", k_bin_width, "k_bin_height:", k_bin_height)
    #     print("k_start_x:", k_start_x, "k_start_y:", k_start_y)

    # [bs, 7, 7]
    # center_q_x = (x_array + 0.5) * q_bin_width + q_start_x
    # center_q_y = (y_array + 0.5) * q_bin_height + q_start_y
    # center_k_x = (x_array + 0.5) * k_bin_width + k_start_x
    # center_k_y = (y_array + 0.5) * k_bin_height + k_start_y
    center_q_x = x_array * q_bin_width + q_start_x
    center_q_y = y_array * q_bin_height + q_start_y
    center_q_x = 2 * center_q_x - 1
    center_q_y = 2 * center_q_y - 1
    k_x = x_array * k_bin_width + k_start_x
    k_y = y_array * k_bin_height + k_start_y
    k_x = 2 * k_x - 1
    k_y = 2 * k_y - 1
    k_grid = torch.stack([k_x, k_y]).permute(1, 0, 2, 3)

    H_in, W_in = flow_fwd.shape[-2:]
    init_grid = torch.meshgrid(torch.arange(H_in), torch.arange(W_in))
    init_grid = torch.stack(init_grid[::-1], dim=0).repeat(N, 1, 1, 1)
    init_grid = init_grid.float().to(flow_fwd.device)
    flow_fwd_grid = init_grid + flow_fwd
    flow_fwd_grid = F.grid_sample(flow_fwd_grid, k_grid.permute(0, 2, 3, 1))
    center_k_x = 2 * flow_fwd_grid[:, 0] / (W_in - 1) - 1
    center_k_y = 2 * flow_fwd_grid[:, 1] / (H_in - 1) - 1

    # flow_fwd_grid = F.grid_sample(flow_fwd, k_grid)
    # center_k_x = center_k_x + flow_fwd_grid[:, 0]
    # center_k_y = center_k_y + flow_fwd_grid[:, 1]

    # [bs, 49, 49]
    dist_center = torch.sqrt((center_q_x.view(-1, H * W, 1) - center_k_x.view(-1, 1, H * W)) ** 2
                             + (center_q_y.view(-1, H * W, 1) - center_k_y.view(-1, 1, H * W)) ** 2) / max_bin_diag
    pos_mask = (dist_center < pos_ratio).float().detach()
    # if torch.distributed.get_rank() == 0:
    #     print("pos_mask_sum_shape:", pos_mask.sum(-1).sum(-1).shape, pos_mask.sum(-1).shape, pos_mask.shape)
    #     print("pos_mask_sum:", pos_mask.sum(-1).sum(-1), pos_mask.sum(-1))

    # [bs, 49, 49]
    logit = torch.bmm(q.transpose(1, 2), k)

    loss = (logit * pos_mask).sum(-1).sum(-1) / (pos_mask.sum(-1).sum(-1) + 1e-6)

    return -2 * loss.mean()


def calc_diag(coord_q, coord_k, H, W):
    # [bs, 1, 1]
    q_bin_width = ((coord_q[:, 2] - coord_q[:, 0]) / W).view(-1, 1, 1)
    q_bin_height = ((coord_q[:, 3] - coord_q[:, 1]) / H).view(-1, 1, 1)
    k_bin_width = ((coord_k[:, 2] - coord_k[:, 0]) / W).view(-1, 1, 1)
    k_bin_height = ((coord_k[:, 3] - coord_k[:, 1]) / H).view(-1, 1, 1)

    # [bs, 1, 1]
    q_bin_diag = torch.sqrt(q_bin_width ** 2 + q_bin_height ** 2)
    k_bin_diag = torch.sqrt(k_bin_width ** 2 + k_bin_height ** 2)
    max_bin_diag = torch.max(q_bin_diag, k_bin_diag)

    return [q_bin_width, q_bin_height], [k_bin_width, k_bin_height], max_bin_diag


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
        self.pixpro_no_headsim      = args.pixpro_no_headsim
        self.flowe_loss             = args.flowe_loss

        self.same_loss = False
        # if self.flowe_loss and args.aug in ["mySimCLR", "myBYOL"]:
        #     self.same_loss = True

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

        if self.pixpro_no_headsim:
            feat = feat_value
        else:
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
        # if torch.distributed.get_rank() == 0:
        #     print("im_1_shape:", im_1.shape, "im_2_shape:", im_2.shape)
        #     print("feat_1_shape:", feat_1.shape)
        #     print("proj_1_shape:", proj_1.shape)
        #     print("pred_1_shape:", pred_1.shape)

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
        loss = regression_loss(pred_1, proj_2_ng, coord1, coord2, self.pixpro_pos_ratio, self.flowe_loss, self.same_loss) \
            + regression_loss(pred_2, proj_1_ng, coord2, coord1, self.pixpro_pos_ratio, self.flowe_loss, self.same_loss)

        if self.pixpro_ins_loss_weight > 0.:
            loss_instance = self.regression_loss(pred_instance_1, proj_instance_2_ng) + \
                         self.regression_loss(pred_instance_2, proj_instance_1_ng)
            loss = loss + self.pixpro_ins_loss_weight * loss_instance

        return loss
