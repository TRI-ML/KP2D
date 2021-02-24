# Copyright 2020 Toyota Research Institute.  All rights reserved.

import torch
from torch import nn

from kp2d.utils.image import image_grid


class SpaceToDepth(nn.Module):
    def __init__(self, block_size=4):
        super().__init__()
        assert block_size == 4
        self.bs = block_size

    def forward(self, x):
        N, C, H, W = x.size()
        x = x.view(N, C, H // self.bs, self.bs, W // self.bs, self.bs)  # (N, C, H//bs, bs, W//bs, bs)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()  # (N, bs, bs, C, H//bs, W//bs)
        x = x.view(N, C * (self.bs ** 2), H // self.bs, W // self.bs)  # (N, C*bs^2, H//bs, W//bs)
        return x


@torch.jit.script
class SpaceToDepthJit(object):
    def __call__(self, x: torch.Tensor):
        # assuming hard-coded that block_size==4 for acceleration
        N, C, H, W = x.size()
        x = x.view(N, C, H // 4, 4, W // 4, 4)  # (N, C, H//bs, bs, W//bs, bs)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()  # (N, bs, bs, C, H//bs, W//bs)
        x = x.view(N, C * 16, H // 4, W // 4)  # (N, C*bs^2, H//bs, W//bs)
        return x


class SpaceToDepthModule(nn.Module):
    def __init__(self, remove_model_jit=False):
        super().__init__()
        if not remove_model_jit:
            self.op = SpaceToDepthJit()
        else:
            self.op = SpaceToDepth()

    def forward(self, x):
        return self.op(x)


def get_vanilla_conv_block(in_channels, out_channels, momentum=0.9):
    return torch.nn.Sequential(
        torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
        torch.nn.BatchNorm2d(out_channels, momentum=momentum)
    )


class SepConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels,
                 stride=1, dilation=1, bias=True,
                 residual_concat=False, group_size=16,
                 ):
        super().__init__()
        self.in_features = in_channels
        self.residual_concat = residual_concat
        self.depthwise = nn.Conv2d(in_channels, in_channels, groups=in_channels // group_size or 1, kernel_size=3,
                                   padding=1, stride=stride, dilation=dilation, bias=bias)
        self.pointwise = nn.Conv2d(in_channels * (2 if residual_concat else 1), out_channels, kernel_size=1, bias=False)
        self.bn = nn.Sequential(nn.BatchNorm2d(out_channels), nn.LeakyReLU(inplace=False))

    def forward(self, x):
        features = torch.cat([x, self.depthwise(x)], dim=1) if self.residual_concat else x + self.depthwise(x)
        y = self.pointwise(features)
        return self.bn(y)


class MixSepConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels,
                 stride=1, dilation=1, bias=True,
                 residual_concat=False, group_size=16,
                 ):
        super().__init__()
        if in_channels % 2:
            old_features = in_channels
            in_channels = old_features + 1
            self.prep = nn.Conv2d(old_features, in_channels, kernel_size=1, bias=False)
        self.in_features = in_channels
        self.residual_concat = residual_concat
        self.dw3 = nn.Conv2d(in_channels // 2, in_channels // 2,
                             groups=in_channels // (2 * group_size) or 1,
                             kernel_size=3,
                             padding=1, stride=stride, dilation=dilation, bias=bias)
        self.dw5 = nn.Conv2d(in_channels // 2, in_channels // 2,
                             groups=in_channels // (2 * group_size) or 1,
                             kernel_size=5,
                             padding=2, stride=stride, dilation=dilation, bias=bias)

        self.pw = nn.Conv2d(in_channels * (2 if residual_concat else 1), out_channels, kernel_size=1, bias=False)
        self.bn = nn.Sequential(nn.BatchNorm2d(out_channels), nn.LeakyReLU(inplace=False))

    def forward(self, x):
        if x.shape[1] < 4:
            x = self.prep(x)
        q = self.in_features // 2
        x3 = self.dw3(x[:, :q, :, :])
        x5 = self.dw5(x[:, q:, :, :])
        features = torch.cat([x3, x5, x], dim=1) if self.residual_concat else torch.cat([x3, x5], dim=1) + x
        y = self.pw(features)
        return self.bn(y)

@torch.jit.script
def _preprocess_grid(x, cell: int, step: float):
    return x.mul(cell) + step


class Stem(torch.nn.Module):
    def __init__(self,
                 with_drop,
                 base_conv_block=get_vanilla_conv_block,
                 ):
        super().__init__()
        c0, c1, c2, c3, c4 = 3, 32, 64, 128, 256

        self.conv1a = base_conv_block(c0, c1)
        self.conv1b = base_conv_block(c1, c1)
        self.conv2a = base_conv_block(c1, c2)
        self.conv2b = base_conv_block(c2, c2)
        self.conv3a = base_conv_block(c2, c3)
        self.conv3b = base_conv_block(c3, c3)
        self.conv4a = base_conv_block(c3, c4)
        self.conv4b = base_conv_block(c4, c4)
        self.relu = torch.nn.LeakyReLU(inplace=True)
        if with_drop:
            self.dropout = torch.nn.Dropout2d(0.2)
        else:
            self.dropout = None
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        B, _, H, W = x.shape

        x = self.relu(self.conv1a(x))
        x = self.relu(self.conv1b(x))
        if self.dropout:
            x = self.dropout(x)
        x = self.pool(x)
        x = self.relu(self.conv2a(x))
        x = self.relu(self.conv2b(x))
        if self.dropout:
            x = self.dropout(x)
        x = self.pool(x)
        x = self.relu(self.conv3a(x))
        skip = self.relu(self.conv3b(x))
        if self.dropout:
            skip = self.dropout(skip)
        x = self.pool(skip)
        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))
        if self.dropout:
            x = self.dropout(x)

        return x, skip


class KeypointNet(torch.nn.Module):
    """
    Keypoint detection network.

    Parameters
    ----------
    use_color : bool
        Use color or grayscale images.
    do_upsample: bool
        Upsample desnse descriptor map.
    with_drop : bool
        Use dropout.
    do_cross: bool
        Predict keypoints outside cell borders.
    kwargs : dict
        Extra parameters
    """

    def __init__(self, base_conv_block=get_vanilla_conv_block,
                 use_color=True, do_upsample=True, with_drop=True, do_cross=True,
                 **kwargs):
        super().__init__()

        self.training = True

        self.use_color = use_color
        self.with_drop = with_drop
        self.do_cross = do_cross
        self.do_upsample = do_upsample

        self.bn_momentum = 0.1
        self.cross_ratio = 2.0

        if self.do_cross is False:
            self.cross_ratio = 1.0

        c4, c5, d1 = 256, 256, 512

        self.stem = Stem(base_conv_block=base_conv_block, with_drop=with_drop)

        # Score Head.
        self.convDa = base_conv_block(c4, c5)
        self.convDb = torch.nn.Conv2d(c5, 1, kernel_size=3, stride=1, padding=1)

        # Location Head.
        self.convPa = base_conv_block(c4, c5)
        self.convPb = torch.nn.Conv2d(c5, 2, kernel_size=3, stride=1, padding=1)

        # Desc Head.
        self.convFa = base_conv_block(c4, c5)
        self.convFb = base_conv_block(c5, d1)
        self.convFaa = base_conv_block(c4, c5)
        self.convFbb = torch.nn.Conv2d(c5, 256, kernel_size=3, stride=1, padding=1)

        self.relu = torch.nn.LeakyReLU(inplace=True)
        if self.with_drop:
            self.dropout = torch.nn.Dropout2d(0.2)
        else:
            self.dropout = None
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.cell = 8
        self.upsample = torch.nn.PixelShuffle(upscale_factor=2)

    def forward(self, x):
        """
        Processes a batch of images.

        Parameters
        ----------
        x : torch.Tensor
            Batch of input images (B, 3, H, W)

        Returns
        -------
        score : torch.Tensor
            Score map (B, 1, H_out, W_out)
        coord: torch.Tensor
            Keypoint coordinates (B, 2, H_out, W_out)
        feat: torch.Tensor
            Keypoint descriptors (B, 256, H_out, W_out)
        """
        B, _, H, W = x.shape
        x, skip = self.stem(x)
        B, _, Hc, Wc = x.shape

        score = self.relu(self.convDa(x))
        if self.dropout:
            score = self.dropout(score)
        score = self.convDb(score).sigmoid()

        border_mask = torch.ones(B, Hc, Wc)
        border_mask[:, 0] = 0
        border_mask[:, Hc - 1] = 0
        border_mask[:, :, 0] = 0
        border_mask[:, :, Wc - 1] = 0
        border_mask = border_mask.unsqueeze(1)
        score = score * border_mask.to(score.device)

        center_shift = self.relu(self.convPa(x))
        if self.dropout:
            center_shift = self.dropout(center_shift)
        center_shift = self.convPb(center_shift).tanh()

        step = (self.cell - 1) / 2.
        center_base = image_grid(B, Hc, Wc,
                                 ones=False, normalized=False)
        center_base = _preprocess_grid(center_base, self.cell, step).to(center_shift.device)
        coord_un = center_base.add(center_shift.mul(self.cross_ratio * step))
        coord = coord_un.clone()
        coord[:, 0] = torch.clamp(coord_un[:, 0], min=0, max=W - 1)
        coord[:, 1] = torch.clamp(coord_un[:, 1], min=0, max=H - 1)

        feat = self.relu(self.convFa(x))
        if self.dropout:
            feat = self.dropout(feat)
        if self.do_upsample:
            feat = self.upsample(self.convFb(feat))
            feat = torch.cat([feat, skip], dim=1)
        feat = self.relu(self.convFaa(feat))
        feat = self.convFbb(feat)

        if self.training is False:
            coord_norm = coord[:, :2].clone()
            coord_norm[:, 0] = (coord_norm[:, 0] / (float(W - 1) / 2.)) - 1.
            coord_norm[:, 1] = (coord_norm[:, 1] / (float(H - 1) / 2.)) - 1.
            coord_norm = coord_norm.permute(0, 2, 3, 1)

            feat = torch.nn.functional.grid_sample(feat, coord_norm, align_corners=True)

            dn = torch.norm(feat, p=2, dim=1)  # Compute the norm.
            feat = feat.div(torch.unsqueeze(dn, 1))  # Divide by norm to normalize.
        return score, coord, feat
