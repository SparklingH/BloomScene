#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
from math import exp

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch import Tensor, nn
import numpy as np
import cv2
from typing import Literal
from enum import Enum
import abc
from utils.general import build_rotation
from scipy.ndimage import median_filter


class CMD(nn.Module):
    def __init__(self):
        super(CMD, self).__init__()

    def forward(self, x1, x2, n_moments=5):
        x1 = torch.clamp(x1, min=-1e6, max=1e6)
        x2 = torch.clamp(x2, min=-1e6, max=1e6)
         
        assert not torch.isnan(x1).any(), "x1 contains NaN"
        assert not torch.isinf(x1).any(), "x1 contains Inf"
        assert not torch.isnan(x2).any(), "x2 contains NaN"
        assert not torch.isinf(x2).any(), "x2 contains Inf"

        mx1 = torch.mean(x1, 0)
        mx2 = torch.mean(x2, 0)
        sx1 = x1 - mx1
        sx2 = x2 - mx2
        dm = self.matchnorm(mx1, mx2)
        scms = dm
        for i in range(n_moments - 1):
            scms = scms + self.scm(sx1, sx2, i + 2)
        return scms / x1.shape[0]

    def matchnorm(self, x1, x2):
         
        power = torch.clamp(torch.pow(torch.abs(x1 - x2) + 1e-6, 2), max=1e6)
        summed = torch.clamp(torch.sum(power), max=1e6)
        sqrt = torch.sqrt(summed + 1e-6)
        return sqrt

    def scm(self, sx1, sx2, k):
         
        ss1 = torch.mean(torch.pow(torch.abs(sx1) + 1e-6, k), 0)
        ss2 = torch.mean(torch.pow(torch.abs(sx2) + 1e-6, k), 0)
        return self.matchnorm(ss1, ss2)


def bilateral_filter(depth, spatial_sigma=2.0, color_sigma=0.1, kernel_size=5):
    batch_size, height, width = depth.shape
    x = torch.arange(kernel_size, dtype=torch.float32).to(depth.device) - kernel_size // 2
    y = x.unsqueeze(0).expand(kernel_size, kernel_size)
    spatial_kernel = torch.exp(- (y**2 + y.t()**2) / (2 * spatial_sigma**2))
    spatial_kernel = spatial_kernel / spatial_kernel.sum()

    depth_padded = F.pad(depth, (kernel_size // 2, kernel_size // 2, kernel_size // 2, kernel_size // 2), mode='replicate')
    depth_padded = depth_padded.unsqueeze(1)
    depth_unfolded = F.unfold(depth_padded, kernel_size=kernel_size).view(batch_size, 1, kernel_size, kernel_size, height, width)
    depth_unfolded = depth_unfolded.permute(0, 4, 5, 1, 2, 3).squeeze(3)

    color_diff = (depth.unsqueeze(3).unsqueeze(4) - depth_unfolded).abs()
    color_kernel = torch.exp(-color_diff / (2 * color_sigma**2))
    bilateral_loss = torch.sum(spatial_kernel * color_kernel * (depth.unsqueeze(3).unsqueeze(4) - depth_unfolded)**2, dim=(3, 4))
    bilateral_loss = bilateral_loss.mean()

    return bilateral_loss


def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()


def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)



def image2canny(image, thres1, thres2, isEdge1=True):
    canny_mask = torch.from_numpy(cv2.Canny((image.detach().cpu().numpy()*255.).astype(np.uint8), thres1, thres2)/255.)
    if not isEdge1:
        canny_mask = 1. - canny_mask
    return canny_mask.float()


class DepthLossType(Enum):
    """Enum for specifying depth loss"""
    HuberL1 = "HuberL1"


class DepthLoss(nn.Module):
    """Factory method class for various depth losses"""
    def __init__(self, depth_loss_type: DepthLossType, **kwargs):
        super().__init__()
        self.depth_loss_type = depth_loss_type
        self.kwargs = kwargs
        self.loss = self._get_loss_instance()

    @abc.abstractmethod
    def forward(self, *args) -> Tensor:
        return self.loss(*args)

    def _get_loss_instance(self) -> nn.Module:
        if self.depth_loss_type == DepthLossType.HuberL1:
            return HuberL1(**self.kwargs)
        else:
            raise ValueError(f"Unsupported loss type: {self.depth_loss_type}")

       

class HuberL1(nn.Module):
    def __init__(
        self,
        tresh=0.2,
        implementation: Literal["scalar", "per-pixel"] = "scalar",
        **kwargs,
    ):
        super().__init__()
        self.tresh = tresh
        self.implementation = implementation

    def forward(self, pred, gt, rgb: Tensor):
        l1 = torch.abs(pred - gt)
        d = self.tresh * torch.max(l1)
        loss = ((pred - gt) ** 2 + d**2) / (2 * d)
        loss[l1 >= d] = l1[l1 >= d]

        grad_img_x = torch.mean(
            torch.abs(rgb[..., :, :-1, :] - rgb[..., :, 1:, :]), -1, keepdim=True
        )
        grad_img_y = torch.mean(
            torch.abs(rgb[..., :-1, :, :] - rgb[..., 1:, :, :]), -1, keepdim=True
        )

        loss = loss.reshape(512, 512).unsqueeze(0).unsqueeze(-1)

        loss_x = torch.exp(-grad_img_x) * loss[..., :, :-1, :]
        loss_y = torch.exp(-grad_img_y) * loss[..., :-1, :, :]

        if self.implementation == "scalar":
            return loss_x.mean() + loss_y.mean()
        else:
            return loss
