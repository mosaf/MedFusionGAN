import torch
import torchvision
import torch.nn.functional as F
from math import exp
from torch import nn
import numpy as np


class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, resize=True, device=None):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        vgg = torchvision.models.vgg16(pretrained=True).to(device)
        vgg.eval()
        blocks.append(vgg.features[:4].eval())
        blocks.append(vgg.features[4:9].eval())
        blocks.append(vgg.features[9:16].eval())
        blocks.append(vgg.features[16:23].eval())
        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.resize = resize
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1))

    def forward(self, input, target, feature_layers=[0, 1, 2, 3], style_layers=[]):
        input, target = input * 0.5 + 0.5, target * 0.5 + 0.5
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
            # target = torch.cat([target[:,0,:,:].unsqueeze(1), target[:,1,:,:].unsqueeze(1), target[:,1,:,:].unsqueeze(1)], dim = 1)
        input = (input-self.mean) / self.std
        target = (target-self.mean) / self.std
        # if self.resize:
        #     input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
        #     target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
        loss = 0.0
        x = input
        y = target
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            if i in feature_layers:
                loss += torch.nn.functional.l1_loss(x, y)
            if i in style_layers:
                act_x = x.reshape(x.shape[0], x.shape[1], -1)
                act_y = y.reshape(y.shape[0], y.shape[1], -1)
                gram_x = act_x @ act_x.permute(0, 2, 1)
                gram_y = act_y @ act_y.permute(0, 2, 1)
                loss += torch.nn.functional.l1_loss(gram_x, gram_y)
        return loss




class SSIM_Loss:
    def __init__(self, window_size=11, window=None, size_average=True, full=False, val_range=None):
        self.window_size = window_size
        self.window = window
        self.size_average = size_average
        self.full = full
        self.val_range = val_range

    @staticmethod
    def gaussian(window_size, sigma):
        # gauss = torch.tensor([exp(-(x - window_size//2)**2/(2*sigma**2)) for x in range(window_size)])
        x_cord = torch.arange(window_size)
        x_grid = x_cord.repeat(window_size).view(window_size, window_size)
        y_grid = x_grid.t()
        xy_grid = torch.stack([x_grid, y_grid], dim = -1)
        gauss = (1./(2.*3.14*sigma ** 2)) *\
                  torch.exp(
                      -torch.sum((xy_grid - window_size // 2)**2., dim=-1) /\
                      (2*sigma**2)
                  ).unsqueeze(0).unsqueeze(0)

        return gauss/gauss.sum()


    def create_window(self, window_size, channel=1):
        # _1D_window = self.gaussian(window_size, 1.5).unsqueeze(1)
        # _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = self.gaussian(window_size, 1.5).repeat(channel, 1, 1, 1)
        return window


    def ssim_loss(self, img1, img2):
        # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
        if self.val_range is None:
            if torch.max(img1) > 128:
                max_val = 255
            else:
                max_val = 1

            if torch.min(img1) < -0.5:
                min_val = -1
            else:
                min_val = 0
            L = max_val - min_val
        else:
            L = self.val_range

        padd = 0
        # img1 = torch.cat([img1, img1], dim = 1)
        (_, channel, height, width) = img1.size()
        if self.window is None:
            real_size = min(self.window_size, height, width)
            window = self.create_window(real_size, channel=channel).to(img1.device)
            # print(window.shape)
            # exit()
        mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
        mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

        C1 = (0.01 * L) ** 2
        C2 = (0.03 * L) ** 2

        v1 = 2.0 * sigma12 + C2
        v2 = sigma1_sq + sigma2_sq + C2
        cs = v1 / v2  # contrast sensitivity

        ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)
        # print(ssim_map.shape)
        # exit()
        if self.size_average:
            cs = cs.mean()
            ret = ssim_map.mean()
        else:
            cs = cs.mean(1).mean(1).mean(1)
            ret = ssim_map.mean(1).mean(1).mean(1)
        if ret > 1:
            return ret - 1
        else:
            return 1 - ret
        # if self.full:
        #     return 1-ret + 1-cs
        # else:
        #     # return 1 - 0.5 * (ret + 1)
        #     return 1 - ret




class TV_loss2(nn.Module):
    def __init__(self, REGULARIZATION=1e-3):
        super(TV_loss2, self).__init__()
        self.regularization = REGULARIZATION
        self.l1 = torch.nn.L1Loss()
    def forward(self, x, y):
        img = x - y
        bs_img, c_img, h_img, w_img = img.size()
        tv_h = torch.pow(img[:, :, 1:, :] - img[:, :, :-1, :], 2).sum()
        tv_w = torch.pow(img[:, :, :, 1:] - img[:, :, :, :-1], 2).sum()
        return self.regularization * (tv_h + tv_w) / (bs_img * c_img * h_img * w_img)

        # return self.l1(reg_loss_x, reg_loss_y)

