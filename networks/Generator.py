import torch
from torch import nn
from torch.nn import functional as F

class up_conv(nn.Module):
    """
    Up Convolution Block
    """

    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
            # nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class Residual_down(nn.Module):
    """The residual block of ResNet"""

    def __init__(self, input_channels: int, num_channels: int, use_1x1conv: bool = True, stride: int = 1):
        super(Residual_down, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = input_channels, out_channels = num_channels, kernel_size = 4,
                               stride = 2, padding = (stride, stride))
        self.conv2 = nn.Conv2d(in_channels = num_channels, out_channels = num_channels, kernel_size = (3, 3),
                               padding = 1)

        if use_1x1conv:
            self.conv3 = nn.Conv2d(in_channels = input_channels, out_channels = num_channels, kernel_size = 4,
                                   stride = 2, padding = 1)
        else:
            self.conv3 = None

        self.bn1 = nn.BatchNorm2d(num_features = num_channels)
        self.bn2 = nn.BatchNorm2d(num_features = num_channels)
        self.activation = nn.LeakyReLU(negative_slope = 0.2)

    def forward(self, X):
        Y = self.activation(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return self.activation(Y)


class residual_fianl(nn.Module):
    """The residual block of ResNet"""

    def __init__(self, input_channels: int, num_channels: int, use_1x1conv: bool = True, stride: int = 1, useReLu=False):
        super(residual_fianl, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = input_channels, out_channels = num_channels, kernel_size = (3, 3),
                               stride = 1, padding = (stride, stride))
        self.conv2 = nn.Conv2d(in_channels = num_channels, out_channels = num_channels, kernel_size = (3, 3),
                               padding = 1)

        if use_1x1conv:
            self.conv3 = nn.Conv2d(in_channels = input_channels, out_channels = num_channels, kernel_size = 3,
                                   stride = 1, padding = 1)
        else:
            self.conv3 = None

        self.bn1 = nn.BatchNorm2d(num_features = num_channels)
        self.bn2 = nn.BatchNorm2d(num_features = num_channels)
        self.activation = nn.LeakyReLU(negative_slope = 0.2)
        self.useReLu = useReLu
    def forward(self, X):
        Y = self.activation(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        if self.useReLu:
            return self.activation(Y)
        else:
            # return Y
            return torch.tanh(Y)



class Gen_unet02(nn.Module):
    def __init__(self, in_channel = 1, out_channel = 1, features = 32):
        super(Gen_unet02, self).__init__()
        self.initial_down = nn.Sequential(
            nn.Conv2d(in_channel, features, 4, 2, 1, padding_mode = 'reflect'),
            nn.LeakyReLU(negative_slope = 0.2),
        )  # 128
        self.down1 = Residual_down(features, features * 2)  # 32 --> 64
        self.down2 = Residual_down(features * 2, features * 4)  # 64 --> 128
        self.down3 = Residual_down(features * 4, features * 8)  # 128 --> 256
        self.down4 = Residual_down(features * 8, features * 16)  # 256 --> 512
        self.down5 = Residual_down(features * 16, features * 16)  # 512 --> 512
        self.down6 = Residual_down(features * 16, features * 16)  # 512 --> 512
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features * 16, features * 32, 4, 2, 1, padding_mode = 'reflect'),
            nn.LeakyReLU(negative_slope = 0.2))  # 512 --> 1024

        self.up1 = up_conv(features * 32, features * 16)  # 1024 --> 512
        self.up2 = up_conv(features * 16 * 2, features * 16)  # 1024 --> 512
        self.up3 = up_conv(features * 16 * 2, features * 16)  # 1024 --> 256
        self.up4 = up_conv(features * 16 * 2, features * 8)  # 512 --> 256
        self.up5 = up_conv(features * 8 * 2, features * 4)  # 512 --> 256
        self.up6 = up_conv(features * 4 * 2, features * 2)  # 256 --> 128
        self.up7 = up_conv(features * 2 * 2, features)  # 128 --> 32

        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(features, out_channel, kernel_size = 4, stride = 2, padding = 1)
            # ,nn.Tanh()
        )  ### for RGB

        self.final_up1 = residual_fianl(out_channel, out_channel, useReLu=True)
        self.final_up2 = residual_fianl(out_channel, out_channel, useReLu=False)
        # self.final_up3 = residual_fianl(out_channel, out_channel, useReLu=False)

    def forward(self, x, y):
        z = torch.cat([x, y], dim=1)
        d0 = self.initial_down(z)
        d1 = self.down1(d0)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)

        #
        bottleneck = self.bottleneck(d6)
        #
        up1 = self.up1(bottleneck)
        up2 = self.up2(torch.cat([up1, d6], dim = 1))
        up3 = self.up3(torch.cat([up2, d5], dim = 1))
        up4 = self.up4(torch.cat([up3, d4], dim = 1))
        up5 = self.up5(torch.cat([up4, d3], dim = 1))
        up6 = self.up6(torch.cat([up5, d2], dim = 1))
        up7 = self.up7(torch.cat([up6, d1], dim = 1))
        out = self.final_up(up7)
        out = self.final_up1(out)
        out = self.final_up2(out)
        # out = self.final_up3(out)
        return out



