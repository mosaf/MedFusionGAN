import torch
import torch.nn as nn


class CNNBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride=2):
        super(CNNBlock, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_channel, out_channel, kernel_size=(4, 4), stride=(stride, stride), bias=False, padding_mode='reflect'),
                                   nn.BatchNorm2d(out_channel),
                                   # nn.LeakyReLU(0.2))
                                   nn.LeakyReLU(negative_slope=0.2))

    def forward(self, x):
        return self.conv(x)


class Discriminator(nn.Module):
    def __init__(self, in_channels=1, features=None): # 256 -> 30x30
        super(Discriminator, self).__init__()
        if features is None:
            features = [64, 128, 256, 512]
        in_channels = in_channels * 2
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, features[0], kernel_size=(4, 4), stride=(2, 2), padding=1, padding_mode='reflect'),
            nn.LeakyReLU(negative_slope=0.2)
        )

        layers = []
        in_channels = features[0]
        for i, feature in enumerate(features[1::]):
            layers.append(
                CNNBlock(in_channels, feature, stride=1 if feature == features[-1] else 2)
            )
            in_channels = feature
        layers.append(nn.Conv2d(in_channels, 1, kernel_size=(4, 4), padding=1, padding_mode='reflect'))
        layers.append(nn.Sigmoid())
        self.model = nn.Sequential(*layers)

    def forward(self, x, y):
        x = torch.cat([x, y], dim=1)
        x = self.initial(x)
        return self.model(x)




