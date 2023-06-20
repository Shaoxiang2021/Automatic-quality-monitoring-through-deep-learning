"""
implementation of model classes
"""

import torch
import torch.nn as nn


class ResidualBlock2D(nn.Module):
    def __init__(self, in_channels, channels, stride):
        """
        2D Residual Block consisting of 2 Conv Layers with BatchNorm.
        If in_channels != channels or stride != 1, downsampling is
        performed on the identity tensor automatically.

        Inherits nn.Module class and overwrites forward method.

        Args:
            in_channels: number if incoming channels
            channels: number if output channels
            stride: stride in conv layers
        """

        super(ResidualBlock2D, self).__init__()

        # figure out if downsampling is needed
        if in_channels != channels or stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(num_features=channels)
            )
        else:
            self.downsample = None

        # construct layers
        self.a = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=channels,
                               kernel_size=(3, 3),
                               stride=(stride, stride),
                               padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=channels)
        self.conv2 = nn.Conv2d(in_channels=channels,
                               out_channels=channels,
                               kernel_size=(3, 3),
                               stride=(1, 1),
                               padding=1)
        self.bn2 = nn.BatchNorm2d(num_features=channels)

    def forward(self, x):
        out = self.a(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            x = self.downsample(x)
        out = self.a(out + x)
        return out


class MyModel(nn.Module):

    def __init__(self, in_channels, num_classes, channels, num_blocks):
        """
        a basic ResNet inspired 2D CNN model.

        Args:
            in_channels: image channels (e.g. 1 or 3)
            num_classes: number if output classes
            channels: channels for residual blocks
            num_blocks: number of residual blocks
        """
        super(MyModel, self).__init__()

        # define first conv an max pool layer
        self.conv_in = nn.Conv2d(in_channels=in_channels,
                                 out_channels=channels,
                                 kernel_size=(7, 7),
                                 stride=(2, 2),
                                 padding=(3, 3))
        self.bn_in = nn.BatchNorm2d(num_features=channels)
        self.maxpool_in = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # define as many residual blocks as specified in a nn.ModuleList
        self.res_blocks = nn.ModuleList([ResidualBlock2D(in_channels=channels,
                                                         channels=channels,
                                                         stride=1) for i in range(0, num_blocks)])

        # average pooling over channel dims  and linear classifier
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(in_features=channels, out_features=num_classes)

        # activation function
        self.a = nn.ReLU()

        # initialize conv and batchnorm layers with custom distributions (optional)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):

        # hardcoded for first layers
        out = self.a(self.bn_in(self.conv_in(x)))
        out = self.maxpool_in(out)

        # loop through res. blocks
        for rb in self.res_blocks:
            out = rb(out)

        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        return self.fc(out)

