import torch
import torch.nn as nn

class DenseBlock(nn.Module):
    def __init__(self, in_channels, out_channels, d, growthRate, up=False):
        super(DenseBlock, self).__init__()

        mid_channels = growthRate

        self.up = up

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(
            in_channels, mid_channels, kernel_size=3, padding=1
        )
        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.conv2 = nn.Conv2d(
            mid_channels, out_channels, kernel_size=3, padding=d, dilation=d, groups=mid_channels
        )
        self.bn3 = nn.BatchNorm2d(out_channels)

        if self.up:
            self.conv3 = nn.Conv2d(
                out_channels, out_channels//2, kernel_size=1
            )
            self.conv4 = nn.Conv2d(
                in_channels, out_channels//2, kernel_size=1
            )
        else:
            self.conv3 = nn.Conv2d(
                out_channels, mid_channels, kernel_size=1
            )

    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        out = self.conv2(self.relu(self.bn2(out)))
        out = self.conv3(self.relu(self.bn3(out)))

        if self.up:
            x = self.conv4(self.relu(self.bn1(x)))

        out = torch.cat((out, x), 1)

        return self.relu(out)

class DenseLayer(nn.Module):
    def __init__(self, in_channels, out_channels, d, growthRate, layer_size):
        super(DenseLayer, self).__init__()

        layers = []

        for i in range(layer_size):
            layers.append(
                DenseBlock(in_channels, out_channels, d, growthRate)
            )

            in_channels += growthRate

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        x = self.block(x)

        return x

class Down(nn.Module):
    def __init__(self, in_channels):
        super(Down, self).__init__()

        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU()

        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)

        self.maxpool = nn.MaxPool2d(2)

    def forward(self, x):

        x = self.conv(self.relu(self.bn(x)))
        x = self.maxpool(x)

        return x

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, d, growthRate):
        super(Up, self).__init__()

        self.up = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=True
        )
        self.conv1 = DenseBlock(
            in_channels+out_channels, out_channels, d, growthRate, up=True
        )

    def forward(self, x1, x2):
        x1 = self.up(x1)

        x = torch.cat([x2, x1], dim=1)
        x = self.conv1(x)

        return x


class ConvUp(nn.Module):
    def __init__(self, in_channels, layer_size, n_classes):
        super(ConvUp, self).__init__()

        self.in_channels = in_channels
        self.reso = layer_size

        self.conv = nn.Conv2d(
            self.in_channels, n_classes, kernel_size=3, padding=1
        )
        self.up = nn.Upsample(
            scale_factor=256//self.reso, mode='bilinear', align_corners=True
        )

    def forward(self, x):

        out = self.conv(x)
        out = self.up(out)

        return out


class CDDenseUnet(nn.Module):
    def __init__(self, n_channels, n_classes, supervision=False, layer_outputs=[32, 64, 128, 256, 512]):
        super(CDDenseUnet, self).__init__()
        self.in_channels = n_channels
        self.n_classes = n_classes
        self.layers = layer_outputs
        self.num_layers = len(self.layers)

        nBlocks = self.num_layers-1

        self.rates = []
        self.supervision = supervision

        for l in self.layers:

            self.rates.append(int(l/nBlocks))

        self.conv1 = nn.Conv2d(
            self.in_channels, self.layers[0], kernel_size=7, padding=3
        )

        blockLayers = []
        down_blockLayers = []

        d_n = 2

        for i in range(nBlocks):

            blockLayers.append(
                DenseLayer(
                    self.layers[i], self.layers[i+1], d_n, self.rates[i], nBlocks)
            )
            down_blockLayers.append(Down(self.layers[i+1]))

        self.dense = nn.Sequential(*blockLayers)
        self.down_dense = nn.Sequential(*down_blockLayers)

        upLayers = []

        for i in range(nBlocks, 0, -1):
            upLayers.append(
                Up(self.layers[i], self.layers[i-1], d_n, self.rates[i]))

        self.up_dense = nn.Sequential(*upLayers)

        if self.supervision:
            mfpLayers = []

            for i in range(nBlocks):
                mfpLayers.append(
                    ConvUp(self.layers[-i-2], self.layers[i], self.n_classes))

            self.mfp = nn.Sequential(*mfpLayers)

        self.out = nn.Conv2d(self.layers[0], self.n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.conv1(x)  # B, 32, 256, 256

        out = self.dense[0](x1)
        x2 = self.down_dense[0](out)  # B, 64, 128, 128

        out = self.dense[1](x2)
        x3 = self.down_dense[1](out)  # B, 128, 64, 64

        out = self.dense[2](x3)
        x4 = self.down_dense[2](out)  # B, 256, 32, 32

        out = self.dense[3](x4)
        x5 = self.down_dense[3](out)  # B, 512, 16, 16

        up1 = self.up_dense[0](x5, x4)  # B, 256, 32, 32 WEIGHT = 2/30
        up2 = self.up_dense[1](up1, x3)  # B, 128, 64, 64 4/30
        up3 = self.up_dense[2](up2, x2)  # B, 64, 128, 128 8/30
        up4 = self.up_dense[3](up3, x1)  # B, 32, 256, 256 16/30

        f = self.out(up4)

        if self.supervision:
            f1 = self.mfp[0](up1)
            f2 = self.mfp[1](up2)
            f3 = self.mfp[2](up3)
            return f, f1, f2, f3

        return f