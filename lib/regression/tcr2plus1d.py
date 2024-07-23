import torch
from torch import nn

import torchvision

class FFM(nn.Module):
    def __init__(self, in_c, out_c):
        super(FFM, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv3d(in_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_c),
            nn.ReLU(inplace=True)
        )

        self.weight = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(out_c, out_c, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_c, out_c, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Sigmoid()
        )

    def forward(self, x1, x2):

        out = torch.cat((x1, x2), 1)
        f = self.conv1(out)
        att = self.weight(f)

        f_out1 = (x1*att)+x1
        f_out2 = (x2*att)+x2

        f_out = (f_out1+f_out2)/2

        return f_out

class TCr2plus1d(nn.Module):

    def __init__(self, num_classes):
        super(TCr2plus1d, self).__init__()

        model = torchvision.models.video.r2plus1d_18(
            weights=torchvision.models.video.R2Plus1D_18_Weights.DEFAULT)
        self.new_model = nn.Sequential(*(list(model.children())[:-2]))

        self.relu = nn.ReLU(inplace=True)

        self.pool = nn.AdaptiveAvgPool3d(1)

        self.ffm = FFM(1024, 512)
        self.linear = nn.Linear(512, num_classes)

    def forward(self, x1, x2):

        out1 = self.new_model(x1)
        out2 = self.new_model(x2)

        final = self.ffm(out1, out2)

        final = self.pool(final)

        final = final.view(-1, 512)
        final = self.linear(final)

        return final