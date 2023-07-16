import torch
import torch.nn as nn
import torch.nn.functional as F
from torchstat import stat

class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1,
                      bias=False),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class SegNet(nn.Module):
    def __init__(self, in_channel, num_classes, preprocess_flag):
        super().__init__()
        self.preprocess_flag = preprocess_flag
        self.encode1 = nn.Sequential(
            ConvBNReLU(in_channels=in_channel, out_channels=64),
            ConvBNReLU(in_channels=64, out_channels=64),
            nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        )
        self.encode2 = nn.Sequential(
            ConvBNReLU(in_channels=64, out_channels=128),
            ConvBNReLU(in_channels=128, out_channels=128),
            nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        )
        self.encode3 = nn.Sequential(
            ConvBNReLU(in_channels=128, out_channels=256),
            ConvBNReLU(in_channels=256, out_channels=256),
            ConvBNReLU(in_channels=256, out_channels=256),
            nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        )
        self.encode4 = nn.Sequential(
            ConvBNReLU(in_channels=256, out_channels=512),
            ConvBNReLU(in_channels=512, out_channels=512),
            ConvBNReLU(in_channels=512, out_channels=512),
            nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        )
        self.encode5 = nn.Sequential(
            ConvBNReLU(in_channels=512, out_channels=512),
            ConvBNReLU(in_channels=512, out_channels=512),
            ConvBNReLU(in_channels=512, out_channels=512),
            nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        )
        self.decode5 = nn.Sequential(
            ConvBNReLU(in_channels=512, out_channels=512),
            ConvBNReLU(in_channels=512, out_channels=512),
            ConvBNReLU(in_channels=512, out_channels=512),
        )
        self.decode4 = nn.Sequential(
            ConvBNReLU(in_channels=512, out_channels=512),
            ConvBNReLU(in_channels=512, out_channels=512),
            ConvBNReLU(in_channels=512, out_channels=256),
        )
        self.decode3 = nn.Sequential(
            ConvBNReLU(in_channels=256, out_channels=256),
            ConvBNReLU(in_channels=256, out_channels=256),
            ConvBNReLU(in_channels=256, out_channels=128),
        )
        self.decode2 = nn.Sequential(
            ConvBNReLU(in_channels=128, out_channels=128),
            ConvBNReLU(in_channels=128, out_channels=64),
        )
        self.decode1 = nn.Sequential(
            ConvBNReLU(in_channels=64, out_channels=64),
            ConvBNReLU(in_channels=64, out_channels=num_classes),
        )
        self.up = nn.MaxUnpool2d(kernel_size=2, stride=2)
        # self.seghead = nn.Softmax(dim=1)

    @staticmethod
    def preprocess_input(x):
        return F.sigmoid(x)

    def forward(self, x):
        if not self.preprocess_flag:
            x = self.preprocess_input(x)
        x = x.float()
        x, x_encode1_indices = self.encode1(x)
        print('# encode1 output shape:', x.shape)
        x, x_encode2_indices = self.encode2(x)
        print('# encode2 output shape:', x.shape)
        x, x_encode3_indices = self.encode3(x)
        print('# encode3 output shape:', x.shape)
        x, x_encode4_indices = self.encode4(x)
        print('# encode4 output shape:', x.shape)
        x, x_encode5_indices = self.encode5(x)
        print('# encode5 output shape:', x.shape)
        x = self.decode5(self.up(x, x_encode5_indices))
        print('# decode5 output shape:', x.shape)
        x = self.decode4(self.up(x, x_encode4_indices))
        print('# decode4 output shape:', x.shape)
        x = self.decode3(self.up(x, x_encode3_indices))
        print('# decode3 output shape:', x.shape)
        x = self.decode2(self.up(x, x_encode2_indices))
        print('# decode2 output shape:', x.shape)
        x = self.decode1(self.up(x, x_encode1_indices))
        print('# decode1 output shape:', x.shape)
        # x = self.seghead(x)
        print('# output shape:', x.shape)
        return x


if __name__ == '__main__':
    # model = SegNet(1,7,True).cuda(0)
    # print(sum(p.numel() for p in model.parameters()))
    #
    # model(torch.randn(1,1,512,512))
    net = SegNet(1, 8, False)
    stat(net, input_size=(1, 512, 512))
