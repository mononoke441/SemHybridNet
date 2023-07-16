import torch
import torch.nn as nn
import torch.nn.functional as F
from torchstat import stat


class conv_block(nn.Module):
    def __init__(self, in_c, out_c):
        super(conv_block, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=(3, 3), stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            # nn.Dropout(0.3),
            nn.ReLU(inplace=True),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(out_c, out_c, kernel_size=(3, 3), stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            # nn.Dropout(0.3),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x


class Downsample(nn.Module):
    def __init__(self, channel):
        super(Downsample, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=(3, 3), stride=2, padding=1, bias=False),
            nn.BatchNorm2d(channel),
            nn.ReLU()
        )

    def forward(self, x):
        return self.layer(x)


class Upsample(nn.Module):
    def __init__(self, channel):
        super(Upsample, self).__init__()
        self.upsample = nn.ConvTranspose2d(in_channels=channel, out_channels=channel, kernel_size=4, stride=2,
                                           padding=1)
        self.conv1 = nn.Conv2d(channel, channel // 2, kernel_size=(1, 1), stride=1)

    def forward(self, x, featuremap):
        # x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.upsample(x)
        x = self.conv1(x)
        x = torch.cat((x, featuremap), dim=1)
        return x


class Unet(nn.Module):
    def __init__(self, in_channel, out_channel, classes_num):
        super(Unet, self).__init__()
        self.layer1 = conv_block(in_channel, out_channel)
        self.layer2 = Downsample(out_channel)
        self.layer3 = conv_block(out_channel, out_channel * 2)
        self.layer4 = Downsample(out_channel * 2)
        self.layer5 = conv_block(out_channel * 2, out_channel * 4)
        self.layer6 = Downsample(out_channel * 4)
        self.layer7 = conv_block(out_channel * 4, out_channel * 8)
        self.layer8 = Downsample(out_channel * 8)
        self.layer9 = conv_block(out_channel * 8, out_channel * 16)
        self.layer10 = Upsample(out_channel * 16)
        self.layer11 = conv_block(out_channel * 16, out_channel * 8)
        self.layer12 = Upsample(out_channel * 8)
        self.layer13 = conv_block(out_channel * 8, out_channel * 4)
        self.layer14 = Upsample(out_channel * 4)
        self.layer15 = conv_block(out_channel * 4, out_channel * 2)
        self.layer16 = Upsample(out_channel * 2)
        self.layer17 = conv_block(out_channel * 2, out_channel)
        self.final = nn.Conv2d(out_channel, classes_num, 1)

    def forward(self, x):
        x = self.layer1(x)
        f1 = x
        x = self.layer2(x)
        x = self.layer3(x)
        f2 = x
        x = self.layer4(x)
        x = self.layer5(x)
        f3 = x
        x = self.layer6(x)
        x = self.layer7(x)
        f4 = x
        x = self.layer8(x)
        x = self.layer9(x)
        x = self.layer10(x, f4)
        x = self.layer11(x)
        x = self.layer12(x, f3)
        x = self.layer13(x)
        x = self.layer14(x, f2)
        x = self.layer15(x)
        x = self.layer16(x, f1)
        x = self.layer17(x)
        out = self.final(x)
        # print("model_out:",out.shape)
        return out


if __name__ == '__main__':
    net = Unet(1, 64, 8)
    # weights_init(net)
    # print(sum(p.numel() for p in net.parameters()))
    # print(net(torch.randn(1, 1, 512, 512)).shape)
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # net.to(device)
    stat(net, input_size=(1, 512, 512))
