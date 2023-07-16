import torch
import torch.nn as nn
from einops import rearrange
from torchstat import stat
from backbone.utils4SemHybridNet.vit import ViT

'''
# 普通的双卷积层
class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ConvBlock, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        # print('ConvBlock: \n', torch.cuda.memory_summary())
        return x
'''
'''
# 全局特征提取，现用ViT代替
class GlobalFeature(nn.Module):
    def __init__(self, channel, mid_channel):
        super(GlobalFeature, self).__init__()
        self.vit = ()

    def forward(self, x):
        x = self.vit(x)
        # print('gf: ', torch.cuda.memory_summary())
        return x
'''
'''
class FeatureFusion(nn.Module):
    def __init__(self, channel, mid_channel):
        super(FeatureFusion, self).__init__()
        self.local_f = LocalFeature(channel=channel)
        self.global_f = GlobalFeature(channel=channel, mid_channel=mid_channel)
        self.att = AttentionBlock(channel, channel, channel)

    def forward(self, x):
        skip = x.clone()
        l_f = self.local_f(x)
        g_f = self.global_f(x)
        att_f = self.att(l_f, g_f)
        out = skip + att_f
        # print('ff: ', torch.cuda.memory_summary())
        return out
'''
'''
class FeatureFusion(nn.Module):
    def __init__(self, in_channel,base_channel=128):
        super(FeatureFusion, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channel, in_channel//2, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channel//2),
            nn.PReLU(),
            nn.Conv2d(in_channel//2, in_channel//2, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channel//2),
            nn.PReLU(),
            nn.Conv2d(in_channel//2, in_channel, kernel_size=1),
            nn.BatchNorm2d(in_channel),
            nn.Softmax2d()
        )

    def forward(self, gf_x, lf_x):
        x = gf_x + lf_x
        att_x = self.layers(x)
        out = x * att_x
        return out
'''
'''
# Attention gate代码
class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlock, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g = self.W_g(g)
        x = self.W_x(x)
        psi = self.relu(g + x)
        psi = self.psi(psi)
        # print('at: ', torch.cuda.memory_summary())
        return x * psi
'''
'''
# 局部特征提取，卷积堆叠
class LocalFeature(nn.Module):
    def __init__(self, channel):
        super(LocalFeature, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.layer1(x)
        # print('lf: ', torch.cuda.memory_summary())
        return x
'''
'''
class LocalFeature(nn.Module):  # 单通道注意力
    def __init__(self, in_channel, reduction_ratio=16):
        super(LocalFeature, self).__init__()
        self.layers = DDCBlock(in_channel, in_channel, inplace=False)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channel, in_channel // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channel // reduction_ratio, in_channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.layers(x)
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
'''


class DepthWiseConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernerl_size=3):
        super(DepthWiseConv, self).__init__()

        # 逐通道卷积
        self.depth_conv = nn.Conv2d(in_channels=in_channel,
                                    out_channels=in_channel,
                                    kernel_size=kernerl_size,
                                    stride=1,
                                    padding=1,
                                    groups=in_channel)
        # groups是一个数，当groups=in_channel时,表示做逐通道卷积

        # 逐点卷积
        self.point_conv = nn.Conv2d(in_channels=in_channel,
                                    out_channels=out_channel,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0,
                                    groups=1)

    def forward(self, x):
        out = self.depth_conv(x)
        out = self.point_conv(out)
        return out


class DDCBlock(nn.Module):
    def __init__(self, in_channel, out_channel, inplace=True):
        super(DDCBlock, self).__init__()
        self.layers = nn.Sequential(
            DepthWiseConv(in_channel, out_channel),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=inplace),
            DepthWiseConv(out_channel, out_channel),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=inplace)
        )

    def forward(self, x):
        out = self.layers(x)
        return out


class TDCBlock(nn.Module):
    def __init__(self, in_channel, out_channel, mid_channel=32):
        super(TDCBlock, self).__init__()
        self.mid_channel = mid_channel
        self.layers = nn.Sequential(
            DepthWiseConv(in_channel, mid_channel),
            nn.BatchNorm2d(mid_channel),
            nn.ReLU(inplace=True),
            DepthWiseConv(mid_channel, mid_channel),
            nn.BatchNorm2d(mid_channel),
            nn.ReLU(inplace=True),
            DepthWiseConv(mid_channel, out_channel),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        out = self.layers(x)
        return out


class LocalFeature(nn.Module):  # Multi_branch Bar_shape conv attn
    def __init__(self, in_channel, reduction_ratio=16):
        super(LocalFeature, self).__init__()
        mid_channel = in_channel // reduction_ratio
        self.Conv11_1 = nn.Conv2d(in_channel, mid_channel, kernel_size=1)
        self.Conv13 = nn.Conv2d(mid_channel, mid_channel, (1, 3), padding=(0, 1))
        self.Conv31 = nn.Conv2d(mid_channel, mid_channel, (3, 1), padding=(1, 0))
        self.Conv15 = nn.Conv2d(mid_channel, mid_channel, (1, 5), padding=(0, 2))
        self.Conv51 = nn.Conv2d(mid_channel, mid_channel, (5, 1), padding=(2, 0))
        self.Conv17 = nn.Conv2d(mid_channel, mid_channel, (1, 7), padding=(0, 3))
        self.Conv71 = nn.Conv2d(mid_channel, mid_channel, (7, 1), padding=(3, 0))

        self.Conv11_2 = nn.Conv2d(in_channel // reduction_ratio, in_channel, kernel_size=1)

    def forward(self, x):
        skip = x.clone()
        branch_0 = self.Conv11_1(x)

        branch_1 = self.Conv13(branch_0)
        branch_1 = self.Conv31(branch_1)

        branch_2 = self.Conv15(branch_0)
        branch_2 = self.Conv51(branch_2)

        branch_3 = self.Conv17(branch_0)
        branch_3 = self.Conv71(branch_3)

        mix = branch_0 + branch_1 + branch_2 + branch_3
        bar_attn = self.Conv11_2(mix)

        out = skip * bar_attn
        return out


class ChannelAttention(nn.Module):
    def __init__(self, in_channel, reduction_ratio=16):  # in_planes是输入通道数
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_channel, in_channel // reduction_ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_channel // reduction_ratio, in_channel, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = self.sigmoid(avg_out + max_out)
        return out


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.kernel_size = kernel_size
        if self.kernel_size == 3:
            padding_size = 1
        elif self.kernel_size == 7:
            padding_size = 3
        else:
            raise AssertionError("404 Not Found.")
        self.conv1 = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding_size, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        print('avg_out: ', avg_out.shape)
        print('max_out: ', max_out.shape)
        x = torch.cat([avg_out, max_out], dim=1)
        print(x.shape)
        x = self.conv1(x)
        out = self.sigmoid(x)
        return out


class FeatureFusion(nn.Module):
    def __init__(self, in_channel, ):
        super(FeatureFusion, self).__init__()
        self.fc_1 = nn.Conv2d(in_channel * 2, in_channel, 1, bias=False)
        self.bn = nn.BatchNorm2d(in_channel)
        self.sa = SpatialAttention()
        self.ca = ChannelAttention(in_channel)

    def forward(self, gf_x, lf_x):
        # print('gf_x: ', gf_x.shape)
        # print('lf_x: ', lf_x.shape)
        att_gf = self.sa(gf_x)
        x = torch.cat([gf_x * att_gf, lf_x], dim=1)
        # print('cat x: ', x.shape)
        x = self.fc_1(x)
        x = self.bn(x)
        # print('fc x: ', x.shape)

        att_x = self.ca(x)
        out = x * att_x
        return out


class DownSample(nn.Module):
    def __init__(self, args_type='MaxPooling', channel=None):
        super(DownSample, self).__init__()
        if args_type == 'MaxPooling':
            # no return indices
            self.down_sample = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=False)
        elif args_type == 'Conv' and channel is not None:
            self.down_sample = nn.Conv2d(channel, channel, kernel_size=3, stride=2, padding=1)
        else:
            raise AssertionError("Please specify the correct Down_Sample type.")

    def forward(self, x):
        return self.down_sample(x)


class UpSample(nn.Module):
    def __init__(self, args_type='interpolate', channel=None):
        super(UpSample, self).__init__()
        self.args_type = args_type
        if self.args_type == 'ConvTranspose' and channel is not None:
            self.up_sample = nn.ConvTranspose2d(in_channels=channel, out_channels=channel, kernel_size=4, stride=2,
                                                padding=1)
        elif self.args_type == 'interpolate':
            interpolate_mode = 'bilinear'  # nearest
            self.up_sample = nn.Upsample(scale_factor=2, mode=interpolate_mode)
        elif self.args_type == 'MaxUnpooling':
            self.up_sample = nn.MaxUnpool2d(kernel_size=2, stride=2)
            # self.up_sample(x, MaxPool2d_indices)
        else:
            raise AssertionError("Please specify the correct Up_Sample type.")

    def forward(self, x):
        x = self.up_sample(x)
        return x


class EncoderBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, base_width=64, inplace=True):
        super().__init__()
        self.down = DownSample(args_type='Conv', channel=in_channels)
        width = int(out_channels * (base_width / 64))
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, width, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=inplace),
            # DepthWiseConv(width, width),
            # nn.BatchNorm2d(width),
            DDCBlock(width, width, inplace),
            nn.Conv2d(width, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=inplace)
        )

    def forward(self, x):
        down_x = self.down(x)
        out = self.layers(down_x)
        return out


class VitDown(nn.Module):
    # 下采样后调整通道数作为ViT的输入
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.down = DownSample(args_type='Conv', channel=in_channels)
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            # DepthWiseConv(in_channels, out_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        down_x = self.down(x)
        out = self.layers(down_x)
        return out


class Encoder(nn.Module):
    def __init__(self, img_dim, in_channels, out_channels, head_num, mlp_dim, block_num, patch_dim):
        super().__init__()
        # self.conv_term_1 = DDCBlock(in_channels, out_channels // 2)
        self.conv_term_1 = TDCBlock(in_channels, out_channels // 2)

        self.conv_enc_1 = EncoderBottleneck(out_channels // 2, out_channels)  # 256
        self.conv_enc_2 = EncoderBottleneck(out_channels, out_channels * 2, inplace=False)  # 128

        # self.conv_vit_down_1 = EncoderBottleneck(out_channels * 2, out_channels * 4)
        # self.conv_vit_down_2 = EncoderBottleneck(out_channels * 4, out_channels * 8)
        self.conv_vit_down_1 = VitDown(out_channels * 2, out_channels * 4)  # 64
        self.conv_vit_down_2 = VitDown(out_channels * 4, out_channels * 8)  # 32
        # undone
        # conv_vit_down作为局部特征，与vit得到的全局特征融合

        self.vit_img_dim_1 = img_dim // patch_dim * 2
        self.vit_enc_1 = ViT(self.vit_img_dim_1, out_channels * 4, out_channels * 4,
                             head_num, mlp_dim, block_num, patch_dim=1, classification=False)

        self.replace = nn.Conv2d(out_channels * 8, out_channels * 8, 7, padding=3, bias=False)

        # self.pam_1 = PAM_Module(out_channels * 4)
        # self.sdpa_1 = ScaledDotProductAttention(out_channels * 4)

        self.local_enc_1 = LocalFeature(out_channels * 4)
        self.ff1 = FeatureFusion(out_channels * 4)

        # self.vit_img_dim_2 = img_dim // patch_dim
        # self.vit_enc_2 = ViT(self.vit_img_dim_2, out_channels * 8, out_channels * 8,
        #                      head_num, mlp_dim, block_num, patch_dim=1, classification=False)

        # self.pam_2 = PAM_Module(out_channels * 8)
        # self.sdpa_2 = ScaledDotProductAttention(out_channels * 8)

        self.local_enc_2 = LocalFeature(out_channels * 8)
        self.ff2 = FeatureFusion(out_channels * 8)

        self.conv_term_2 = DDCBlock(out_channels * 8, out_channels * 4)

    def forward(self, x):
        x = self.conv_term_1(x)  # hw
        x1 = self.conv_enc_1(x)  # h/2,w/2

        x2 = self.conv_enc_2(x1)  # h/4,w/4

        x3 = self.conv_vit_down_1(x2)  # h/8,w/8
        print(x3.shape)
        x3_l = self.local_enc_1(x3)
        x3_t = self.vit_enc_1(x3)
        # x3_g = self.local_enc_1(x3)
        # x3_t_1 = self.pam_1(x3)
        # x3_t_2 = self.sdpa_1(x3)
        # x3_g = x3_t_1 + x3_t_2
        x3_g = rearrange(x3_t, "b (x y) c -> b c x y", x=self.vit_img_dim_1, y=self.vit_img_dim_1)

        x3 = self.ff1(x3_g, x3_l)

        x4 = self.conv_vit_down_2(x3)  # h/16,w/16

        x4_l = self.local_enc_2(x4)
        # x4_t = self.vit_enc_2(x4)
        # x4_g = self.local_enc_2(x4)
        # x4_t_1 = self.pam_2(x4)
        # x4_t_2 = self.sdpa_2(x4)
        # x4_g = x4_t_1 + x4_t_2
        # x4_g = rearrange(x4_t, "b (x y) c -> b c x y", x=self.vit_img_dim_2, y=self.vit_img_dim_2)
        x4_g = self.replace(x4)
        x4 = self.ff2(x4_g, x4_l)

        x4 = self.conv_term_2(x4)
        # print('x1.shape', x1.shape)
        # print('x2.shape', x2.shape)
        # print('x3.shape', x3.shape)
        # print('x4.shape', x4.shape)
        return x1, x2, x3, x4


class DecoderBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = UpSample(args_type='ConvTranspose', channel=in_channels // 2)
        self.up_2 = UpSample(args_type='ConvTranspose', channel=64)
        # self.up = UpSample()
        self.layers = DDCBlock(in_channels, out_channels)

    def forward(self, x, x_concat=None):
        # print(x.shape)
        if x_concat is not None:
            up_x = self.up(x)
            up_x = torch.cat([up_x, x_concat], dim=1)
        else:
            up_x = self.up_2(x)
            # up_x = self.up(x)
        out = self.layers(up_x)
        return out


class Decoder(nn.Module):
    def __init__(self, out_channels, class_num):
        super().__init__()
        # undone
        # 解码时不需要每层都skip connection，开销过大
        self.skip_dec_1 = DecoderBottleneck(out_channels * 8, out_channels * 2)
        self.skip_dec_2 = DecoderBottleneck(out_channels * 4, out_channels)
        self.skip_dec_3 = DecoderBottleneck(out_channels * 2, out_channels // 2)
        self.conv_dec_4 = DecoderBottleneck(out_channels // 2, out_channels // 4)
        self.out_layer = nn.Sequential(
            # nn.Conv2d(out_channels // 4, out_channels // 8, 3),
            DepthWiseConv(out_channels // 4, out_channels // 8),
            nn.BatchNorm2d(out_channels // 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 8, class_num, 1)
        )

    def forward(self, x1, x2, x3, x4):
        print(x4.shape)
        print(x3.shape)
        print(x2.shape)
        print(x1.shape)
        x = self.skip_dec_1(x4, x3)
        x = self.skip_dec_2(x, x2)
        x = self.skip_dec_3(x, x1)
        x = self.conv_dec_4(x)
        out = self.out_layer(x)
        return out


class SemHybridNet(nn.Module):
    def __init__(self, img_dim, in_channel, out_channel, num_classes, head_num, mlp_dim, block_num, patch_dim,
                 preprocess_flag):
        super(SemHybridNet, self).__init__()
        self.preprocess_flag = preprocess_flag
        self.encoder = Encoder(img_dim, in_channel, out_channel, head_num, mlp_dim, block_num, patch_dim)
        self.decoder = Decoder(out_channel, num_classes)

    def forward(self, x):
        x = x.float()
        x1, x2, x3, x4 = self.encoder(x)
        x = self.decoder(x1, x2, x3, x4)
        return x


if __name__ == '__main__':
    net = SemHybridNet(img_dim=512,
                in_channel=1,
                out_channel=128,
                num_classes=8,
                head_num=8,
                mlp_dim=512,
                block_num=1,
                patch_dim=16,
                preprocess_flag=None)
    # weights_init(net)
    # print(sum(p.numel() for p in net.parameters()))
    # print(net(torch.randn(1, 1, 512, 512)).shape)
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # net.to(device)
    stat(net, input_size=(1, 512, 512))
