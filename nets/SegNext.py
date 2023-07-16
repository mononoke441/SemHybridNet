# %%
from pathlib import Path

p = Path(__file__)
yaml_name = 'config.yaml'
yaml_path = p.parent.parent.joinpath('backbone/utils4segnext', yaml_name)
# print(yaml_path)
import yaml, math

with open(yaml_path) as fh:
    config = yaml.load(fh, Loader=yaml.FullLoader)
import torch.nn.functional as F
import torch.nn as nn
import torch
from backbone.utils4segnext.essentials import MSCANet
from backbone.utils4segnext.decoder import HamDecoder


class SegNext(nn.Module):
    def __init__(self, num_classes, preprocess_flag, in_channel=3, embed_dims=[32, 64, 460, 256],
                 ffn_ratios=[4, 4, 4, 4], depths=[3, 3, 5, 2], num_stages=4,
                 dec_outChannels=256, config=config, dropout=0.0, drop_path=0.0):
        super().__init__()
        self.preprocess_flag = preprocess_flag
        self.cls_conv = nn.Sequential(nn.Dropout2d(p=0.1),
                                      nn.Conv2d(dec_outChannels, num_classes, kernel_size=1))
        self.encoder = MSCANet(in_channnels=in_channel, embed_dims=embed_dims,
                               ffn_ratios=ffn_ratios, depths=depths, num_stages=num_stages, drop_path=drop_path)
        # self.encoder = MSCANet(in_channnels=in_channel, embed_dims=embed_dims,
        #                        ffn_ratios=ffn_ratios, depths=depths, num_stages=num_stages,
        #                        dropout=dropout, drop_path=drop_path)
        self.decoder = HamDecoder(
            outChannels=dec_outChannels, config=config, enc_embed_dims=embed_dims)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if isinstance(m, nn.LayerNorm) or isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, val=1.0)
                nn.init.constant_(m.bias, val=0.0)
            if isinstance(m, nn.Conv2d):
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                fan_out //= m.groups
                nn.init.normal_(m.weight, std=math.sqrt(2.0 / fan_out), mean=0)

    @staticmethod
    def preprocess_input(x):
        return F.sigmoid(x)

    def forward(self, x):
        if not self.preprocess_flag:
            x = self.preprocess_input(x)
        x = x.float()
        enc_feats = self.encoder(x)
        dec_out = self.decoder(enc_feats)
        output = self.cls_conv(dec_out)  # here output will be B x C x H/8 x W/8
        output = F.interpolate(output, size=x.size()[-2:], mode='bilinear', align_corners=True)  # now its same as input
        #  bilinear interpol was used originally
        return output


if __name__ == '__main__':
    from torchstat import stat
    net = SegNext(8, False, in_channel=1).cuda(0)
    print(next(net.parameters()).device)
    stat(net, input_size=(1, 512, 512))
    # print(sum(p.numel() for p in net.parameters()))
    # print(net(torch.rand(1, 1, 512, 512).cuda(0)).shape)
