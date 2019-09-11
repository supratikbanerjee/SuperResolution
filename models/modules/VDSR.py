#https://github.com/thstkdgus35/EDSR-PyTorch/blob/master/src/model/vdsr.py

from models.modules import common

import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F


class VDSR(nn.Module):
    def __init__(self, conv=common.default_conv):
        super(VDSR, self).__init__()

        n_resblocks = 18
        n_feats = 64
        kernel_size = 3 
        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = common.MeanShift(255, rgb_mean, rgb_std)
        self.add_mean = common.MeanShift(255, rgb_mean, rgb_std, 1)

        def basic_block(in_channels, out_channels, act):
            return common.BasicBlock(
                conv, in_channels, out_channels, kernel_size,
                bias=True, bn=False, act=act
            )

        # define body module
        m_body = []
        m_body.append(basic_block(3, n_feats, nn.ReLU(True)))
        for _ in range(n_resblocks - 2):
            m_body.append(basic_block(n_feats, n_feats, nn.ReLU(True)))
        m_body.append(basic_block(n_feats, 3, None))

        self.body = nn.Sequential(*m_body)

    def forward(self, x):
        x = self.sub_mean(x)
        x = F.interpolate(x , scale_factor=2, mode='bicubic')
        res = self.body(x)
        res += x
        x = self.add_mean(res)

        return x 
