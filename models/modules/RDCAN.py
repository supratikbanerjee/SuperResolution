import torch
import torch.nn as nn
from .blocks import ConvBlock, DeconvBlock, MeanShift, get_valid_padding
from .cnn_models import FSRCNN
from models import pac


class CALayer(nn.Module):
    def __init__(self, channel, reduction=8):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class UpBlock(nn.Module):
    def __init__(self, idx, num_features, kernel_size, stride, padding):
        super(UpBlock, self).__init__()
        #print(num_features*(idx+1), num_features, idx)
        self.upCompressIn = nn.Conv2d(num_features*(idx+1), num_features, kernel_size=1, stride=1, 
                    padding=get_valid_padding(1, 1))
        self.act1 = nn.PReLU(num_parameters=1, init=0.2)
        self.upConv = nn.ConvTranspose2d(num_features, num_features, kernel_size, stride=stride, padding=padding)
        self.act2 = nn.PReLU(num_parameters=1, init=0.2)

    def forward(self, x):
        x = self.act1(self.upCompressIn(x))
        x = self.act2(self.upConv(x))
        return x


class DownBlock(nn.Module):
    def __init__(self, idx, num_features, kernel_size, stride, padding):
        super(DownBlock, self).__init__()
        self.downCompressIn = pac.PacConv2d(num_features*(idx+1), num_features, kernel_size=1, stride=1, 
                    padding=get_valid_padding(1, 1))
        self.act1 = nn.PReLU(num_parameters=1, init=0.2)
        self.downConv = nn.Conv2d(num_features, num_features, kernel_size, stride=stride, padding=padding)
        self.act2 = nn.PReLU(num_parameters=1, init=0.2)

    def forward(self, x):
        guide = x[1]
        x = x[0]
        x = self.act1(self.downCompressIn(x, guide))
        x = self.act2(self.downConv(x))
        return x


class FeedbackBlock(nn.Module):
    def __init__(self, num_features, num_groups, upscale_factor, stride, kernel_size, reduction=8):
        super(FeedbackBlock, self).__init__()
        kernel_size = 6
        padding = 2
        self.num_groups = num_groups

        self.upBlocks = nn.ModuleList()
        self.downBlocks = nn.ModuleList()

        for idx in range(self.num_groups):
            self.upBlocks.append(UpBlock(idx, num_features, kernel_size, stride, padding=padding))
            self.downBlocks.append(DownBlock(idx, num_features, kernel_size, stride, padding=padding))

        self.compress_out = nn.Conv2d((num_groups)*num_features, num_features, kernel_size=1, 
            padding=get_valid_padding(1, 1))
        self.prelu2 = nn.PReLU(num_parameters=1, init=0.2)

        self.CALayer = CALayer(num_features, reduction)

    def forward(self, x):
        guide = x[1]
        x = x[0]
        lr_features = []

        LD_L = torch.tensor([]).cuda()
        LD_H = torch.tensor([]).cuda()
        LD_L = torch.cat((LD_L, x), 1)
        prev_LD = x

        for idx in range(self.num_groups):
            LD_H_o = self.upBlocks[idx](LD_L)
            LD_H = torch.cat((LD_H, LD_H_o), 1)

            LD_L_o = self.downBlocks[idx]((LD_H, guide))
            LD_L = torch.cat((LD_L, LD_L_o), 1)

            lr_features.append(LD_L_o) #+prev_LD)
            #prev_LD = LD_L_o

        output = torch.cat(tuple(lr_features), 1)
        output = self.compress_out(output)
        output = self.prelu2(output)
        output = self.CALayer(output)

        return output


class RDCAN(nn.Module):
    def __init__(self, in_channels, out_channels, num_features, num_steps, num_groups, upscale_factor, act_type = 'prelu', norm_type = None, reduction=8):
        super(RDCAN, self).__init__()

        if upscale_factor == 2:
            stride = 2
            kernel_size = 6
        elif upscale_factor == 3:
            stride = 3
            kernel_size = 7
        elif upscale_factor == 4:
            stride = 4
            kernel_size = 8
        elif upscale_factor == 8:
            stride = 8
            kernel_size = 12


        self.num_steps = num_steps
        self.num_features = num_features
        self.upscale_factor = upscale_factor

        self.upscaleNet = FSRCNN(upscale_factor)
        self.load()
        
        
        # LR feature extraction block
        self.conv_in = nn.Conv2d(in_channels, 4*num_features, kernel_size=3, padding=get_valid_padding(3, 1))
        self.prelu1 = nn.PReLU(num_parameters=1, init=0.2)
        self.feat_in = nn.Conv2d(4*num_features, num_features, kernel_size=1, padding=get_valid_padding(1, 1))
        self.prelu2 = nn.PReLU(num_parameters=1, init=0.2)

        # basic block
        self.block = FeedbackBlock(num_features, num_groups, upscale_factor, stride, kernel_size, reduction)

        self.out = nn.ConvTranspose2d(num_features, num_features, kernel_size=kernel_size, stride= stride, padding=2)
        self.prelu3 = nn.PReLU(num_parameters=1, init=0.2)

        self.conv_out = nn.Conv2d(num_features, out_channels, kernel_size=3, padding=get_valid_padding(3, 1))
        self.prelu4 = nn.PReLU(num_parameters=1, init=0.2)


        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find('Conv2d') != -1:
                torch.nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif classname.find('ConvTranspose2d') != -1:
                torch.nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

        

    def forward(self, x):

        guide = self.upscaleNet(x)
        inter_res = nn.functional.interpolate(x, scale_factor=self.upscale_factor, mode='bilinear', align_corners=False)

        x = self.prelu1(self.conv_in(x))
        x = self.prelu2(self.feat_in(x))        

        h = self.block((x, guide))
        
        h = self.prelu3(self.out(h))
        h = torch.add(inter_res, self.prelu4(self.conv_out(h)))
        return h


    def load(self):
        checkpoint = torch.load('trained_models/1000_FSRCNN_x2_netG.pth')
        self.upscaleNet.load_state_dict(checkpoint['state_dict'])
 