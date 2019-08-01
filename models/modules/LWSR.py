import torch.nn as nn
import torch.nn.functional as F
from models import pac
import torch

class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_down = nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.conv_up = nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True)
        self.sig = nn.Sigmoid()
        

    def forward(self, x):
        #print(x.shape)
        y = self.avg_pool(x)
        y = self.conv_down(y)
        y = self.relu(y)
        y = self.conv_up(y)
        y = self.sig(y)
        return x * y

class LWSR(nn.Module):
    def __init__(self, upscale_factor, feature):
        super(LWSR, self).__init__()

        self.upscale_factor = upscale_factor
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=4*feature, kernel_size=3, stride=1, padding=1)
        self.prelu1 = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(in_channels=4*feature, out_channels=feature, kernel_size=1, stride=1, padding=0)
        self.prelu2 = nn.LeakyReLU(0.2, inplace=True)

        #self.upConv = nn.ConvTranspose2d(feature, feature, kernel_size=2, stride=2, padding=2)
        #self.act1 = nn.PReLU(num_parameters=1, init=0.2)
        #self.downCompressIn = pac.PacConv2d(feature, feature, kernel_size=1, stride=1,padding=0)
        #self.act2 = nn.PReLU(num_parameters=1, init=0.2)
        #self.downConv = nn.Conv2d(feature, feature, kernel_size=2, stride=2, padding=2)
        #self.act3 = nn.PReLU(num_parameters=1, init=0.2)

        #self.se = CALayer(feature)
        self.deconv1 = nn.ConvTranspose2d(in_channels=feature, out_channels=feature, kernel_size=6, stride=upscale_factor, padding=2)
        self.prelu3 = nn.LeakyReLU(0.2, inplace=True)
        self.conv4 = nn.Conv2d(in_channels=feature, out_channels=3, kernel_size=1, stride=1, padding=0)
        self.prelu4 = nn.LeakyReLU(0.2, inplace=True)
        self._initialize_weights()

    def forward(self, x):
        #guide = self.upscaleNet(x)
        upres = nn.functional.interpolate(x, scale_factor=self.upscale_factor, mode='bilinear', align_corners=False)
        x = self.conv1(x)
        x = self.prelu1(x)
        x = self.conv2(x)
        x = self.prelu2(x)

        #x = self.upConv(x)
        #x = self.act1(x)
        #x = self.downCompressIn(x, x)
        #x = self.act2(x)
        #x = self.downConv(x)
        #x = self.act3(x)
        #x = self.se(x)

        x = self.deconv1(x)
        x = self.prelu3(x)
        x = self.conv4(x)
        x = self.prelu4(x)
        x = torch.add(upres, x)
        return x

    def _initialize_weights(self):
        print("===> Initializing weights")
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