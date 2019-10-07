import torch.nn as nn
import torch.nn.functional as F
from models import pac
from .blocks import MeanShift
import torch
class SRCNN(nn.Module):
    def __init__(self, upscale_factor):
        super(SRCNN, self).__init__()
        self.upscale_factor = upscale_factor
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=128, kernel_size=9, stride=1, padding=4)
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=5, stride=1, padding=2)

        self._initialize_weights()

    def forward(self, x):
        x = F.interpolate(x , scale_factor=self.upscale_factor, mode='bicubic')
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        return x

    def _initialize_weights(self):
        print("===> Initializing weights")
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)
        nn.init.xavier_uniform_(self.conv3.weight)

class FSRCNN(nn.Module):
    def __init__(self, upscale_factor):
        super(FSRCNN, self).__init__()

        self.upscale_factor = upscale_factor
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=56, kernel_size=5, stride=1, padding=2)
        self.prelu1 = nn.PReLU(num_parameters=1, init=0.2)
        self.conv2 = nn.Conv2d(in_channels=56, out_channels=12, kernel_size=1, stride=1, padding=0)
        self.prelu2 = nn.PReLU(num_parameters=1, init=0.2)
        self.conv31 = nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3, stride=1, padding=1)
        self.conv32 = nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3, stride=1, padding=1)
        self.conv33 = nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3, stride=1, padding=1)
        self.conv34 = nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3, stride=1, padding=1)
        self.prelu3 = nn.PReLU(num_parameters=1, init=0.2)
        self.conv4 = nn.Conv2d(in_channels=12, out_channels=56, kernel_size=1, stride=1, padding=0)
        self.prelu4 = nn.PReLU(num_parameters=1, init=0.2)
        self.deconv1 = nn.ConvTranspose2d(in_channels=56, out_channels=3, kernel_size=8, stride=upscale_factor, padding=2)

        #self._initialize_weights()

    def forward(self, x):
        x = self.conv1(x)
        x = self.prelu1(x)
        x = self.conv2(x)
        x = self.prelu2(x)
        x = self.conv31(x)
        x = self.conv32(x)
        x = self.conv33(x)
        x = self.conv34(x)
        x = self.prelu3(x)
        x = self.conv4(x)
        x = self.prelu4(x)
        x = self.deconv1(x)
        return x

    def _initialize_weights(self):
        print("===> Initializing weights")
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)
        nn.init.xavier_uniform_(self.conv31.weight)
        nn.init.xavier_uniform_(self.conv32.weight)
        nn.init.xavier_uniform_(self.conv33.weight)
        nn.init.xavier_uniform_(self.conv34.weight)
        nn.init.xavier_uniform_(self.conv4.weight)
        nn.init.orthogonal_(self.deconv1.weight, nn.init.calculate_gain('relu'))