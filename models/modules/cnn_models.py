import torch.nn as nn
import torch.nn.functional as F
from models import pac
from .blocks import MeanShift
import torch
class SRCNN(nn.Module):
    def __init__(self, upscale_factor):
        super(SRCNN, self).__init__()
        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = MeanShift(rgb_mean, rgb_std)

        self.upscale_factor = upscale_factor
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=9, stride=1, padding=4)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=3, kernel_size=5, stride=1, padding=2)
        self.add_mean = MeanShift(rgb_mean, rgb_std, 1)
        #self.deconv1 = nn.ConvTranspose2d(in_channels=12, out_channels=3, kernel_size=8, stride=upscale_factor, padding=upscale_factor)
        #self.upscaleNet = FSRCNN(upscale_factor)
        #self.load()

        self._initialize_weights()

    def forward(self, x):
        x = self.sub_mean(x)
        #guide = self.upscaleNet(x)
        x = F.interpolate(x , scale_factor=self.upscale_factor, mode='bicubic')
        #print(x.shape, guide.shape)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = self.add_mean(x)
        return x

    def _initialize_weights(self):
        print("===> Initializing weights")
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)
        nn.init.xavier_uniform_(self.conv3.weight)
        #nn.init.orthogonal_(self.deconv1.weight, nn.init.calculate_gain('relu'))
        #nn.init.orthogonal_(self.deconv1.weight, nn.init.calculate_gain('relu'))

    #def load(self):
    #    checkpoint = torch.load('trained_models/1000_FSRCNN_x2_netG.pth')
    #    self.upscaleNet.load_state_dict(checkpoint['state_dict'])

class FSRCNN(nn.Module):
    def __init__(self, upscale_factor):
        super(FSRCNN, self).__init__()

        self.upscale_factor = upscale_factor
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=56, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(in_channels=56, out_channels=12, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv2d(in_channels=12, out_channels=56, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=56, out_channels=12, kernel_size=1, stride=1, padding=0)
        self.deconv1 = nn.ConvTranspose2d(in_channels=12, out_channels=3, kernel_size=8, stride=upscale_factor, padding=3)

        self._initialize_weights()

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.deconv1(x)
        return x

    def _initialize_weights(self):
        print("===> Initializing weights")
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)
        nn.init.xavier_uniform_(self.conv3.weight)
        nn.init.xavier_uniform_(self.conv4.weight)
        nn.init.orthogonal_(self.deconv1.weight, nn.init.calculate_gain('relu'))