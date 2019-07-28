import torch.nn as nn
import torch.nn.functional as F


class SRCNN(nn.Module):
    def __init__(self, upscale_factor):
        super(SRCNN, self).__init__()

        self.upscale_factor = upscale_factor
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=9, stride=1, padding=3)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=3, kernel_size=5, stride=1, padding=2)

        self._initialize_weights()

    def forward(self, x):
        x = F.interpolate(x , scale_factor=self.upscale_factor, mode='bilinear')
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