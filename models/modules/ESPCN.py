import torch
import torch.nn as nn
import torch.nn.functional as F

class ESPCN(nn.Module):
    def __init__(self, scale):
        super(ESPCN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 3 * scale * scale, kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(scale)
        self.conv4 = nn.Conv2d(3, 3, kernel_size=1, padding=0)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.pixel_shuffle(x)
        x = self.conv4(x)
        return x
