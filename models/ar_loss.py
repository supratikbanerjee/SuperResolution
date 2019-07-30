import torch
import torch.nn as nn
from robust_loss_pytorch import adaptive
import numpy as np

class AdaptiveRobustLoss(nn.Module):
    def __init__(self, type):
        super(AdaptiveRobustLoss, self).__init__()
        self.type = type
 
    def forward(self, output, target):
        diff_imgs = (target-output)
        if self.type == 'G':
            criterion = adaptive.AdaptiveImageLossFunction(image_size = (diff_imgs.shape[2], diff_imgs.shape[3], diff_imgs.shape[1]), 
                float_dtype=np.float32, device=0, color_space='RGB', representation='PIXEL')
            diff_imgs.reshape(diff_imgs.shape[0], diff_imgs.shape[2], diff_imgs.shape[3], diff_imgs.shape[1])
            loss = criterion.lossfun(diff_imgs)
        elif self.type == 'D':
            criterion = adaptive.AdaptiveLossFunction(num_dims = 1, 
                float_dtype=np.float32, device=0)
            loss = criterion.lossfun(diff_imgs)

        return torch.mean(loss)