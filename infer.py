from __future__ import print_function
import argparse
import torch
from torch.autograd import Variable
from PIL import Image
from torchvision.transforms import ToTensor
import torchvision.transforms.functional as F
import time
import numpy as np
from models import networks

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('--input_image', type=str, required=True, help='input image to use')
parser.add_argument('--model', type=str, required=True, help='model file to use')
parser.add_argument('--output_filename', type=str, help='where to save the output image')
parser.add_argument('--cuda', action='store_true', help='use cuda')
opt = parser.parse_args()
device = torch.device("cuda")
# print(opt)
img = Image.open(opt.input_image).convert('RGB')
y, cb, cr = img.split()
#print(y)
checkpoint = torch.load(opt.model)
model =  networks.define_G(checkpoint['net'])
model.load_state_dict(checkpoint['state_dict'])
print(checkpoint['net'])
img_to_tensor = ToTensor()
input = img_to_tensor(img).view(1, -1, img.size[1], img.size[0])
#print(input)
#print(input.shape)

model = model.to(device)
input = input.to(device)
print('start')
model.eval()
start = time.time()
out = model(input)
end = time.time()
print('stop')
out = out.cpu()
out_img_R = out.detach().numpy()[0][0] * 255.0
out_img_G = out.detach().numpy()[0][1] * 255.0
out_img_B = out.detach().numpy()[0][2] * 255.0
#print(out_img_y.size)

out_img_R = out_img_R.clip(0, 255)
out_img_G = out_img_G.clip(0, 255)
out_img_B = out_img_B.clip(0, 255)

out_img_R = Image.fromarray(np.uint8(out_img_R))
out_img_G = Image.fromarray(np.uint8(out_img_G))
out_img_B = Image.fromarray(np.uint8(out_img_B))
out_img = Image.merge('RGB', [out_img_R, out_img_G, out_img_B])
print(end-start)
out_img.save(opt.output_filename)

print('output image saved to ', opt.output_filename)

