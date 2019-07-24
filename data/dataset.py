import torch.utils.data as data
import numpy as np
from tqdm import tqdm
import numpy as np
import random
import os
import scipy.misc as misc
from torchvision.transforms import Compose, CenterCrop, ToTensor

IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP']


class Dataset(data.Dataset):
	def __init__(self, config, phase):
		super(Dataset, self).__init__()
		image_dir = config[phase]['data_location']
		self.PATH = [os.path.join(image_dir, x) for x in os.listdir(image_dir) if is_image(x)]
		self.config = config
		self.phase = phase
		self.repeat = config[self.phase]['repeat']
		self.upscale_factor = config['scale']
		self.LR = []
		self.HR = []
		self.target_transform = Compose([
				ToTensor(),
				])
		self.enum_dl()

	def __getitem__(self, index):
		if self.phase == 'train':
			index = index % len(self.PATH)
		path = self.PATH[index]
		LR = self.LR[index]
		HR = self.HR[index]
		
		if self.phase == 'train':
			crop_size = self.config[self.phase]['lr_size']
			LR, HR = get_patch(LR, HR, crop_size, self.upscale_factor)

		LR = self.target_transform(LR) 
		HR = self.target_transform(HR)
		return LR, HR, path


	def enum_dl(self):
		batch_size = self.config[self.phase]['batch_size']
		with tqdm(total=len(self.PATH), desc=self.config[self.phase]['name'], miniters=1) as t:
			for i, path in enumerate(self.PATH):
				img = load_image(path)
				LR = misc.imresize(img, (int(img.shape[0]/self.upscale_factor),int(img.shape[1]/self.upscale_factor)), interp='bilinear')
				self.LR.append(LR)
				self.HR.append(img)
				t.set_postfix_str("Batches: [%d/%d]" % ((i+1)//batch_size, len(self.PATH)/batch_size))				
				t.update()

	def __len__(self):
		return len(self.PATH) * self.repeat

def get_patch(img_in, img_tar, patch_size, scale):
    ih, iw = img_in.shape[:2]
    oh, ow = img_tar.shape[:2]
    ip = patch_size
    tp = ip * scale
    ix = random.randrange(0, iw - ip + 1)
    iy = random.randrange(0, ih - ip + 1)
    tx, ty = scale * ix, scale * iy
    img_in = img_in[iy:iy + ip, ix:ix + ip, :]
    img_tar = img_tar[ty:ty + tp, tx:tx + tp, :]
    return img_in, img_tar

def is_image(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def load_image(path):
	img = misc.imread(path, mode='RGB')
	return img

def get_train_set(config):

	train_set = Dataset(config, 'train')
	return data.DataLoader(dataset=train_set, batch_size=config['train']['batch_size'], shuffle=config['train']['shuffle'], pin_memory=True)
	
def get_test_set(config):

	test_set = Dataset(config, 'test')
	return data.DataLoader(dataset=test_set, batch_size=config['test']['batch_size'], shuffle=config['test']['shuffle'], pin_memory=True)