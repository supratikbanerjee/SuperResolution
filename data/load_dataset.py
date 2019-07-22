import torch.utils.data as data
from torchvision.transforms import Compose, CenterCrop, ToTensor, Resize, Normalize
import os
from PIL import Image

IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP']

class Dataset(data.Dataset):
	def __init__(self, config, phase):
		super(Dataset, self).__init__()
		image_dir = config[phase]['data_location']
		self.img_file = [os.path.join(image_dir, x) for x in os.listdir(image_dir) if is_image(x)]
		self.phase = phase
		self.upscale_factor = config['scale']
		self.config = config

	def __getitem__(self, index):
		# Can add repeat system here and augment data instead of just repeating iteration
		path = (self.img_file[index])
		input = load_image(self.img_file[index])
		target = input.copy()
		if self.phase == 'train':
			crop_size = self.config[self.phase]['size']

			self.input_transform = Compose([
			CenterCrop(crop_size),
			Resize(crop_size // self.upscale_factor),
			ToTensor(),
			])

			self.target_transform = Compose([
				CenterCrop(crop_size),
				ToTensor(),
				])
		else:
			width, height = input.size
			width = valid_crop(width, self.upscale_factor)
			height =  valid_crop(height, self.upscale_factor)
			crop_size = (height // self.upscale_factor, width // self.upscale_factor)
			t_crop_size = (height, width)

			self.input_transform = Compose([
				Resize(crop_size),
				ToTensor(),
				])

			self.target_transform = Compose([
				CenterCrop(t_crop_size),
				ToTensor(),
				])

		input = self.input_transform(input)
		target = self.target_transform(target)

		return input, target, path

	def __len__(self):
		return len(self.img_file)


def is_image(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def valid_crop(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)

def load_image(path):
	img = Image.open(path).convert('RGB')
	return img