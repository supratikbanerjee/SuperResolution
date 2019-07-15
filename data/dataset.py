import torch.utils.data as data
from torchvision.transforms import Compose, CenterCrop, ToTensor, Resize, Normalize
import os
from PIL import Image


class Dataset(data.Dataset):
	def __init__(self, image_dir, input_transform=None, target_transform=None, path_transform=None):
		super(Dataset, self).__init__()
		self.img_file = [os.path.join(image_dir, x) for x in os.listdir(image_dir) if is_image_file(x)]
		self.input_transform = input_transform
		self.target_transform = target_transform

	def __getitem__(self, index):
		path = (self.img_file[index])
		input = load_image(self.img_file[index])
		target = input.copy()
		if self.input_transform:
			input = self.input_transform(input)
		if self.target_transform:
			target = self.target_transform(target)

		return input, target, path

	def __len__(self):
		return len(self.img_file)
 

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])

def load_image(path):
	img = Image.open(path).convert('RGB')
	return img

def input_transform(crop_size, upscale_factor, mean, stddev):
    return Compose([
        CenterCrop(crop_size),
        Resize(crop_size // upscale_factor),
        ToTensor(),
        #Normalize(mean, stddev),
    ])

def target_transform(crop_size,  mean, stddev):
    return Compose([
        CenterCrop(crop_size),
        ToTensor(),
        #Normalize(mean, stddev),
    ])

def path_transform():
    return Compose([
        ToTensor(),
    ])

def get_train_set(config):
	train_set = Dataset(config['train']['data_location'],
		input_transform=input_transform(config['train']['size'], config['scale'], config['train']['mean'], config['train']['stddev']), 
		target_transform=target_transform(config['train']['size'], config['test']['mean'], config['test']['stddev']), path_transform=path_transform())

	return data.DataLoader(dataset=train_set, 
		num_workers=config['train']['n_workers'], batch_size=config['train']['batch_size'], 
		shuffle=config['train']['shuffle'], pin_memory=True)
	
def get_test_set(config):
	test_set =Dataset(config['test']['data_location'],
		input_transform=input_transform(config['test']['size'], config['scale'], config['test']['mean'], config['test']['stddev']), 
		target_transform=target_transform(config['test']['size'], config['test']['mean'], config['test']['stddev']), path_transform=path_transform())

	return data.DataLoader(dataset=test_set, num_workers=config['test']['n_workers'], 
		batch_size=config['test']['batch_size'], shuffle=config['train']['shuffle'], pin_memory=True)
