import torch.utils.data as data
import numpy as np
from tqdm import tqdm
from data import load_dataset

class Dataset(data.Dataset):
	def __init__(self, config, phase, data_loader):
		super(Dataset, self).__init__()
		self.config = config
		self.phase = phase
		self.data_loader = data_loader
		self.repeat = config[self.phase]['repeat']
		self.LR = []
		self.HR = []
		self.PATH = []
		self.enum_dl()

	def __getitem__(self, index):
		if self.phase == 'train':
			index = index % len(self.PATH)
		path = self.PATH[index]
		LR = self.LR[index]
		HR = self.HR[index]
		return LR, HR, path

	def enum_dl(self):
		batch_size = self.config[self.phase]['batch_size']
		with tqdm(total=len(self.data_loader), desc=self.config[self.phase]['name'], miniters=1) as t:
			for i, data in enumerate(self.data_loader):
				self.LR.append((data[0].squeeze()))
				self.HR.append((data[1].squeeze()))
				self.PATH.append(data[2][0])
				t.set_postfix_str("Loaded: [%d/%d]" % ((i+1)*batch_size, len(self.data_loader)*batch_size))				
				t.update()

	def __len__(self):
		return len(self.PATH) * self.repeat


def get_train_set(config):
	train_data = load_dataset.Dataset(config, 'train')
	data_loader = data.DataLoader(dataset=train_data, batch_size=config['train']['batch_size'], num_workers=config['train']['n_workers'], shuffle=config['train']['shuffle'], pin_memory=True)
	train_set = Dataset(config, 'train', data_loader)
	return data.DataLoader(dataset=train_set)
	
def get_test_set(config):
	test_data = load_dataset.Dataset(config, 'test')
	data_loader = data.DataLoader(dataset=test_data, batch_size=config['test']['batch_size'], num_workers=config['test']['n_workers'], shuffle=config['test']['shuffle'], pin_memory=True)
	test_set = Dataset(config, 'test', data_loader)
	return data.DataLoader(dataset=test_set)