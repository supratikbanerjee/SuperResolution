import torch
from math import log10
from torch.autograd import Variable
import models.networks as networks
from collections import OrderedDict
from collections import defaultdict


class CNN():
	def __init__(self, config):
		super(CNN, self).__init__()

		self.logs = OrderedDict()
		self.eval = {'psnr':0.0, 'ssim':0.0, 'ssim_epoch':0, 'psnr_epoch':0}
		self.config = config
		self.train_config = self.config['train']
		self.device = self.config['device']

		# define model
		self.netG = networks.define_G(self.config).to(self.device)

		self.pixel_criterion = networks.loss_criterion(self.train_config['pixel_criterion']).to(self.device)

		self.pixel_weight = self.train_config['pixel_weight']


		self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=self.train_config['lr_G'],
                                                weight_decay=self.train_config['weight_decay_G'],
                                                betas=(self.train_config['beta1_G'], self.train_config['beta2_G']))
		#self.scheduler_G = torch.optim.lr_scheduler.StepLR(self.optimizer_G, step_size=train_config['lr_step'], 
		#	gamma=self.train_config['lr_gamma'])

	def update_eval(self, psnr, ssim, epoch):
		if ssim > self.eval['ssim']:
			self.eval['ssim'] = ssim
			self.eval['ssim_epoch'] = epoch
		if psnr > self.eval['psnr']:
			self.eval['psnr'] = psnr
			self.eval['psnr_epoch'] = epoch

	def get_eval(self):
		return self.eval

	def get_target(self, input, target):
		return torch.empty_like(input).fill_(target)

	def train(self, batch):
		self.lr_input, self.hr_real = Variable(batch[0].to(self.device)), Variable(batch[1].to(self.device))
		self.netG.train()
		self.netG.zero_grad()
		self.hr_fake = self.netG(self.lr_input)
		#print(self.hr_fake[0].shape, len(self.hr_fake))
		pixel_loss_g = 0.0
		if self.train_config['cl_train']:
			loss_steps = [self.pixel_criterion(sr, self.hr_real)  for sr in self.hr_fake]
			for step in range(len(loss_steps)):
				pixel_loss_g += self.pixel_weight * loss_steps[step]
		else:
			pixel_loss_g = self.pixel_weight * self.pixel_criterion(self.hr_fake, self.hr_real) 
		pixel_loss_g.backward()
		self.optimizer_G.step()
		self.logs['p_G'] = pixel_loss_g.item()

	def test(self, batch):
		self.netG.eval()
		self.lr_input, self.hr_real = Variable(batch[0].to(self.device)), Variable(batch[1].to(self.device))
		with torch.no_grad():
			self.hr_fake = self.netG(self.lr_input)
			if isinstance(self.hr_fake, list):
				self.hr_fake = self.hr_fake[-1]

	def save(self, epoch):
		model_path = self.config['logger']['path'] + self.config['name'] +'/checkpoint/'
		model_name = "{}_{}_x{}_netG.pth".format(epoch, self.config['network_G']['model'], self.config['dataset']['scale'])
		to_save = {
		'state_dict': self.netG.state_dict(),
		'net': self.config
		}
		torch.save(to_save, model_path+model_name)

	def plot_loss(self, visualizer, epoch):
		loss_g = defaultdict(list)
		loss_g['pixel'] = self.logs['p_G']
		visualizer.plot(loss_g, epoch, 'Generator Loss', 'loss')

	def print_network_params(self, logger):
		param = OrderedDict()
		param['G'] = sum(map(lambda x: x.numel(), self.netG.parameters()))
		logger.log('Network G : {:s}, with parameters: [{:,d}]'.format(self.config['network_G']['model'], param['G']))
		
	def get_current_visuals(self):
		visuals = OrderedDict()
		visuals['LR'] = self.lr_input.detach()[0].float().cpu()
		visuals['SR'] = self.hr_fake.detach()[0].float().cpu()
		visuals['HR'] = self.hr_real.detach()[0].float().cpu()
		return visuals

	def get_logs(self):
		return self.logs
