import torch
from math import log10
from torch.autograd import Variable
import models.networks as networks
from collections import OrderedDict
from collections import defaultdict


class GAN():
	def __init__(self, config):
		super(GAN, self).__init__()

		self.logs = OrderedDict()
		self.eval = {'psnr':0.0, 'ssim':0.0, 'ssim_epoch':0, 'psnr_epoch':0}
		self.config = config
		train_config = self.config['train']
		self.device = self.config['device']

		# define model
		self.netG = networks.define_G(self.config).to(self.device)
		self.netD = networks.define_D(self.config).to(self.device)
		self.netF = networks.define_F(self.config).to(self.device)

		self.pixel_criterion = networks.loss_criterion(train_config['pixel_criterion']).to(self.device)
		self.adversarial_criterion = networks.loss_criterion(train_config['adversarial_criterion']).to(self.device)
		self.feature_criterion = networks.loss_criterion(train_config['feature_criterion']).to(self.device)

		self.pixel_weight = train_config['pixel_weight']
		self.adversarial_weight = train_config['gan_weight']
		self.feature_weight = train_config['feature_weight']
		self.real_weightD = train_config['real_weightD']
		self.fake_weightD = train_config['fake_weightD']

		self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=train_config['lr_G'],
                                                weight_decay=train_config['weight_decay_G'],
                                                betas=(train_config['beta1_G'], train_config['beta2_G']))
		#self.scheduler_G = torch.optim.lr_scheduler.StepLR(self.optimizer_G, step_size=train_config['lr_step'], 
		#	gamma=train_config['lr_gamma'])

		self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=train_config['lr_D'],
                                                weight_decay=train_config['weight_decay_D'],
                                                betas=(train_config['beta1_D'], train_config['beta2_D']))
		#self.scheduler_D = torch.optim.lr_scheduler.StepLR(self.optimizer_D, step_size=train_config['lr_step'], 
		#	gamma=train_config['lr_gamma'])

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

	def train_generator(self):
		#self.scheduler_G.step()
		self.netG.train()
		self.netG.zero_grad()
		real_features = Variable(self.netF(self.hr_real))
		fake_features = self.netF(self.hr_fake)
		pixel_loss_g = self.pixel_weight * self.pixel_criterion(self.hr_fake, self.hr_real) 
		feature_loss_g = self.feature_weight * self.feature_criterion(fake_features, real_features)
		content_loss = pixel_loss_g + feature_loss_g 
		proba = self.netD(self.hr_fake)
		adversarial_loss_g = self.adversarial_weight * self.adversarial_criterion(proba, self.get_target(proba, True))
		perceptual_loss = content_loss + adversarial_loss_g 
		# total_loss_g = pixel_loss_g + adversarial_loss_g + feature_loss_g
		self.logs['p_G'] = pixel_loss_g.item()
		self.logs['f_G'] = feature_loss_g.item()
		self.logs['a_G'] = adversarial_loss_g.item()
		self.logs['totalG'] = perceptual_loss.item()
		perceptual_loss.backward()
		self.optimizer_G.step()
		#return total_loss_g.item()

	def train_discriminator(self):
		#self.scheduler_D.step()
		self.netD.train()
		self.netD.zero_grad()
		real_proba = self.netD(self.hr_real)
		real_loss_d = self.adversarial_criterion(real_proba, self.get_target(real_proba, True))
		fake_proba = self.netD(Variable(self.hr_fake))
		fake_loss_d = self.adversarial_criterion(fake_proba, self.get_target(fake_proba, False))
		#print(real_proba, fake_proba)
		total_loss_d = ( self.real_weightD * real_loss_d + self.fake_weightD * fake_loss_d)
		self.logs['r_D'] = real_loss_d.item()
		self.logs['f_D'] = fake_loss_d.item()
		self.logs['totalD'] = total_loss_d.item()
		total_loss_d.backward()
		self.optimizer_D.step()
		#return total_loss_d.item()

	def train(self, batch):
		self.lr_input, self.hr_real = Variable(batch[0].squeeze().to(self.device)), Variable(batch[1].squeeze().to(self.device))
		self.hr_fake = self.netG(self.lr_input)
		self.train_generator()
		self.train_discriminator()
		

	def test(self, batch):
		self.netG.eval()
		self.lr_input, self.hr_real = Variable(batch[0].to(self.device)), Variable(batch[1].to(self.device))
		with torch.no_grad():
			self.hr_fake = self.netG(self.lr_input)

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
		loss_d = defaultdict(list)
		loss_G_vs_D = defaultdict(list)
		loss_g['pixel'] = self.logs['p_G']
		loss_g['feature'] =  self.logs['f_G']
		loss_g['adversarial'] = self.logs['a_G']
		loss_d['real'] = self.logs['r_D']
		loss_d['fale'] = self.logs['f_D']
		loss_G_vs_D['perceptual (G)'] = self.logs['totalG']
		loss_G_vs_D['adversarial (D)'] = self.logs['totalD']
		visualizer.plot(loss_g, epoch, 'Generator Loss', 'loss')
		visualizer.plot(loss_d, epoch, 'Discriminator Loss', 'loss')
		visualizer.plot(loss_G_vs_D, epoch, 'Generator vs Discriminator Loss', 'loss')

	def print_network_params(self, logger):
		param = OrderedDict()
		param['G'] = sum(map(lambda x: x.numel(), self.netG.parameters()))
		param['D'] = sum(map(lambda x: x.numel(), self.netD.parameters()))
		param['F'] = sum(map(lambda x: x.numel(), self.netF.parameters()))
		logger.log('Network G : {:s}, with parameters: [{:,d}]'.format(self.config['network_G']['model'], param['G']))
		logger.log('Network D : {:s}, with parameters: [{:,d}]'.format(self.config['network_D']['model'], param['D']))
		logger.log('Network F : {:s}, with parameters: [{:,d}]'.format(self.config['train']['feature_extractor'], param['F']))


	def get_current_visuals(self):
		visuals = OrderedDict()
		visuals['LR'] = self.lr_input.detach()[0].float().cpu()
		visuals['SR'] = self.hr_fake.detach()[0].float().cpu()
		visuals['HR'] = self.hr_real.detach()[0].float().cpu()
		return visuals

	def get_logs(self):
		return self.logs
