import torch
from math import log10
from torch.autograd import Variable
import models.networks as networks
from collections import OrderedDict
from collections import defaultdict
import time

class GAN():
	def __init__(self, config):
		super(GAN, self).__init__()

		self.logs = OrderedDict()
		self.eval = {'psnr':0.0, 'ssim':0.0, 'ssim_epoch':0, 'psnr_epoch':0}
		self.config = config
		self.device = self.config['device']
		self.dual_D = self.config['dual_D']

		self.netG = networks.define_G(self.config).to(self.device)

		if self.config['is_train']:
			self.train_config = self.config['train']
			
			# define model
			self.netD = networks.define_D(self.config).to(self.device)
			self.netF = networks.define_F(self.config).to(self.device)
			if self.dual_D:
				self.netFD = networks.define_D({'network_D':{'model':'fdiscriminator_vgg_128'}}).to(self.device)

			self.pixel_criterion = networks.loss_criterion(self.train_config['pixel_criterion']).to(self.device)
			self.adversarial_criterion = networks.loss_criterion(self.train_config['adversarial_criterion']).to(self.device)
			self.feature_criterion = networks.loss_criterion(self.train_config['feature_criterion']).to(self.device)

			self.pixel_weight = self.train_config['pixel_weight']
			self.adversarial_weight = self.train_config['gan_weight']
			self.feature_weight = self.train_config['feature_weight']
			self.real_weightD = self.train_config['real_weightD']
			self.fake_weightD = self.train_config['fake_weightD']

			if self.train_config['pixel_criterion'] == 'ADAPTIVE':
				paramsG = list(self.netG.parameters()) + list(self.pixel_criterion.parameters())
			else:
				paramsG = self.netG.parameters()

			if self.train_config['adversarial_criterion'] == 'ADAPTIVE':
				paramsD = list(self.netD.parameters()) + list(self.pixel_criterion.parameters())
			else:
				paramsD = self.netD.parameters()

			self.optimizer_G = torch.optim.Adam(paramsG, lr=self.train_config['lr_G'],
	                                                weight_decay=self.train_config['weight_decay_G'],
	                                                betas=(self.train_config['beta1_G'], self.train_config['beta2_G']))
			if self.train_config['lr_scheme'] == 'MultiStepLR':
				self.scheduler_G = torch.optim.lr_scheduler.MultiStepLR(self.optimizer_G, 
					self.train_config['lr_step'], 
					self.train_config['lr_gamma'])

			self.optimizer_D = torch.optim.Adam(paramsD, lr=self.train_config['lr_D'],
	                                                weight_decay=self.train_config['weight_decay_D'],
	                                                betas=(self.train_config['beta1_D'], self.train_config['beta2_D']))
			if self.train_config['lr_scheme'] == 'MultiStepLR':
				self.scheduler_D = torch.optim.lr_scheduler.MultiStepLR(self.optimizer_D, 
					self.train_config['lr_step'], 
					self.train_config['lr_gamma'])
			self.load()
		else:
			self.load()

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
		real_features = self.netF(Variable(self.hr_real))

		pixel_loss_g = 0.0
		feature_loss_g = 0.0
		adversarial_loss_g = 0.0
		if self.train_config['cl_train']:
			loss_steps = [self.pixel_criterion(sr, self.hr_real)  for sr in self.hr_fake]
			fake_features = [self.netF(Variable(sr)) for sr in self.hr_fake]
			loss_steps_f = [self.feature_criterion(sr, real_features)  for sr in fake_features]
			probas = [self.netD(Variable(sr)) for sr in self.hr_fake]
			### TODO RAGAN
			loss_sets_a = [self.adversarial_criterion(proba, self.get_target(proba, True)) for proba in probas]
			for step in range(len(loss_steps)):
				pixel_loss_g += self.pixel_weight * loss_steps[step]
				feature_loss_g += self.feature_weight * loss_steps_f[step]
				adversarial_loss_g += self.adversarial_weight * loss_sets_a[step]
		else:
			fake_features = self.netF(self.hr_fake)
			pixel_loss_g = self.pixel_weight * self.pixel_criterion(self.hr_fake, self.hr_real) 
			feature_loss_g = self.feature_weight * self.feature_criterion(fake_features, real_features)
			proba = self.netD(self.hr_fake)
			if self.dual_D:
				fproba = self.netFD(fake_features)

			if self.train_config['gan_type'] == 'gan':
				adversarial_loss_g = self.adversarial_criterion(proba, self.get_target(proba, True))
			elif self.train_config['gan_type'] == 'ragan':
				
				proba_r = self.netD(self.hr_real)
				
				adversarial_loss_g =  (
					self.adversarial_criterion(proba_r - torch.mean(proba), self.get_target(proba_r, False)) +
					self.adversarial_criterion(proba - torch.mean(proba_r), self.get_target(proba, True))) / 2

				if self.dual_D:
					fproba_r = self.netFD(real_features)
					adv_feat_g = (
					self.adversarial_criterion(fproba_r - torch.mean(fproba), self.get_target(fproba_r, False)) +
					self.adversarial_criterion(fproba - torch.mean(fproba_r), self.get_target(fproba, True))) / 2
					adversarial_loss_g += adv_feat_g


		content_loss = pixel_loss_g + feature_loss_g
		perceptual_loss = content_loss + self.adversarial_weight * adversarial_loss_g
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
		if self.dual_D:
			self.netFD.train()
			self.netFD.zero_grad()
		
		real_proba = self.netD(self.hr_real)
		
		fake_loss_d = 0.0
		if self.train_config['cl_train']:
			fake_probas = [self.netD(Variable(sr)) for sr in self.hr_fake]
			real_loss_d = self.adversarial_criterion(real_proba, self.get_target(real_proba, True))
			loss_sets_d = [self.adversarial_criterion(fake_proba, self.get_target(fake_proba, False)) for fake_proba in fake_probas]
			for step in range(len(loss_sets_d)):
				fake_loss_d += loss_sets_d[step]
		else:
			real_features = self.netF(Variable(self.hr_real))
			fake_features = self.netF(Variable(self.hr_fake))

			fake_proba = self.netD(Variable(self.hr_fake))
			
			if self.train_config['gan_type'] == 'gan':
				real_loss_d = self.adversarial_criterion(real_proba, self.get_target(real_proba, True))
				fake_loss_d = self.adversarial_criterion(fake_proba, self.get_target(fake_proba, False))
			elif self.train_config['gan_type'] == 'ragan':
				real_loss_d = self.adversarial_criterion(real_proba - torch.mean(fake_proba), self.get_target(real_proba, True))
				fake_loss_d = self.adversarial_criterion(fake_proba - torch.mean(real_proba), self.get_target(fake_proba, False))
				if self.dual_D:
					real_fproba = self.netFD(Variable(real_features))
					fake_fproba = self.netFD(Variable(fake_features))
					real_floss_d = self.adversarial_criterion(real_fproba - torch.mean(fake_fproba), self.get_target(real_fproba, True))
					fake_floss_d = self.adversarial_criterion(fake_fproba - torch.mean(real_fproba), self.get_target(fake_fproba, False))

		#print(real_proba, fake_proba)
		total_loss_d = (self.real_weightD * real_loss_d + self.fake_weightD * fake_loss_d)
		if self.dual_D:
			total_floss_d = (self.real_weightD * real_floss_d + self.fake_weightD * fake_floss_d)
			total_loss_d += total_floss_d
		total = total_loss_d

		self.logs['r_D'] = real_loss_d.item()
		self.logs['f_D'] = fake_loss_d.item()
		self.logs['totalD'] = total.item()
		total.backward()
		self.optimizer_D.step()
		#return total_loss_d.item()

	def train(self, batch):
		self.lr_input, self.hr_real = Variable(batch[0].to(self.device)), Variable(batch[1].to(self.device))
		self.hr_fake = self.netG(self.lr_input)
		self.train_generator()
		self.train_discriminator()
		if isinstance(self.hr_fake, list):
				self.hr_fake = self.hr_fake[-1]
		

	def test(self, batch):
		self.netG.eval()
		self.lr_input, self.hr_real = Variable(batch[0].to(self.device)), Variable(batch[1].to(self.device))
		with torch.no_grad():
			infer_time_start = time.time()
			self.hr_fake = self.netG(self.lr_input)
			infer_time = time.time() - infer_time_start
			if isinstance(self.hr_fake, list):
				self.hr_fake = self.hr_fake[-1]
		if self.config['is_train']:
			valid_loss = self.pixel_weight * self.pixel_criterion(self.hr_fake, self.hr_real)
			self.logs['v_l'] = valid_loss.item()

		return infer_time

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
		loss_tv = defaultdict(list)
		loss_tv['train'] = self.logs['p_G']
		loss_tv['valid'] = self.logs['v_l']
		loss_g['pixel'] = self.logs['p_G']
		loss_g['feature'] =  self.logs['f_G']
		loss_g['adversarial'] = self.logs['a_G']
		loss_d['real'] = self.logs['r_D']
		loss_d['fale'] = self.logs['f_D']
		loss_G_vs_D['perceptual (G)'] = self.logs['totalG']
		loss_G_vs_D['adversarial (D)'] = self.logs['totalD']
		visualizer.plot(loss_tv, epoch, 'Train VS Valid Loss', 'loss')
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
		if self.dual_D:
			param['FD'] = sum(map(lambda x: x.numel(), self.netFD.parameters()))
			logger.log('Network FD : {:s}, with parameters: [{:,d}]'.format('feature_D', param['FD']))
		


	def get_current_visuals(self):
		visuals = OrderedDict()
		visuals['LR'] = self.lr_input.detach()[0].float().cpu()
		visuals['SR'] = self.hr_fake.detach()[0].float().cpu()
		visuals['HR'] = self.hr_real.detach()[0].float().cpu()
		return visuals

	def get_logs(self):
		return self.logs

	def load(self):
		checkpoint = torch.load(self.config['path']['pretrain_model_G'])
		self.netG.load_state_dict(checkpoint['state_dict'])

	def update_learning_rate(self, epoch):
		self.scheduler_G.step(epoch)
		self.scheduler_D.step(epoch)