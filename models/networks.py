import torchvision
import torch.nn as nn
from models.modules.capsule_network import CapsuleNetwork
import models.modules.cnn_models as CNN_arch
import models.modules.discriminator_vgg_arch as SRGAN_arch
import models.modules.RRDBNet_arch as RRDBNet_arch
import models.modules.SRResNet_arch as SRResNet_arch


# Generator
def define_G(config):
	net_config = config['network_G']
	model = net_config['model']
	if model == 'SRResNet':
		netG = SRResNet_arch.SRResNet(config['network_G']['layers'], config['dataset']['scale'])
	elif model == 'MSRResNet':
		netG = SRResNet_arch.MSRResNet(in_nc=net_config['in_nc'], out_nc=net_config['out_nc'],
                                       nf=net_config['nf'], nb=net_config['nb'], upscale=config['dataset']['scale'])
	elif model == 'RRDBNet':
		netG = RRDBNet_arch.RRDBNet(in_nc=net_config['in_nc'], out_nc=net_config['out_nc'],
			nf=net_config['nf'], nb=net_config['nb'])
	elif model == 'SRCNN':
		netG = CNN_arch.SRCNN(config['dataset']['scale'])
	elif model == 'FSRCNN':
		netG = CNN_arch.FSRCNN(config['dataset']['scale'])
	else:
		raise NotImplementedError('Generator model [{:s}] not recognized'.format(model))
	return netG

# Discriminator
def define_D(config):
	net_config = config['network_D']
	model = net_config['model']
	if model == 'LeakyVGG':
		netD = SRGAN_arch.DLeakyVGG()
	elif model == 'SwishVGG':
		netD = SRGAN_arch.DSwishVGG()
	elif model == 'discriminator_vgg_128':
		netD = SRGAN_arch.Discriminator_VGG_128(in_nc=net_config['in_nc'], nf=net_config['nf'])
	elif model == 'CapsNet':
		netD = CapsuleNetwork(image_width=net_config['size'],
                         image_height=net_config['size'],
                         image_channels=net_config['channels'],
                         conv_inputs=net_config['conv_in'],
                         conv_outputs=net_config['conv_out'],
                         num_primary_units=net_config['num_units'],
                         primary_unit_size=net_config['unit_size'],
                         num_output_units=net_config['out'],
                         output_unit_size=net_config['out_unit_size'])
	else:
		raise NotImplementedError('Discriminator model [{:s}] not recognized'.format(model))
	return netD

# Perceptual Loss Feature Extraction Model
def define_F(config):
	model = config['train']['feature_extractor']
	if model == 'vgg19':
		netF = SRGAN_arch.FeatureExtractor(torchvision.models.vgg19(pretrained=True))
	elif model == 'vgg19_bn':
		feature_layer = 34
		use_bn = False
		netF = SRGAN_arch.VGGFeatureExtractor(feature_layer=feature_layer, use_bn=use_bn,
                                          use_input_norm=True, device=config['device'])
	else:
		raise NotImplementedError('FeatureExtraction model [{:s}] not recognized.'.format(model))
	netF.eval()
	return netF

# Pixel Loss
def pixel_criterion(config):
	loss = config['pixel_criterion']
	if loss == 'l1':
		criterion = nn.L1Loss()
	elif loss == 'l2':
		criterion = nn.MSELoss()
	else:
		raise NotImplementedError('Loss type [{:s}] not recognized.'.format(loss))
	return criterion

# Feature Loss
def feature_criterion(config):
	loss = config['feature_criterion']
	if loss == 'l1':
		criterion = nn.L1Loss()
	elif loss == 'l2':
		criterion = nn.MSELoss()
	else:
		raise NotImplementedError('Loss type [{:s}] not recognized.'.format(loss))
	return criterion

# GAN Loss
def adversarial_loss(config):
	loss = config['gan_type']
	if loss == 'ragan' or 'gan':
		criterion = nn.BCEWithLogitsLoss()
	else:
		raise NotImplementedError('Loss type [{:s}] not recognized.'.format(loss))
	return criterion
