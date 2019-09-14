import os
import argparse
import time

import torch
import yaml

import util
from data import dataset
from models import create_model
from tqdm import tqdm


def main(config):
	device = torch.device(config['device'])

	##### Setup Visdom #####
	if config['use_visdom']:
		visualizer = util.Visualizer(config['name'], port=config['visdom_port'])


	##### Setup Dirs #####
	experiment_dir = config['logger']['path'] + config['name']
	util.mkdir_and_rename(
                experiment_dir)  # rename experiment folder if exists
	util.mkdirs((experiment_dir+'/val_image', experiment_dir+'/checkpoint'))

	##### Setup Logger #####
	logger = util.Logger('train', experiment_dir, 'train_' + config['name'])
	logger_val = util.Logger('val', experiment_dir, 'validation_' + config['name'])

	##### print Experiment Config
	logger.log(util.dict2str(config))

	###### Load Dataset #####
	training_data_loader = dataset.get_train_set(config['dataset'])
	logger.log('Training Dataset {:s} Loaded.'.format(config['dataset']['train']['name']))

	testing_data_loader = dataset.get_test_set(config['dataset'])
	logger.log('Validation Dataset {:s} Loaded.'.format(config['dataset']['test']['name']))

	train_size = len(training_data_loader)
	test_size = len(testing_data_loader)
	batch_size = config['dataset']['train']['batch_size']
	num_images = train_size * batch_size
	epochs = config['train']['epoch']
	total_iters = epochs * train_size

	##### Random Seed #####
	seed = config['train']['manual_seed']
	if seed is None:
		seed = random.randint(1, 10000)
	logger.log('Random seed: {}'.format(seed))
	util.set_random_seed(seed)


	logger.log('Number of train images: {:,d}, iters: {:,d}'.format(num_images, train_size))
	logger.log('Total iters: {:d} for {:,d} epochs'.format(total_iters, epochs))

	
	logger.log('Number of validation images: {:,d}'.format(test_size))

	trainer = create_model(config, logger)
	trainer.print_network_params(logger)

	print_freq = config['logger']['print_freq']
	val_freq = config['train']['val_freq']
	# val_freq += val_freq % train_size # Comment or others might not understand
	start_epoch = 1
	total_steps = 0	

	##### Start Training #####
	logger.log('Start training from epoch: {:d}, iter: {:d}'.format(start_epoch, total_steps))
	iter_start_time = time.time()
	for epoch in range(start_epoch, epochs+1):
		print()
		with tqdm(total=train_size, desc='Epoch: [%d/%d]'%(epoch, epochs), miniters=1) as t:
			for i, batch in enumerate(training_data_loader):
				trainer.train(batch)
				t.update()
				total_steps += 1

		if epoch % print_freq == 0:
			tt = time.time() - iter_start_time
			logs = trainer.get_logs()
			message = '[ Train ] - epoch:{}, iter:{}, time:{:.3f}'.format(epoch, total_steps, tt)
			for k, v in logs.items():
				message += '{:s}: {:.4f} '.format(k, v)
			logger.log(message, screen=False)
			if config['use_visdom']:
				visuals = trainer.get_current_visuals()
				visuals['LR'] = util.tensor2img(visuals['LR'])
				visuals['SR'] = util.tensor2img(visuals['SR'])
				visuals['HR'] = util.tensor2img(visuals['HR'])
				visualizer.display_current_results(visuals, epoch)
				trainer.plot_loss(visualizer, epoch)
				
			iter_start_time = time.time()


        ##### Start Validation #####
		if epoch % val_freq == 0:
			#logger_val = logging.getLogger('val')
			logger_val.log('Start validation at epoch: {:d}, iter: {:d}'.format(epoch, total_steps))
			valid_start_time = time.time()
			avg_psnr = 0.0
			avg_ssim = 0.0
			idx = 0
			for i, batch in enumerate(testing_data_loader):
				idx += 1
				img_name = batch[2][0][batch[2][0].rindex('/')+1:]
				# print(img_name)
				img_name = img_name[:img_name.index('.')]
				img_dir = experiment_dir+'/val_image/'+img_name
				util.mkdir(img_dir)
				inf_time = trainer.test(batch)                
				visuals = trainer.get_current_visuals()
				sr_img = util.tensor2img(visuals['SR'])  # uint8
				gt_img = util.tensor2img(visuals['HR'])  # uint8
				if epoch % config['logger']['img_freq'] == 0:
					save_img_path = os.path.join(img_dir, '{:d}_{:s}_{:d}.png'.format(epoch, img_name, total_steps))
					util.save_img(sr_img, save_img_path)
				crop_size = config['dataset']['scale']
				psnr, ssim = util.eval_psnr_and_ssim(sr_img, gt_img, crop_size) 
				avg_psnr += psnr
				avg_ssim += ssim
				logger_val.log('[ Inference ] - name:{}, dim:{}, time:{:.8f}, psnr: {:.4f}, ssim {:.4f}'.format(img_name, gt_img.shape[:2], inf_time, psnr, ssim))
			avg_psnr = avg_psnr / idx
			avg_ssim = avg_ssim / idx
			valid_t = time.time() - valid_start_time
			if config['use_visdom']:
				psnr_result = {'psnr': avg_psnr}
				visualizer.plot(psnr_result, epoch, 'psnr_eval', 'psnr')
				ssim_result = {'ssim': avg_ssim}
				visualizer.plot(ssim_result, epoch, 'ssim_eval', 'ssim')
				trainer.plot_loss(visualizer, epoch)
			logger_val.log('[ Test ] - epoch:{}, iter:{}, time:{:.3f}, psnr: {:.4f}, ssim {:.4f}'.format(
                    epoch, total_steps, valid_t, avg_psnr, avg_ssim))
			trainer.update_eval(avg_psnr, avg_ssim, epoch)
			best_eval = trainer.get_eval()
			logger_val.log('[ Best ] - psnr_epoch:{}, psnr:{:.4f}, ssim_epoch:{}, ssim: {:.4f}'.format(
                    best_eval['psnr_epoch'], best_eval['psnr'], best_eval['ssim_epoch'], best_eval['ssim']))
			iter_start_time = time.time()
		
		if epoch % config['logger']['chkpt_freq'] == 0:
			trainer.save(epoch)

		trainer.update_learning_rate(epoch)
	if config['use_visdom']:
		visualizer.save()
	

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-config', type=str, help='Path to config YAML file.')
	args = parser.parse_args()
	with open(args.config, 'r') as stream:
	    config = yaml.safe_load(stream)
	main(config)
