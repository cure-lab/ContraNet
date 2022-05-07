from utils.model_ops import *
from utils.misc import *

import torch
import torch.nn as nn
import torch.nn.functional as F

import json
import os
import sys
import warnings
from argparse import ArgumentParser
from torch.backends import cudnn
from utils.log import make_run_name
import random


import glob
import os
import random
from os.path import dirname, abspath, exists, join


from data_utils.load_dataset import *

from utils.log import make_checkpoint_dir, make_logger
from utils.losses import *
from utils.load_checkpoint import load_checkpoint
from utils.misc import *

from sync_batchnorm.batchnorm import convert_model
from worker import make_worker

import torch
from torch.utils.data import DataLoader
from torch.nn import DataParallel

from tensorboardX import SummaryWriter
import encoder as encoder_source

import numpy as np
import sys
import glob
import random
from scipy import ndimage
from sklearn.manifold import TSNE
from os.path import join
from PIL import Image
from tqdm import tqdm
from datetime import datetime
import numpy
from densenet import densenet169

from metrics.IS import calculate_incep_score
from metrics.FID import calculate_fid_score
from metrics.F_beta import calculate_f_beta_score
from metrics.Accuracy import calculate_accuracy
#from utils.ada import augment
from utils.biggan_utils import interp
from utils.sample import sample_latents, sample_1hot, make_mask, target_class_sampler
from utils.misc import *
from utils.losses import calc_derv4gp, calc_derv4dra, calc_derv, latent_optimise, set_temperature
from utils.losses import Conditional_Contrastive_loss, Proxy_NCA_loss, NT_Xent_loss
from utils.diff_aug import DiffAugment
from utils.cr_diff_aug import CR_DiffAug

import torch
import torch.nn as nn
from torch.nn import DataParallel
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
import torch.nn.functional as F
import torchvision
from torchvision import transforms

import encoder
import pytorch_ssim as ssim_package

from utilss.utils import rm_dir, cuda, where
import torch.optim as optim
import foolbox as fb 
import time



RUN_NAME_FORMAT = (
	"{framework}-"
	"{phase}-"
	"{timestamp}"
	)

LOG_FORMAT = (
	"Step: {step:>7} "
	"Progress: {progress:<.1%} "
	"Elapsed: {elapsed} "
	"temperature: {temperature:<.6} "
	"ada_p: {ada_p:<.6} "
	"Discriminator_loss: {dis_loss:<.6} "
	"Generator_loss: {gen_loss:<.6} "
)



class DiscOptBlock(nn.Module):
	def __init__(self, in_channels, out_channels, d_spectral_norm, activation_fn):
		super(DiscOptBlock, self).__init__()
		self.d_spectral_norm = d_spectral_norm

		if d_spectral_norm:
			self.conv2d0 = snconv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)
			self.conv2d1 = snconv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
			self.conv2d2 = snconv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
		else:
			self.conv2d0 = conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)
			self.conv2d1 = conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
			self.conv2d2 = conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)

			self.bn0 = batchnorm_2d(in_features=in_channels)
			self.bn1 = batchnorm_2d(in_features=out_channels)

		if activation_fn == "ReLU":
			self.activation = nn.ReLU(inplace=True)
		elif activation_fn == "Leaky_ReLU":
			self.activation = nn.LeakyReLU(negative_slope=0.1, inplace=True)
		elif activation_fn == "ELU":
			self.activation = nn.ELU(alpha=1.0, inplace=True)
		elif activation_fn == "GELU":
			self.activation = nn.GELU()
		else:
			raise NotImplementedError

		self.average_pooling = nn.AvgPool2d(2)


	def forward(self, x):
		x0 = x
		x = self.conv2d1(x)
		if self.d_spectral_norm is False:
			x = self.bn1(x)
		x = self.activation(x)
		x = self.conv2d2(x)
		x = self.average_pooling(x)

		x0 = self.average_pooling(x0)
		if self.d_spectral_norm is False:
			x0 = self.bn0(x0)
		x0 = self.conv2d0(x0)

		out = x + x0
		return out


class DiscBlock(nn.Module):
	def __init__(self, in_channels, out_channels, d_spectral_norm, activation_fn, downsample=True):
		super(DiscBlock, self).__init__()
		self.d_spectral_norm = d_spectral_norm
		self.downsample = downsample

		if activation_fn == "ReLU":
			self.activation = nn.ReLU(inplace=True)
		elif activation_fn == "Leaky_ReLU":
			self.activation = nn.LeakyReLU(negative_slope=0.1, inplace=True)
		elif activation_fn == "ELU":
			self.activation = nn.ELU(alpha=1.0, inplace=True)
		elif activation_fn == "GELU":
			self.activation = nn.GELU()
		else:
			raise NotImplementedError

		self.ch_mismatch = False
		if in_channels != out_channels:
			self.ch_mismatch = True

		if d_spectral_norm:
			if self.ch_mismatch or downsample:
				self.conv2d0 = snconv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)
			self.conv2d1 = snconv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
			self.conv2d2 = snconv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
		else:
			if self.ch_mismatch or downsample:
				self.conv2d0 = conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)
			self.conv2d1 = conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
			self.conv2d2 = conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)

			if self.ch_mismatch or downsample:
				self.bn0 = batchnorm_2d(in_features=in_channels)
			self.bn1 = batchnorm_2d(in_features=in_channels)
			self.bn2 = batchnorm_2d(in_features=out_channels)

		self.average_pooling = nn.AvgPool2d(2)


	def forward(self, x):
		x0 = x

		if self.d_spectral_norm is False:
			x = self.bn1(x)
		x = self.activation(x)
		x = self.conv2d1(x)
		if self.d_spectral_norm is False:
			x = self.bn2(x)
		x = self.activation(x)
		x = self.conv2d2(x)
		if self.downsample:
			x = self.average_pooling(x)

		if self.downsample or self.ch_mismatch:
			if self.d_spectral_norm is False:
				x0 = self.bn0(x0)
			x0 = self.conv2d0(x0)
			if self.downsample:
				x0 = self.average_pooling(x0)

		out = x + x0
		return out


class Discriminator(nn.Module):
	"""Discriminator."""
	def __init__(self, img_size, d_conv_dim, d_spectral_norm, attention, attention_after_nth_dis_block, activation_fn, conditional_strategy,
				 hypersphere_dim, num_classes, nonlinear_embed, normalize_embed, initialize, D_depth, mixed_precision):
		super(Discriminator, self).__init__()
		d_in_dims_collection = {"32": [3] + [d_conv_dim*2, d_conv_dim*2, d_conv_dim*2],
								"64": [3] + [d_conv_dim, d_conv_dim*2, d_conv_dim*4, d_conv_dim*8],
								"128": [3] +[d_conv_dim, d_conv_dim*2, d_conv_dim*4, d_conv_dim*8, d_conv_dim*16],
								"256": [3] +[d_conv_dim, d_conv_dim*2, d_conv_dim*4, d_conv_dim*8, d_conv_dim*8, d_conv_dim*16],
								"512": [3] +[d_conv_dim, d_conv_dim, d_conv_dim*2, d_conv_dim*4, d_conv_dim*8, d_conv_dim*8, d_conv_dim*16]}

		d_out_dims_collection = {"32": [d_conv_dim*2, d_conv_dim*2, d_conv_dim*2, d_conv_dim*2],
								 "64": [d_conv_dim, d_conv_dim*2, d_conv_dim*4, d_conv_dim*8, d_conv_dim*16],
								 "128": [d_conv_dim, d_conv_dim*2, d_conv_dim*4, d_conv_dim*8, d_conv_dim*16, d_conv_dim*16],
								 "256": [d_conv_dim, d_conv_dim*2, d_conv_dim*4, d_conv_dim*8, d_conv_dim*8, d_conv_dim*16, d_conv_dim*16],
								 "512": [d_conv_dim, d_conv_dim, d_conv_dim*2, d_conv_dim*4, d_conv_dim*8, d_conv_dim*8, d_conv_dim*16, d_conv_dim*16]}

		d_down = {"32": [True, True, False, False],
				  "64": [True, True, True, True, False],
				  "128": [True, True, True, True, True, False],
				  "256": [True, True, True, True, True, True, False],
				  "512": [True, True, True, True, True, True, True, False]}

		self.nonlinear_embed = nonlinear_embed
		self.normalize_embed = normalize_embed
		self.conditional_strategy = conditional_strategy
		self.mixed_precision = mixed_precision

		self.in_dims  = d_in_dims_collection[str(img_size)]
		self.out_dims = d_out_dims_collection[str(img_size)]
		down = d_down[str(img_size)]

		self.blocks = []
		for index in range(len(self.in_dims)):
			if index == 0:
				self.blocks += [[DiscOptBlock(in_channels=self.in_dims[index],
											  out_channels=self.out_dims[index],
											  d_spectral_norm=d_spectral_norm,
											  activation_fn=activation_fn)]]
			else:
				self.blocks += [[DiscBlock(in_channels=self.in_dims[index],
										   out_channels=self.out_dims[index],
										   d_spectral_norm=d_spectral_norm,
										   activation_fn=activation_fn,
										   downsample=down[index])]]

			if index+1 == attention_after_nth_dis_block and attention is True:
				self.blocks += [[Self_Attn(self.out_dims[index], d_spectral_norm)]]

		self.blocks = nn.ModuleList([nn.ModuleList(block) for block in self.blocks])

		if activation_fn == "ReLU":
			self.activation = nn.ReLU(inplace=True)
		elif activation_fn == "Leaky_ReLU":
			self.activation = nn.LeakyReLU(negative_slope=0.1, inplace=True)
		elif activation_fn == "ELU":
			self.activation = nn.ELU(alpha=1.0, inplace=True)
		elif activation_fn == "GELU":
			self.activation = nn.GELU()
		else:
			raise NotImplementedError

		if d_spectral_norm:
			self.linear1 = snlinear(in_features=self.out_dims[-1], out_features=1)
			if self.conditional_strategy in ['ContraGAN', 'Proxy_NCA_GAN', 'NT_Xent_GAN']:
				self.linear2 = snlinear(in_features=self.out_dims[-1], out_features=hypersphere_dim)
				if self.nonlinear_embed:
					self.linear3 = snlinear(in_features=hypersphere_dim, out_features=hypersphere_dim)
				self.embedding = sn_embedding(num_classes, hypersphere_dim)
			elif self.conditional_strategy == 'ProjGAN':
				self.embedding = sn_embedding(num_classes, self.out_dims[-1])
			elif self.conditional_strategy == 'ACGAN':
				self.linear4 = snlinear(in_features=self.out_dims[-1], out_features=num_classes)
			else:
				pass
		else:
			self.linear1 = linear(in_features=self.out_dims[-1], out_features=1)
			if self.conditional_strategy in ['ContraGAN', 'Proxy_NCA_GAN', 'NT_Xent_GAN']:
				self.linear2 = linear(in_features=self.out_dims[-1], out_features=hypersphere_dim)
				if self.nonlinear_embed:
					self.linear3 = linear(in_features=hypersphere_dim, out_features=hypersphere_dim)
				self.embedding = embedding(num_classes, hypersphere_dim)
			elif self.conditional_strategy == 'ProjGAN':
				self.embedding = embedding(num_classes, self.out_dims[-1])
			elif self.conditional_strategy == 'ACGAN':
				self.linear4 = linear(in_features=self.out_dims[-1], out_features=num_classes)
			else:
				pass

		# Weight init
		if initialize is not False:
			init_weights(self.modules, initialize)


	def forward(self, x, label, evaluation=False):
		with torch.cuda.amp.autocast() if self.mixed_precision is True and evaluation is False else dummy_context_mgr() as mp:
			h = x
			for index, blocklist in enumerate(self.blocks):
				for block in blocklist:
					h = block(h)
			h = self.activation(h)
			h = torch.sum(h, dim=[2,3]) 

			if self.conditional_strategy == 'no':
				authen_output = torch.squeeze(self.linear1(h))
				return authen_output

			elif self.conditional_strategy in ['ContraGAN', 'Proxy_NCA_GAN', 'NT_Xent_GAN']:
				authen_output = torch.squeeze(self.linear1(h))
				cls_proxy = self.embedding(label)
				cls_embed = self.linear2(h)
				if self.nonlinear_embed:
					cls_embed = self.linear3(self.activation(cls_embed))
				if self.normalize_embed:
					cls_proxy = F.normalize(cls_proxy, dim=1)
					cls_embed = F.normalize(cls_embed, dim=1)
				return cls_proxy, cls_embed, authen_output

			elif self.conditional_strategy == 'ProjGAN':
				authen_output = torch.squeeze(self.linear1(h))
				proj = torch.sum(torch.mul(self.embedding(label), h), 1)
				return proj + authen_output

			elif self.conditional_strategy == 'ACGAN':
				authen_output = torch.squeeze(self.linear1(h))
				cls_output = self.linear4(h)
				return cls_output, authen_output

			else:
				raise NotImplementedError


def main():
	parser = ArgumentParser(add_help=False)
	parser.add_argument('-c', '--config_path', type=str, default='./configs/CIFAR10/DiffAugGAN(P).json')
	parser.add_argument('-n', '--nodes', default=1, type=int, metavar='N')
	parser.add_argument('--checkpoint_folder', default="./best_version1_cifar10_checkpoints")
	parser.add_argument('--load_current', default=True)

	parser.add_argument('--log_output_path', default="./adv_train_mydiscriminator_logs")
	parser.add_argument('--seed', default=1)
	parser.add_argument('--num_workers', default=8)
	parser.add_argument('-t', '--train', default=True)
	parser.add_argument('-e', '--eval', default=True)
	parser.add_argument('--print_every', type=int, default=100, help='control log interval')
	parser.add_argument('--save_every', type=int, default=2000, help='control evaluation and save interval')
	parser.add_argument('--eval_type', type=str, default='test', help='[train/valid/test]')
	args = parser.parse_args()


	if args.config_path is not None:
		with open(args.config_path) as f:
			model_configs = json.load(f)
		train_configs = vars(args)
	else:
		raise NotImplementedError

	fix_all_seed(train_configs['seed'])
	cudnn.benchmark, cudnn.deterministic = False, True

	gpus_per_node, rank = torch.cuda.device_count(), torch.cuda.current_device()
	local_rank = rank
	# adding classifier
	checkpoint_path = '/research/dept6/yjyang/SP2020/V2CIFAR10_Generation/cifar10_models/state_dicts'
	classifier = densenet169().to(local_rank)
	classifier_path = os.path.join(checkpoint_path,"densenet169.pt")
	classifier_ckpt = torch.load(classifier_path)
	classifier.load_state_dict(classifier_ckpt)
	classifier.eval().to(local_rank)
	mean = [0.4914, 0.4822, 0.4465]
	std = [0.2023, 0.1994, 0.2010]
	mean =torch.tensor(mean, dtype=torch.float32).cuda()
	std =  torch.tensor(std, dtype=torch.float32).cuda()
	mean = mean[:,None, None]
	std = std[:, None, None]
	bounds = (0, 1)
	preprocessing = dict(mean = [0.4914, 0.4822, 0.4465], std = [0.2023, 0.1994, 0.2010],  axis=-3)
	fmodel = fb.PyTorchModel(classifier, bounds=bounds, preprocessing=preprocessing)

	max_iter = 100







	world_size = gpus_per_node*train_configs['nodes']
	if world_size == 1:
		warnings.warn('You have chosen a specific GPU. This will completely disable data parallelism.')

	run_name =  "pn_batchsize64_mydiscriminator_"+ make_run_name(RUN_NAME_FORMAT, framework=train_configs['config_path'].split('/')[-1][:-5], phase='train') 
	#print(run_name)

	cfgs = dict2clsattr(train_configs, model_configs)
	step = 0
	best_step = 0
	global_rank = local_rank = rank

	writer = SummaryWriter(log_dir=join("./mydiscriminator_logs", run_name)) if local_rank == 0 else None
	if local_rank == 0:
		logger = make_logger(run_name, None)
		logger.info('Run name : {run_name}'.format(run_name=run_name))
		logger.info(train_configs)
		logger.info(model_configs)
	else:
		logger=None

	if local_rank == 0: logger.info('Load train datasets...')
	train_dataset = LoadDataset(cfgs.dataset_name, cfgs.data_path, train=True, download=True, 
		resize_size=cfgs.img_size, hdf5_path=None, random_flip=True)


	if local_rank == 0: logger.info('Train dataset size : {dataset_size}'.format(dataset_size=len(train_dataset)))
	if local_rank == 0: logger.info('Load {mode} datasets ....'.format(mode=cfgs.eval_type))

	eval_mode = True if cfgs.eval_type == 'train' else False

	eval_dataset = LoadDataset(cfgs.dataset_name, cfgs.data_path, train=eval_mode, download=True,
		resize_size=cfgs.img_size, hdf5_path=None, random_flip=True)

	if local_rank == 0: logger.info('Eval dataset size: {dataset_size}'.format(dataset_size=len(eval_dataset)))
	train_sampler = None
	cfgs.batch_size=64
	train_dataloader = DataLoader(train_dataset, batch_size=cfgs.batch_size, shuffle=True, pin_memory=True,
									num_workers=cfgs.num_workers, sampler=train_sampler, drop_last=True)
	eval_dataloader = DataLoader(eval_dataset, batch_size=cfgs.batch_size, shuffle=False, pin_memory=True,
									num_workers=cfgs.num_workers, drop_last=False)

	if local_rank == 0: logger.info('Build model...')
	module = __import__('models.{architecture}'.format(architecture=cfgs.architecture), fromlist=['something'])
	Dis = Discriminator(cfgs.img_size, cfgs.d_conv_dim, cfgs.d_spectral_norm, cfgs.attention, cfgs.attention_after_nth_dis_block,
							   cfgs.activation_fn, cfgs.conditional_strategy, cfgs.hypersphere_dim, cfgs.num_classes, cfgs.nonlinear_embed,
							   cfgs.normalize_embed, cfgs.d_init, cfgs.D_depth, False).to(local_rank)
	Gen = module.Generator(cfgs.z_dim, cfgs.shared_dim, cfgs.img_size, cfgs.g_conv_dim, cfgs.g_spectral_norm, cfgs.attention,
						   cfgs.attention_after_nth_gen_block, cfgs.activation_fn, cfgs.conditional_strategy, cfgs.num_classes,
						   cfgs.g_init, cfgs.G_depth, False).to(local_rank)
	encoder = encoder_source.Encoder(isize=32, nz=80, nc=3, ndf=64).to(local_rank)
	vae = encoder_source.VAE().to(local_rank)

	D_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, Dis.parameters()), cfgs.d_lr, [cfgs.beta1, cfgs.beta2], eps=1e-6)
	G_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, Gen.parameters()), cfgs.g_lr, [cfgs.beta1, cfgs.beta2], eps=1e-6)
	opt_encoder = torch.optim.Adam([{'params':encoder.parameters()},{'params':vae.parameters()}], cfgs.g_lr, [cfgs.beta1, cfgs.beta2], eps=1e-6)

	if cfgs.checkpoint_folder is None:
		checkpoint_dir = make_checkpoint_dir(cfgs.checkpoint_folder, run_name)
	else:
		#when = "current" if cfgs.load_current is True else "best"
		when = "best"
		if not exists(abspath(cfgs.checkpoint_folder)):
			raise NotADirectoryError
		checkpoint_dir = make_checkpoint_dir(cfgs.checkpoint_folder, run_name)
		# g_checkpoint_dir = glob.glob(join(checkpoint_dir,"model=G-{when}-weights-step*.pth".format(when=when)))[0]
		# d_checkpoint_dir = glob.glob(join(checkpoint_dir,"model=D-{when}-weights-step*.pth".format(when=when)))[0]
		# e_checkpoint_dir = glob.glob(join(checkpoint_dir,"model=E-{when}-weights-step*.pth".format(when=when)))[0]
		# v_checkpoint_dir = glob.glob(join(checkpoint_dir,"model=V-{when}-weights-step*.pth".format(when=when)))[0]
		g_checkpoint_dir = glob.glob(join(cfgs.checkpoint_folder,"model=G-{when}-weights-step*.pth".format(when=when)))[0]
		#d_checkpoint_dir = glob.glob(join(cfgs.checkpoint_folder,"model=D-{when}-weights-step*.pth".format(when=when)))[0]
		#d_checkpoint_dir = "/research/dept6/yjyang/SP2020/PyTorch-StudioGAN/cifar10_new_src/best_version1_cifar10_checkpoints/acc93.pth"
		e_checkpoint_dir = glob.glob(join(cfgs.checkpoint_folder,"model=E-{when}-weights-step*.pth".format(when=when)))[0]
		v_checkpoint_dir = glob.glob(join(cfgs.checkpoint_folder,"model=V-{when}-weights-step*.pth".format(when=when)))[0]
		#discriminator_checkpoint = torch.load(d_checkpoint_dir)
		

		#Dis.load_state_dict(discriminator_checkpoint['state_dict'])
		#D_optimizer.load_state_dict(discriminator_checkpoint['optimizer'])
		
		# for state in D_optimizer.state.values():
		# 	for k, v in state.items():
		# 		if isinstance(v, torch.Tensor):
		# 			state[k] = v.cuda()


		# step_dis = discriminator_checkpoint['step']




		Gen, G_optimizer, trained_seed, _, step, prev_ada_p = load_checkpoint(Gen, G_optimizer, g_checkpoint_dir)
		# Dis, D_optimizer, trained_seed, run_name, step, prev_ada_p, best_step, best_fid, best_fid_checkpoint_path =\
		# 	 load_checkpoint(Dis, D_optimizer, d_checkpoint_dir, metric=True)
		encoder, opt_encoder, trained_seed, _, step, prev_ada_p = load_checkpoint(encoder, opt_encoder, e_checkpoint_dir)
		vae = load_checkpoint(vae, opt_encoder, v_checkpoint_dir)

		if local_rank == 0: logger = make_logger(run_name, None)
		writer = SummaryWriter(log_dir=join('./Advtrain_mydiscriminator_logs', run_name)) if global_rank == 0 else None
		# if cfgs.train_configs['train']:
		# 	assert cfgs.seed == trained_seed, "Seed for sampling random numbers should be same!"

		if local_rank == 0: logger.info('Generator checkpoint is {}'.format(g_checkpoint_dir))
		#if local_rank == 0: logger.info('Discriminator checkpoint is {}'.format(d_checkpoint_dir))
		if local_rank == 0: logger.info('encoder checkpoint is {}'.format(e_checkpoint_dir))
		if local_rank == 0: logger.info('vae checkpoint is {}'.format(v_checkpoint_dir))


	if world_size > 1:
		Gen = DataParallel(Gen, output_device=local_rank)
		Dis = DataParallel(Dis, output_device=local_rank)
		encoder = DataParallel(encoder, output_device=local_rank)
		vae = torch.nn.DataParallel(vae, device_ids=[0])

	gen_model = Gen
	dis_model = Dis
	D_loss = loss_hinge_dis
	gen_model.eval()
	vae.eval()
	encoder.eval()

	dis_model.train()
	total_step = 200000
	if global_rank==0: logger.info('Start training....')
	step_count = 0#step_dis
	train_iter = iter(train_dataloader)
	start_time = datetime.now()
	acc_best = 0
	data_id = 0
	while step_count <= total_step:
		D_optimizer.zero_grad()
		data_id = data_id + 1
		try:
			real_images, real_labels = next(train_iter)
		except StopIteration:
			train_iter = iter(train_dataloader)
			real_images, real_labels = next(train_iter)

		real_images, real_labels = real_images.to(local_rank), real_labels.to(local_rank)
		# adding some noise to real_images
		#real_images = torch.empty_like(real_images, dtype=real_images.dtype).uniform_(-2/128.0, 2/128.0) + real_images
		img_to_classifier = (((real_images + 1)/2) - mean)/std

		acc_of_classifier = fb.utils.accuracy(fmodel, (real_images+1)/2, real_labels)
		logger.info("data_id:{}".format(data_id))
		logger.info("acc of classifier is:{}".format(acc_of_classifier))
		if data_id%2 == 0: # 50% training adversarial samples
			epsilon = torch.rand(1).to(local_rank)*0.5
			outputs = classifier(img_to_classifier.to(local_rank))
			y_classifier_ori = torch.argmax(outputs, dim=1).to(local_rank)
			attack =  fb.attacks.LinfProjectedGradientDescentAttack(steps=40)

			raw, x_adv, is_adv = attack(fmodel, ((real_images+1)/2), y_classifier_ori, epsilons=epsilon)
			#x_adv = torch.FloatTensor(x_adv).to(local_rank)
			real_images = (torch.tensor(x_adv)*2.0 -1).to(local_rank)
			#print(x_adv.type())
			real_images = (x_adv*2 -1).to(local_rank)

		else:
			# adding some noise to real_images
			real_images = torch.empty_like(real_images, dtype=real_images.dtype).uniform_(-2/128.0, 2/128.0) + real_images
		#real_images = DiffAugment(real_images, policy=policy)
		latent_i_real = encoder(real_images)
		z_mean_real, z_log_var_real, zs_real = vae(latent_i_real)

		fake_images_correct_labels = gen_model(zs_real, real_labels)
		
		if data_id%2 == 0:
			x_adv_to_classifier = ((x_adv - mean)/std)
			outputs = classifier(x_adv_to_classifier.to(local_rank))
			y_classifier = torch.argmax(outputs, dim=1)

			wrong_labels = y_classifier
		else:
			Int_Modi = random.randint(1, 9)
			wrong_labels = ((real_labels + Int_Modi) % 10).to(local_rank)

		fake_images_wrong_labels  =  gen_model(zs_real, wrong_labels)
		difference_pos = fake_images_correct_labels
		difference_neg = fake_images_wrong_labels
		dis_out_pos = dis_model(difference_pos, real_labels)
		dis_out_neg = dis_model(difference_neg, wrong_labels)
		dis_acml_loss = D_loss(dis_out_pos, dis_out_neg)*1.0

		dis_acml_loss.backward()
		D_optimizer.step()
		step_count += 1
		if step_count % 100 == 0 and global_rank == 0:
			# print(dis_acml_loss.item())
			# log_message = LOG_FORMAT.format(step=step_count,
			# 								progress=step_count/total_step,
			# 								elapsed=elapsed_time(start_time),
			# 								temperature=0,
			# 								ada_p='No',
			# 								dis_loss=dis_acml_loss.item(),
			# 								gen_loss=dis_acml_loss.item(),
			# 								)
			logger.info("Dis_loss is {dis_loss_is}".format(dis_loss_is=dis_acml_loss.item()))


			writer.add_scalars('Losses', {'discriminator': dis_acml_loss.item()}, step_count)
		if step_count % 500 == 0 or step_count == total_step:
			
			if global_rank == 0:
				when = "current"
				dis_model.eval()
				test_iter = iter(eval_dataloader)
				i = 0
				sum_pos = 0
				sum_neg = 0
				acc = 0
				acc_pos = 0
				acc_neg = 0
				length = 0
				for data in test_iter:
					#print("{i} is :".format(i=i))
					
					#i = i + 1
					#try:
					#	test_images, test_labels = next(test_iter)
					#except StopIteration:
						#test_iter = iter(eval_dataloader)
					
					test_images, test_labels = data
					length = test_labels.size(0) + length
					test_images, test_labels = test_images.to(local_rank), test_labels.to(local_rank)

					epsilon = torch.rand(1).to(local_rank)
					raw, x_adv_test, is_adv = attack(fmodel, ((test_images+1)/2), test_labels, epsilons=epsilon)

					test_images = (x_adv_test*2.0)-1.0

					latent_i_test = encoder(test_images)
					z_mean_real, z_log_var_real, zs_real = vae(latent_i_test)

					fake_images_correct_labels = gen_model(zs_real, test_labels)
					#Int_Modi = random.randint(1, 9)
					#wrong_labels = ((test_labels + Int_Modi) % 10).to(local_rank)
					x_adv_test_to_classifier = ((x_adv_test - mean)/std)
					outputs_test = classifier(x_adv_test_to_classifier.to(local_rank))
					y_classifier = torch.argmax(outputs_test, dim=1)
					wrong_labels = y_classifier

					fake_images_wrong_labels  =  gen_model(zs_real, wrong_labels)

					difference_pos = fake_images_correct_labels
					difference_neg = fake_images_wrong_labels
					dis_out_pos = dis_model(difference_pos, test_labels)
					dis_out_neg = dis_model(difference_neg, wrong_labels)
					decision_pos = dis_out_pos > 0
					#print(decision_pos)
					decision_neg = dis_out_neg < 0
					
					acc_pos += torch.sum(decision_pos)
					acc_neg += torch.sum(decision_neg)

					acc += (torch.sum(decision_pos) + torch.sum(decision_neg))
					#print(torch.sum(decision_pos))
				accuracy_pos = acc_pos/(length*1.0)
				accuracy_neg = acc_neg/(length*1.0)

				accuracy = acc / (length*2.0)
				logger.info("accuracy is {accuracy}".format(accuracy=accuracy.item()))
				logger.info("pos___acc is {accuracy_pos}".format(accuracy_pos=accuracy_pos.item()))
				logger.info("neg___acc is {accuracy_neg}".format(accuracy_neg=accuracy_neg.item()))
				if accuracy > acc_best:
					acc_best = accuracy

					logger.info("best_acc is {acc_best}".format(acc_best=acc_best.item()))
					if isinstance(dis_model, DataParallel) or isinstance(dis_model, DistributedDataParallel):
						dis = dis_model.module
					else:
						dis = dis_model
					d_states = {'run_name':run_name, 'step':step_count, 'state_dict':dis.state_dict(), 'optimizer':D_optimizer.state_dict()}
					if len(glob.glob(join(checkpoint_dir,"advtrain_pn_best_batchsize64_acc=D-{when}-weights-step*.pth".format(when=when)))) >= 1:
						find_and_remove(glob.glob(join(checkpoint_dir,"advtrain_pn_best_batchsize64_acc=D-{when}-weights-step*.pth".format(when=when)))[0])
					d_checkpoint_output_path = join(checkpoint_dir, "advtrain_pn_best_batchsize64_acc=D-{when}-weights-step={step}acc={acc_best}.pth".format(when=when, step=str(step_count), acc_best=acc_best.item()))
					torch.save(d_states, d_checkpoint_output_path)
					if global_rank==0: logger.info("Save best model to {}".format(checkpoint_dir))



				if isinstance(dis_model, DataParallel) or isinstance(dis_model, DistributedDataParallel):
					dis = dis_model.module
				else:
					dis = dis_model
				d_states = {'run_name':run_name, 'step':step_count, 'state_dict':dis.state_dict(), 'optimizer':D_optimizer.state_dict()}
				if len(glob.glob(join(checkpoint_dir,"advtrain_pn_old_model_batchsize64_acc=D-{when}-weights-step*.pth".format(when=when)))) >= 1:
					find_and_remove(glob.glob(join(checkpoint_dir,"advtrain_pn_old_model_batchsize64_acc=D-{when}-weights-step*.pth".format(when=when)))[0])
				d_checkpoint_output_path = join(checkpoint_dir, "advtrain_pn_old_model_batchsize64_acc=D-{when}-weights-step={step}.pth".format(when=when, step=str(step_count)))
				torch.save(d_states, d_checkpoint_output_path)
				if global_rank==0: logger.info("Save model to {}".format(checkpoint_dir))
				dis_model.train()






def save(step, is_best):
	when = "best" if is_best is True else "current"
	dis_model.eval()
	if isinstance(dis_model, DataParallel) or isinstance(dis_model, DistributedDataParallel):
		dis = dis_model.module
	else:
		dis = dis_model
	d_states = {'seed':seed, 'run_name':run_name, 'step':step_count, 'state_dict':dis.state_dict(), 'optimizer':D_optimizer.state_dict()}
	if len(glob.glob(join(checkpoint_dir,"model=G-{when}-weights-step*.pth".format(when=when)))) >= 1:
		find_and_remove(glob.glob(join(checkpoint_dir,"model=D-{when}-weights-step*.pth".format(when=when)))[0])
	d_checkpoint_output_path = join(checkpoint_dir, "model=D-{when}-weights-step={step}.pth".format(when=when, step=str(step_count)))
	torch.save(d_states, d_checkpoint_output_path)
	if global_rank==0: logger.info("Save model to {}".format(checkpoint_dir))
	dis_model.train()
	



		

if __name__=='__main__':
	main()


