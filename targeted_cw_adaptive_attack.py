import misc.utils as utils
from misc.load_dataset import LoadDataset
import models.MobileNetV2 as MobileNet
from models.resnet import *
from datetime import datetime
from torch.utils.data import DataLoader
import argparse
import json
import os
import torch
import torch.nn as nn
import glob
import models.GANv2 as GANv2
import torchvision
from utils.log import make_checkpoint_dir, make_logger
import tensorflow as tf
from densenet import densenet169
#from tensorboardX import SummaryWriter
import lib.pytorch_ssim as ssim_package
import numpy
from utilss.utils import rm_dir, cuda, where
import torch.optim as optim
import foolbox as fb 
import time

import warnings
warnings.filterwarnings("ignore")

import misc.utils as utils
from misc.load_dataset import LoadDataset
import models.MobileNetV2 as MobileNet
from models.resnet import *
from datetime import datetime
from torch.utils.data import DataLoader
import argparse
import json
import os
import torch
import torch.nn as nn
import glob
import models.GANv2 as GANv2
import torchvision
from utils.log import make_checkpoint_dir, make_logger
#import tensorflow as tf
from densenet import densenet169
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
#from tensorboardX import SummaryWriter
import lib.pytorch_ssim as ssim_package
import numpy
from utilss.utils import rm_dir, cuda, where
import torch.optim as optim
#import foolbox as fb 
import time
from utils.model_ops import *
from adaptiveCWattackl2 import CarliniWagnerL2Attack
torch.manual_seed(0)
import random
random.seed(0)
numpy.random.seed(0)



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
		# with torch.cuda.amp.autocast() if self.mixed_precision is True and evaluation is False else dummy_context_mgr() as mp:
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

def sample_targetlabel(y, num_classes=10):
    y_target = torch.randint_like(y, 0, 10).cuda()
    index = torch.where(y_target == y)[0]
    while index.size(0)!= 0:
        y_target_new = torch.randint(0, 10, (index.size(0),)).cuda()
        y_target[index] = y_target_new
        index = torch.where(y_target == y)[0]
    return y_target

def test(orimodel, genmodel, mlpModel, device, dataset, encoder, vae, gen,
		 inference_m, writer, adaptive_loss, test_sample_number):
	checkpoint_path = './'
	classifier = densenet169().to(device)

	classifier_path = os.path.join(checkpoint_path,"densenet169.pt")
	classifier_ckpt = torch.load(classifier_path)
	classifier.load_state_dict(classifier_ckpt)


	classifier.eval().to(device)


	mean = [0.4914, 0.4822, 0.4465]
	std = [0.2023, 0.1994, 0.2010]
	mean =torch.tensor(mean, dtype=torch.float32).cuda()
	std =  torch.tensor(std, dtype=torch.float32).cuda()
	mean = mean[:,None, None]
	std = std[:, None, None]
	
	bounds = (0, 1)
	preprocessing = dict(mean = [0.4914, 0.4822, 0.4465], std = [0.2023, 0.1994, 0.2010],  axis=-3)
	fmodel = fb.PyTorchModel(classifier, bounds=bounds, preprocessing=preprocessing)


	global test_iter
	mlpModel.eval().to(device)
	orimodel.eval().to(device)

	genmodel.eval().to(device)
	orimodel.query_features()
	genmodel.query_features()

	ssim_size_average = lambda x, y: ssim_package.ssim(x, y, size_average=False)
	classifier_criterion = nn.CrossEntropyLoss()
	mlploss = nn.CrossEntropyLoss()
	pred_list, pos_pred_list, neg_pred_list = [], [], []
	data_id = 0

	# parameters for cw attack
	# parameters for cw attack
	c = 1#1e-4 
	max_iter = 100
	learning_rate = 0.01
	data_id = 0
	acc_batch = [0,0,0,0,0,0,0,0,0,0,0]
	tested_number = 0
	acc_cw = 0
	date = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()) 
	logger = make_logger(date+"l2_ssim_dis_dml_adaptive_cw_implementation_torchattack", None)

	Dis = Discriminator(cfgs.img_size, cfgs.d_conv_dim, cfgs.d_spectral_norm, cfgs.attention, cfgs.attention_after_nth_dis_block,
					cfgs.activation_fn, cfgs.conditional_strategy, cfgs.hypersphere_dim, cfgs.num_classes, cfgs.nonlinear_embed,
					cfgs.normalize_embed, cfgs.d_init, cfgs.D_depth, False).to(device)
	d_checkpoint_dir ='./pretrain' 
	denoisecGAN_adding_noise_adv_best = '/dis.pth' 
	d_checkpoint_dir = d_checkpoint_dir + denoisecGAN_adding_noise_adv_best
	discriminator_checkpoint = torch.load(d_checkpoint_dir)
	Dis.load_state_dict(discriminator_checkpoint['state_dict'])
	dis_model = Dis

	dis_model.eval()



	
	for img, classId in dataset:
		
		if data_id * len(img) > test_sample_number:
			output_acc_name = "./cwadaptivenpy/"+date +"_adaptive_cw_targeted_loss==" + adaptive_loss
			acc = 1 - (numpy.array(acc_cw.cpu())/tested_number)
			numpy.save(output_acc_name, acc)
			numpy.save("./cwadaptivenpy/"+"adaptive_cw_targeted_loss=="+adaptive_loss, acc)
			logger.info("acc is saved as {}.npy and {}.npy".format(output_acc_name, "adaptive_cw_targeted_loss__"+adaptive_loss))
			break 
		data_id = data_id + 1
		img, img_pos, img_neg, class_id, wrong_y = utils.generate1(
			img, classId, device, encoder, vae, gen, next=True)
		img = (img + 1.0)/2.0

		x_adv_image = torch.zeros(10, 3, 32, 32)
		fake_adv_image = torch.zeros(10, 3, 32, 32)

		img_to_classifier = (img - mean)/std
		#acc_of_classifier = fb.utils.accuracy(fmodel, (img+1)/2, class_id)
		acc_of_classifier = fb.utils.accuracy(fmodel, img, class_id)
		logger.info("acc of classifier is:{}".format(acc_of_classifier))
		# kappa is the confidence of cw attack
		kappa_list = [0]#[0, 0.01, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3, 0.4]

		outputs = classifier(img_to_classifier.to(device))

		y_classifier_ori = torch.argmax(outputs, dim=1)
		logger.info("original classifier prediction:{}".format(y_classifier_ori))

		tested_number += len(img)

		CWattack =  CarliniWagnerL2Attack
		model = classifier
		cw_adversary = CWattack(model, encoder, vae, gen, orimodel, genmodel, mlpModel, dis_model,
					batchsize=len(img), num_classes=10,
					confidence=0, targeted=True, learning_rate=0.05,
					binary_search_steps=9, max_iterations=1000, abort_early=True,
					initial_const=0.01, clip_min=0.0, clip_max=1.0,
					loss_fn=None, normalize_fn=None, adaptive_loss=adaptive_loss)
					

		y_target = sample_targetlabel(class_id, num_classes=10)
		logger.info("targeted y                    :{}".format(y_target.data))
	
		adv_input = cw_adversary.perturb(img, y_target)

		adv_input_scale = adv_input * 2 -  1  # belongs to [-1,1]

		outputs = classifier(((adv_input-mean)/std).to(device))
		
		# justify whether attack is successful
		y_classifier = torch.argmax(outputs, dim=1)
		logger.info("adv__img classifier prediction:{}".format(y_classifier))
		logger.info("targeted_y==adv_img_y         :{}".format((y_classifier==y_target)*1))
		logger.info("all_targeted successful:{}".format(torch.sum(y_classifier==y_target)))
		logger.info("all_incorrect successful:{}".format(torch.sum(y_classifier!=class_id)))


		latent_i = encoder(adv_input_scale.to(device))
		z_mean, z_log_var, z = vae(latent_i.to(device))
				
		fake = gen(z.to(device), y_classifier)

		feat_emb, _ = orimodel(adv_input_scale)
		feat_pos, _ = genmodel(fake)
		pos_pair = torch.cat([feat_emb, feat_pos], dim=1)
		y_mlp_pos = torch.ones(len(img)).long().to(device)#target attack

		pos_pred = mlpModel((pos_pair, y_classifier))
		dis_threshold = -1.334
		dml_threshold = 0.0107
		ssim_threshold = 0.0551
		
		dis_out_fake = dis_model(fake, y_classifier)

		pred_y = (pos_pred[:, 1] > dml_threshold).long()
		dml_decision = pred_y == 1
		logger.info("dml_decision:{}".format(dml_decision*1))
		ssim_decision = (ssim_size_average(fake, img) > ssim_threshold)
		
		
		dis_decision =  (dis_out_fake > dis_threshold)*1 == 1
		logger.info("dis_decision:{}".format(dis_decision*1))

		judgement_equation = (y_classifier!=class_id)*(y_classifier == y_target)*(ssim_decision)*(dml_decision)*(dis_decision)
		logger.info("all_decision:{}".format(judgement_equation*1))
		acc_cw += torch.sum(judgement_equation)
		logger.info(">>>>>total_tested sample number:{}".format(tested_number))
		logger.info("=====cw_adaptive_targeted_attack_successful_rate:{}=========".format(acc_cw/tested_number))
		

	return acc_cw


test_iter = 0
train_iter = 0

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Train MoCo on GTSRB')
	parser.add_argument('-a', '--arch', default='MobileNetV2', type=str)
	parser.add_argument('-c', '--config_path', type=str,
						default='./config/configsCifar10.json')
	parser.add_argument('--tag', default='', type=str)
	# lr: 0.06 for batch 512 (or 0.03 for batch 256)
	parser.add_argument(
		'--lr', '--learning-rate', default=0.06, type=float, metavar='LR',
		help='initial learning rate', dest='lr')
	parser.add_argument(
		'--drop_p', default=0.1, type=float, metavar='LR',
		help='MLP drop rate', dest='drop_p')
	parser.add_argument(
		'--epochs', default=100, type=int, metavar='N',
		help='number of total epochs to run')
	parser.add_argument(
		'--feature_m', action='store_true', help='use the minus of inputs')
	parser.add_argument(
		'--cond', action='store_true',
		help='After MLP pretrain, train end-to-end')
	parser.add_argument(
		'--device_id', default=[], nargs='*', type=int, help='cuda device ids')
	parser.add_argument('--cos', default=-1, type=int,
						help='use cosine lr schedule')

	parser.add_argument('--batch_size', default=64, type=int,
						metavar='N', help='mini-batch size')
	parser.add_argument('--wd', default=5e-4, type=float,
						metavar='W', help='weight decay')
	parser.add_argument('-l','--adaptive_loss', default='all', type=str, help="all,all_lambda, ssim_dis_dml, dis_dml, ssim_dis, ssim_dml, dis, dml, ssim")
	parser.add_argument('--test_sample_number', default=1000, type=int, help='1000')
	  
	# utils
	parser.add_argument(
		'--resume', default='./pretrain/MobileNetV2_91V97.48.pth', type=str, metavar='PATH',
		help='path to latest checkpoint (default: none)')
	parser.add_argument(
		'--results_dir', default='', type=str, metavar='PATH',
		help='path to cache (default: none)')

	args = parser.parse_args()
	if args.cos == -1:
		args.cos = args.epochs
	if args.config_path is not None:
		with open(args.config_path) as f:
			model_configs = json.load(f)
		train_configs = vars(args)
	else:
		raise NotImplementedError
	cfgs = utils.dict2clsattr(train_configs, model_configs)
	args.dataset_name = cfgs.dataset_name
	args.data_path = cfgs.data_path
	args.img_size = cfgs.img_size
	if args.dataset_name == "cifar10":
		args.class_num = 10
		args.feature_num = 12
	elif args.dataset_name == "cifar100":
		args.class_num = 100
		args.feature_num = 120

	# policy = "color,translation,cutout"
	policy = ""
	#print("Using data augmentation policy: {}".format(policy))
	assert args.arch in ["MobileNetV2"]
	args.tag += cfgs.dataset_name
	if args.cond:
		args.tag += "Cond"
	if args.results_dir == '':
		if args.tag != '':
			args.results_dir = './results/{}-{}-'.format(args.arch, args.tag) + \
				datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
		else:
			args.results_dir = './results/{}-'.format(args.arch) + \
				datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

	#print(args)
	device = torch.device("cuda")
	#print(' prepared models...')
	# initialize models.
	gen = GANv2.Generator(
		cfgs.z_dim, cfgs.shared_dim, cfgs.img_size, cfgs.g_conv_dim, cfgs.
		g_spectral_norm, cfgs.attention, cfgs.attention_after_nth_gen_block,
		cfgs.activation_fn, cfgs.conditional_strategy, cfgs.num_classes, cfgs.
		g_init, cfgs.G_depth, False)
	encoder = GANv2.Encoder(isize=32, nz=80, nc=3, ndf=64)
	vae = GANv2.VAE()

	gen.load_state_dict(torch.load(cfgs.G_weights)['state_dict'])
	encoder.load_state_dict(torch.load(cfgs.E_weights)['state_dict'])
	vae.load_state_dict(torch.load(cfgs.V_weights)['state_dict'])

	gen.eval().cuda()
	encoder.eval().cuda()
	vae.eval().cuda()

	#print(' prepared dataset...')
	train_data = LoadDataset(
		args.dataset_name, args.data_path, train=True, download=False,
		resize_size=args.img_size, hdf5_path=None, random_flip=True)
	test_data = LoadDataset(
		args.dataset_name, args.data_path, train=False, download=False,
		resize_size=args.img_size, hdf5_path=None, random_flip=False)

	train_loader = DataLoader(
		train_data, batch_size=args.batch_size, shuffle=True, num_workers=4,
		pin_memory=True, drop_last=True)
	test_loader = DataLoader(
		test_data, batch_size=args.batch_size, shuffle=False, num_workers=4,
		pin_memory=True)

	print(' preparing utils...')
	if args.feature_m:
		inference_m = utils.inference_mlp_m
	else:
		inference_m = utils.inference_mlp

	mlploss = nn.CrossEntropyLoss()
	device_ids = args.device_id
	orimodel = eval("MobileNet.{}(n_class={})".format(
		args.arch, args.feature_num))
	orimodel = orimodel.cuda()
	genmodel = eval("MobileNet.{}(n_class={})".format(
		args.arch, args.feature_num))
	genmodel = genmodel.cuda()



	if args.feature_m:
		mlpModel = MobileNet.MLP(
			orimodel.last_channel, p=args.drop_p, class_num=args.class_num)
	else:
		mlpModel = MobileNet.MLP(
			orimodel.last_channel * 2, p=args.drop_p, class_num=args.class_num)
	mlpModel = mlpModel.cuda()
	# define optimizer
	optimizer = torch.optim.SGD(
		mlpModel.parameters(), lr=args.lr, weight_decay=args.wd, momentum=0.9)

	# load model if resume
	epoch_start = 1
	if args.resume is not '':
		checkpoint = torch.load(args.resume)
		mlpModel.load_state_dict(checkpoint['mlp_state'])
		orimodel.load_state_dict(checkpoint['ori_state'])
		genmodel.load_state_dict(checkpoint['gen_state'])
		if optimizer is not None:
			optimizer.load_state_dict(checkpoint['optimizer'])
			optimizer_loaded = True
		else:
			optimizer_loaded = False
		epoch_start = checkpoint['epoch'] + 1
		print('Loaded from: {}'.format(args.resume))
	if optimizer_loaded and not args.cond:
		lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
			optimizer, args.cos, eta_min=1e-6,
			#last_epoch=epoch_start - 1
		)
	else:
		lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
			optimizer, args.cos, eta_min=1e-6
		)
	if args.cond:
		epoch_start = 1
		assert args.resume
		ori_Optimizer = torch.optim.SGD(
			orimodel.parameters(),
			lr=args.lr, weight_decay=args.wd, momentum=0.9)

		gen_Optimizer = torch.optim.SGD(
			genmodel.parameters(),
			lr=args.lr, weight_decay=args.wd, momentum=0.9)
		ori_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
			ori_Optimizer, args.epochs - epoch_start, eta_min=1e-6
		)
		gen_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
			gen_Optimizer, args.epochs - epoch_start, eta_min=1e-6
		)
		cond_optim = [ori_Optimizer, gen_Optimizer]
	else:
		ori_lr_scheduler = None
		gen_lr_scheduler = None
		cond_optim = None

	if len(device_ids) > 0:
		gen = torch.nn.DataParallel(gen, device_ids)
		encoder = torch.nn.DataParallel(encoder, device_ids)
		vae = torch.nn.DataParallel(vae, device_ids)
		orimodel = torch.nn.DataParallel(orimodel, device_ids)
		genmodel = torch.nn.DataParallel(genmodel, device_ids)
		mlpModel = torch.nn.DataParallel(mlpModel, device_ids)
		orimodel.query_features = orimodel.module.query_features
		genmodel.query_features = genmodel.module.query_features

	# logging
	results = {'train_loss': [], 'test_acc@1': []}
	if not os.path.exists(args.results_dir):
		os.mkdir(args.results_dir)
	# dump args
	with open(args.results_dir + '/args.json', 'w') as fid:
		json.dump(args.__dict__, fid, indent=2)

	writer = SummaryWriter(os.path.join(args.results_dir, "tensorboard"))
	# training loop
	best_prec_at_1 = 0
	train_iter = epoch_start * len(train_data) // args.batch_size
	test_iter = epoch_start
	# if args.cond:
	best_prec_at_1 = test(
			orimodel, genmodel, mlpModel, device, test_loader, encoder, vae,
			gen, inference_m, writer, args.adaptive_loss, args.test_sample_number)
