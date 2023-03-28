
import glob
import os
import random
from os.path import dirname, abspath, exists, join
from torchlars import LARS

from data_utils.load_dataset import *
from metrics.inception_network import InceptionV3
from metrics.prepare_inception_moments import prepare_inception_moments
from utils.log import make_checkpoint_dir, make_logger
from utils.losses import *
from utils.load_checkpoint import load_checkpoint
from utils.misc import *
from utils.biggan_utils import ema, ema_DP_SyncBN
from sync_batchnorm.batchnorm import convert_model
from invencoder_worker import make_worker
from models import invencoder_resnet as module
import torch
from torch.utils.data import DataLoader
from torch.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel as DDP
from tensorboardX import SummaryWriter
#import encoder as encoder_source



def prepare_train_eval(local_rank, gpus_per_node, world_size, run_name, train_configs, model_configs, hdf5_path_train):
	cfgs = dict2clsattr(train_configs, model_configs)
	prev_ada_p, step, best_step, best_fid, best_fid_checkpoint_path, mu, sigma, inception_model = None, 0, 0, None, None, None, None, None
	run_name__ = run_name
	if cfgs.distributed_data_parallel:
		global_rank = cfgs.nr*(gpus_per_node) + local_rank
		print("Use GPU: {} for training.".format(global_rank))
		setup(global_rank, world_size)
		torch.cuda.set_device(local_rank)
	else:
		global_rank = local_rank

	writer = SummaryWriter(log_dir=join('./invencoder_logs', run_name)) if local_rank == 0 else None
	if local_rank == 0:
		logger = make_logger(run_name, None)
		logger.info('Run name : {run_name}'.format(run_name=run_name))
		logger.info(train_configs)
		logger.info(model_configs)
	else:
		logger = None

	##### load dataset #####
	if local_rank == 0: logger.info('Load train datasets...')
	train_dataset = LoadDataset(cfgs.dataset_name, cfgs.data_path, train=True, download=True, resize_size=cfgs.img_size,
								hdf5_path=hdf5_path_train, random_flip=cfgs.random_flip_preprocessing)
	if cfgs.reduce_train_dataset < 1.0:
		num_train = int(cfgs.reduce_train_dataset*len(train_dataset))
		train_dataset, _ = torch.utils.data.random_split(train_dataset, [num_train, len(train_dataset) - num_train])
	if local_rank == 0: logger.info('Train dataset size : {dataset_size}'.format(dataset_size=len(train_dataset)))

	if local_rank == 0: logger.info('Load {mode} datasets...'.format(mode=cfgs.eval_type))
	eval_mode = True if cfgs.eval_type == 'train' else False
	eval_dataset = LoadDataset(cfgs.dataset_name, cfgs.data_path, train=eval_mode, download=True, resize_size=cfgs.img_size,
							   hdf5_path=None, random_flip=False)
	if local_rank == 0: logger.info('Eval dataset size : {dataset_size}'.format(dataset_size=len(eval_dataset)))

	if cfgs.distributed_data_parallel:
		train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
		cfgs.batch_size = cfgs.batch_size//world_size
	else:
		train_sampler = None

	train_dataloader = DataLoader(train_dataset, batch_size=cfgs.batch_size, shuffle=(train_sampler is None), pin_memory=True,
								  num_workers=cfgs.num_workers, sampler=train_sampler, drop_last=True)
	eval_dataloader = DataLoader(eval_dataset, batch_size=cfgs.batch_size, shuffle=False, pin_memory=True, num_workers=cfgs.num_workers, drop_last=False)

	##### build model #####
	if local_rank == 0: logger.info('Build model...')
	#module = __import__('models.{architecture}'.format(architecture=cfgs.architecture), fromlist=['something'])
	if local_rank == 0: logger.info('Modules are located on models.{architecture}.'.format(architecture=cfgs.architecture))
	Gen = module.Generator(cfgs.z_dim, cfgs.shared_dim, cfgs.img_size, cfgs.g_conv_dim, cfgs.g_spectral_norm, cfgs.attention,
						   cfgs.attention_after_nth_gen_block, cfgs.activation_fn, cfgs.conditional_strategy, cfgs.num_classes,
						   cfgs.g_init, cfgs.G_depth, cfgs.mixed_precision).to(local_rank)

	Dis = module.Discriminator(cfgs.img_size, cfgs.d_conv_dim, cfgs.d_spectral_norm, cfgs.attention, cfgs.attention_after_nth_dis_block,
							   cfgs.activation_fn, cfgs.conditional_strategy, cfgs.hypersphere_dim, cfgs.num_classes, cfgs.nonlinear_embed,
							   cfgs.normalize_embed, cfgs.d_init, cfgs.D_depth, cfgs.mixed_precision).to(local_rank)

	#d2Dis = module.Discriminator(cfgs.img_size, cfgs.d_conv_dim, cfgs.d_spectral_norm, cfgs.attention, cfgs.attention_after_nth_dis_block,
								#cfgs.activation_fn, cfgs.conditional_strategy, cfgs.hypersphere_dim, cfgs.num_classes, cfgs.nonlinear_embed,
								#cfgs.normalize_embed, cfgs.d_init, cfgs.D_depth, cfgs.mixed_precision).to(local_rank)

	#encoder = encoder_source.Encoder(isize=32, nz=80, nc=3, ndf=64).to(local_rank)
	encoder = module.invencoder(cfgs.img_size, cfgs.d_conv_dim, cfgs.d_spectral_norm, cfgs.attention, cfgs.attention_after_nth_dis_block,
							   cfgs.activation_fn, cfgs.conditional_strategy, cfgs.hypersphere_dim, cfgs.num_classes, cfgs.nonlinear_embed,
							   cfgs.normalize_embed, cfgs.d_init, cfgs.D_depth, cfgs.mixed_precision).to(local_rank)

	
	#vae = encoder_source.VAE().to(local_rank)
	

	if cfgs.ema:
		if local_rank == 0: logger.info('Prepare EMA for G with decay of {}.'.format(cfgs.ema_decay))
		Gen_copy = module.Generator(cfgs.z_dim, cfgs.shared_dim, cfgs.img_size, cfgs.g_conv_dim, cfgs.g_spectral_norm, cfgs.attention,
									cfgs.attention_after_nth_gen_block, cfgs.activation_fn, cfgs.conditional_strategy, cfgs.num_classes,
									initialize=False, G_depth=cfgs.G_depth, mixed_precision=cfgs.mixed_precision).to(local_rank)
		if not cfgs.distributed_data_parallel and world_size > 1 and cfgs.synchronized_bn:
			Gen_ema = ema_DP_SyncBN(Gen, Gen_copy, cfgs.ema_decay, cfgs.ema_start)
		else:
			Gen_ema = ema(Gen, Gen_copy, cfgs.ema_decay, cfgs.ema_start)
	else:
		Gen_copy, Gen_ema = None, None

	if local_rank == 0: logger.info(count_parameters(Gen))
	if local_rank == 0: logger.info(Gen)

	if local_rank == 0: logger.info(count_parameters(Dis))
	if local_rank == 0: logger.info(Dis)


	if local_rank == 0: logger.info(count_parameters(encoder))
	if local_rank == 0: logger.info(encoder)

	# if local_rank == 0: logger.info(count_parameters(d2Dis))
	# if local_rank == 0: logger.info(d2Dis)

	### define loss functions and optimizers
	G_loss = {'vanilla': loss_dcgan_gen, 'least_square': loss_lsgan_gen, 'hinge': loss_hinge_gen, 'wasserstein': loss_wgan_gen}
	D_loss = {'vanilla': loss_dcgan_dis, 'least_square': loss_lsgan_dis, 'hinge': loss_hinge_dis, 'wasserstein': loss_wgan_dis}
	#d2D_loss = loss_hinge_dis

	if cfgs.optimizer == "SGD":
		G_optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, Gen.parameters()), cfgs.g_lr, momentum=cfgs.momentum, nesterov=cfgs.nesterov)
		D_optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, Dis.parameters()), cfgs.d_lr, momentum=cfgs.momentum, nesterov=cfgs.nesterov)
	elif cfgs.optimizer == "RMSprop":
		G_optimizer = torch.optim.RMSprop(filter(lambda p: p.requires_grad, Gen.parameters()), cfgs.g_lr, momentum=cfgs.momentum, alpha=cfgs.alpha)
		D_optimizer = torch.optim.RMSprop(filter(lambda p: p.requires_grad, Dis.parameters()), cfgs.d_lr, momentum=cfgs.momentum, alpha=cfgs.alpha)
	elif cfgs.optimizer == "Adam":
		G_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, Gen.parameters()), cfgs.g_lr, [cfgs.beta1, cfgs.beta2], eps=1e-6)

		D_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, Dis.parameters()), cfgs.d_lr, [cfgs.beta1, cfgs.beta2], eps=1e-6)

		#d2D_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, d2Dis.parameters()), cfgs.d_lr, [cfgs.beta1, cfgs.beta2], eps=1e-6)

		#opt_encoder = torch.optim.Adam([{'params':encoder.parameters()},{'params':vae.parameters()}], cfgs.g_lr, [cfgs.beta1, cfgs.beta2], eps=1e-6)
		opt_encoder = torch.optim.Adam(filter(lambda p: p.requires_grad, encoder.parameters()), cfgs.d_lr, [cfgs.beta1, cfgs.beta2], eps=1e-6)
	else:
		raise NotImplementedError

	if cfgs.LARS_optimizer:
		G_optimizer = LARS(optimizer=G_optimizer, eps=1e-8, trust_coef=0.001)
		D_optimizer = LARS(optimizer=D_optimizer, eps=1e-8, trust_coef=0.001)

	##### load checkpoints if needed #####
	checkpoint_dir_save = make_checkpoint_dir(None, run_name) # the path used to save the models
	
	if cfgs.checkpoint_folder is None:
		checkpoint_dir = make_checkpoint_dir(cfgs.checkpoint_folder, run_name)
	else:
		when = "current" if cfgs.load_current is True else "best"
		if not exists(abspath(cfgs.checkpoint_folder)):
			raise NotADirectoryError
		checkpoint_dir = make_checkpoint_dir(cfgs.checkpoint_folder, run_name)
		g_checkpoint_dir = glob.glob(join(checkpoint_dir,"model=G-{when}-weights-step*.pth".format(when=when)))[0]
		d_checkpoint_dir = glob.glob(join(checkpoint_dir,"model=D-{when}-weights-step*.pth".format(when=when)))[0]
		e_checkpoint_dir = './checkpoints/invencoder_config-train-2021_03_30_15_42_57/model=E-current-weights-step=100000.pth'#glob.glob(join(checkpoint_dir,"model=E-{when}-weights-step*.pth".format(when=when)))[0]
		#v_checkpoint_dir = glob.glob(join(checkpoint_dir,"model=V-{when}-weights-step*.pth".format(when=when)))[0]
		#d2d_checkpoint_dir = glob.glob(join(checkpoint_dir,"model=2D-{when}-weights-step*.pth".format(when=when)))[0]
		#d2d_checkpoint_dir = "/research/dept6/yjyang/SP2020/PyTorch-StudioGAN/cifar10_new_src/checkpoints/diff_best_batchsize64_acc=D-current-weights-step=139500acc=0.9378999471664429.pth"
		
		Gen, G_optimizer, trained_seed, run_name, step, prev_ada_p = load_checkpoint(Gen, G_optimizer, g_checkpoint_dir)
		
		Dis, D_optimizer, trained_seed, run_name, step, prev_ada_p, best_step, best_fid, best_fid_checkpoint_path =\
			load_checkpoint(Dis, D_optimizer, d_checkpoint_dir, metric=True)
		encoder, opt_encoder, trained_seed, run_name, step, prev_ada_p, best_step, best_fid, best_fid_checkpoint_path =\
			load_checkpoint(encoder, opt_encoder, e_checkpoint_dir, metric=True)
		#d2Dis, d2D_optimizer, trained_seed, run_name, step, prev_ada_p, best_step, best_fid, best_fid_checkpoint_path =\
		#	load_checkpoint(2Dis, D_optimizer, d_checkpoint_dir, metric=True)
		#d2discriminator_checkpoint = torch.load(d2d_checkpoint_dir)
		

		#d2Dis.load_state_dict(d2discriminator_checkpoint['state_dict'])
		#d2D_optimizer.load_state_dict(d2discriminator_checkpoint['optimizer'])
		
		# for state in d2D_optimizer.state.values():
		# 	for k, v in state.items():
		# 		if isinstance(v, torch.Tensor):
		# 			state[k] = v.cuda()

		# encoder, opt_encoder, trained_seed, run_name, step, prev_ada_p = load_checkpoint(encoder, opt_encoder, e_checkpoint_dir)
		# vae =  load_checkpoint(vae, opt_encoder, v_checkpoint_dir)

		#if local_rank == 0: logger = make_logger(run_name, None)
		cfgs.ema = False
		if cfgs.ema:
			g_ema_checkpoint_dir = glob.glob(join(checkpoint_dir, "model=G_ema-{when}-weights-step*.pth".format(when=when)))[0]
			Gen_copy = load_checkpoint(Gen_copy, None, g_ema_checkpoint_dir, ema=True)
			Gen_ema.source, Gen_ema.target = Gen, Gen_copy

		writer = SummaryWriter(log_dir=join('./invencoder_logs', run_name__)) if global_rank == 0 else None
		cfgs.seed = trained_seed
		if cfgs.train_configs['train']:
			assert cfgs.seed == trained_seed, "Seed for sampling random numbers should be same!"

		if local_rank == 0: logger.info('Generator checkpoint is {}'.format(g_checkpoint_dir))
		if local_rank == 0: logger.info('Discriminator checkpoint is {}'.format(d_checkpoint_dir))
		if local_rank == 0: logger.info('encoder checkpoint is {}'.format(e_checkpoint_dir))
		
		#if local_rank == 0: logger.info('vae checkpoint is {}'.format(v_checkpoint_dir))
		#if local_rank == 0: logger.info('2Discriminator checkpoint is {}'.format(d2d_checkpoint_dir))

		if cfgs.freeze_layers > -1 :
			prev_ada_p, step, best_step, best_fid, best_fid_checkpoint_path = None, 0, 0, None, None


	##### wrap models with DP and convert BN to Sync BN #####
	if world_size > 1:
		if cfgs.distributed_data_parallel:
			if cfgs.synchronized_bn:
				process_group = torch.distributed.new_group([w for w in range(world_size)])
				Gen = torch.nn.SyncBatchNorm.convert_sync_batchnorm(Gen, process_group)
				Dis = torch.nn.SyncBatchNorm.convert_sync_batchnorm(Dis, process_group)
				if cfgs.ema:
					Gen_copy = torch.nn.SyncBatchNorm.convert_sync_batchnorm(Gen_copy, process_group)

			Gen = DDP(Gen, device_ids=[local_rank])
			Dis = DDP(Dis, device_ids=[local_rank])
			if cfgs.ema:
				Gen_copy = DDP(Gen_copy, device_ids=[local_rank])
		else:
			Gen = DataParallel(Gen, output_device=local_rank)
			Dis = DataParallel(Dis, output_device=local_rank)
			# d2Dis = DataParallel(d2Dis, output_device=local_rank)
			encoder = DataParallel(encoder, output_device=local_rank)
			# vae = torch.nn.DataParallel(vae, device_ids=[0])
			if cfgs.ema:
				Gen_copy = DataParallel(Gen_copy, output_device=local_rank)

			if cfgs.synchronized_bn:
				Gen = convert_model(Gen).to(local_rank)
				Dis = convert_model(Dis).to(local_rank)
				if cfgs.ema:
					Gen_copy = convert_model(Gen_copy).to(local_rank)

	##### load the inception network and prepare first/secend moments for calculating FID #####
	if cfgs.eval:
		inception_model = InceptionV3().to(local_rank)
		if world_size > 1 and cfgs.distributed_data_parallel:
			toggle_grad(inception_model, on=True)
			inception_model = DDP(inception_model, device_ids=[local_rank], broadcast_buffers=False, find_unused_parameters=True)
		elif world_size > 1 and cfgs.distributed_data_parallel is False:
			inception_model = DataParallel(inception_model, output_device=local_rank)
		else:
			pass

		mu, sigma = prepare_inception_moments(dataloader=eval_dataloader,
											  generator=Gen,
											  eval_mode=cfgs.eval_type,
											  inception_model=inception_model,
											  splits=1,
											  run_name=run_name__,
											  logger=logger,
											  device=local_rank)

	worker = make_worker(
		cfgs=cfgs,
		run_name=run_name,
		best_step=0,
		logger=logger,
		writer=writer,
		n_gpus=world_size,
		gen_model=Gen,
		dis_model=Dis,
		#d2dis_model=d2Dis,
		inception_model=inception_model,
		Gen_copy=Gen_copy,
		Gen_ema=Gen_ema,
		train_dataset=train_dataset,
		eval_dataset=eval_dataset,
		train_dataloader=train_dataloader,
		eval_dataloader=eval_dataloader,
		G_optimizer=G_optimizer,
		D_optimizer=D_optimizer,
		#d2D_optimizer=None,
		G_loss=G_loss[cfgs.adv_loss],
		D_loss=D_loss[cfgs.adv_loss],
		#d2D_loss=None,
		prev_ada_p=prev_ada_p,
		global_rank=global_rank,
		local_rank=local_rank,
		bn_stat_OnTheFly=cfgs.bn_stat_OnTheFly,
		checkpoint_dir=checkpoint_dir_save,#notice
		mu=mu,
		sigma=sigma,
		best_fid=best_fid,
		best_fid_checkpoint_path=best_fid_checkpoint_path,

		encoder=encoder,
		#vae=None,
		opt_encoder=opt_encoder,

	)

	if cfgs.train_configs['train']:
		step = worker.train(current_step=step, total_step=cfgs.total_step)

	if cfgs.eval:
		is_save = worker.evaluation(step=step, standing_statistics=cfgs.standing_statistics, standing_step=cfgs.standing_step)

	if cfgs.save_images:
		worker.save_images(is_generate=True, png=True, npz=True, standing_statistics=cfgs.standing_statistics, standing_step=cfgs.standing_step)

	if cfgs.image_visualization:
		worker.run_image_visualization(nrow=cfgs.nrow, ncol=cfgs.ncol, standing_statistics=cfgs.standing_statistics, standing_step=cfgs.standing_step)

	if cfgs.k_nearest_neighbor:
		worker.run_nearest_neighbor(nrow=cfgs.nrow, ncol=cfgs.ncol, standing_statistics=cfgs.standing_statistics, standing_step=cfgs.standing_step)

	if cfgs.interpolation:
		assert cfgs.architecture in ["big_resnet", "biggan_deep"], "StudioGAN does not support interpolation analysis except for biggan and biggan_deep."
		worker.run_linear_interpolation(nrow=cfgs.nrow, ncol=cfgs.ncol, fix_z=True, fix_y=False,
										standing_statistics=cfgs.standing_statistics, standing_step=cfgs.standing_step)
		worker.run_linear_interpolation(nrow=cfgs.nrow, ncol=cfgs.ncol, fix_z=False, fix_y=True,
										standing_statistics=cfgs.standing_statistics, standing_step=cfgs.standing_step)

	if cfgs.frequency_analysis:
		worker.run_frequency_analysis(num_images=len(train_dataset),
									  standing_statistics=cfgs.standing_statistics, standing_step=cfgs.standing_step)

	if cfgs.tsne_analysis:
		worker.run_tsne(dataloader=eval_dataloader,
						standing_statistics=cfgs.standing_statistics, standing_step=cfgs.standing_step)
