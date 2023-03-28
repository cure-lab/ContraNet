import glob
import os
import random
from os.path import dirname, abspath, exists, join
from data_utils.load_dataset import *

import torch
from torch.utils.data import DataLoader
from torch.nn import DataParallel

import encoder as encoder_source
import models.big_resnet as module 

# ** the largest test GTSRB image is 232 *266
import json
import os
import shutil

import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import torch.nn as nn
import gtsrb_dataset as dataset
import matplotlib.pyplot as plt
import os
import matplotlib.image as mpimg
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
from torch.autograd import Variable
# from options import Options
import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib as mpl 
mpl.use("Agg")
import os
from PIL import Image
import numpy
import matplotlib.colors as mcolors
import matplotlib.font_manager as font_manager
from matplotlib.ticker import FormatStrFormatter
import matplotlib.ticker as tck

from matplotlib import ticker
print(mpl.matplotlib_fname())
print(mpl.get_cachedir())
from torchsummary import summary




class CenterCropLongEdge(object):
	"""
	this code is borrowed from https://github.com/ajbrock/BigGAN-PyTorch
	MIT License
	Copyright (c) 2019 Andy Brock
	"""
	def __call__(self, img):
		return transforms.functional.center_crop(img, min(img.size))

	def __repr__(self):
		return self.__class__.__name__



def ticks_format(value, index):
    """
    get the value and returns the value as:
       integer: [0,99]
       1 digit float: [0.1, 0.99]
       n*10^m: otherwise
    To have all the number of the same size they are all returned as latex strings
    """
    exp = np.floor(np.log10(value))
    base = value/10**exp
    if exp == 0 or exp == 1:   
        return '${0:d}$'.format(int(value))
    if exp == -1:
        return '${0:.1f}$'.format(value)
    else:
        return '${0:d}\\times10^{{{1:d}}}$'.format(int(base), int(exp))



mpl.rcParams.update(
    {
        'text.usetex': False,
        'font.family': 'stixgeneral',
        'mathtext.fontset': 'stix',
    }
)
mpl.rc('xtick', labelsize=9) 
mpl.rc('ytick', labelsize=9) 
# colors=list(mcolors.TABLEAU_COLORS.keys())

title_font = {'fontname':'Times New Roman', 'size':'12', 'color':'black', 'weight':'heavy',
              'verticalalignment':'bottom'} # Bottom vertical alignment for more space
axis_font = {'fontname':'Times New Roman', 'size':'11','style':'italic'}

# Set the font properties (for use in legend)   
font_path = '/research/dept6/yjyang/anaconda3/envs/rygaoMOCO/lib/python3.7/site-packages/matplotlib/mpl-data/fonts/ttf/times-new-roman.ttf'
font_prop = font_manager.FontProperties(fname=font_path, size=7)

# load models
Gen = module.Generator(80, 128, 32, 96, True, True,
						   2, 'ReLU', 'ProjGAN', 43,
						   "ortho", 'N/A', False).cuda()


Dis = module.Discriminator(32, 96, True, True, 1,
							   'ReLU', 'ProjGAN', 'N/A', 43, False,
							   False, 'ortho', 'N/A', False).cuda()

encoder = encoder_source.Encoder(isize=32, nz=80, nc=3, ndf=64).cuda()
print(summary(encoder, (3, 32, 32))) 
# print(summary(Gen,[(3, 32, 32),(1,1)]))
vae = encoder_source.VAE().cuda()
print(summary(vae,(80,)))
print(summary(Gen,[(80,),()]))
print(summary(Dis,[(3,32,32),()]))
exit()
gen_checkpoint = torch.load('/research/dept6/yjyang/rygao/pretrain/gtsrb_cGAN/model=G-current-weights-step=90000.pth')
Gen.load_state_dict(gen_checkpoint['state_dict'])

encoder_checkpoint = torch.load('/research/dept6/yjyang/rygao/pretrain/gtsrb_cGAN/model=E-current-weights-step=90000.pth')
encoder.load_state_dict(encoder_checkpoint['state_dict'])
vae_checkpoint = torch.load('/research/dept6/yjyang/rygao/pretrain/gtsrb_cGAN/model=V-current-weights-step=90000.pth')
vae.load_state_dict(vae_checkpoint['state_dict'])

transform = transforms.Compose([
		transforms.ToTensor(),
	])
transform_contranet = transforms.Compose([
		CenterCropLongEdge(), transforms.Resize(32), transforms.ToTensor()
])
train_dataset = dataset.GTSRB(
	root_dir = './', train=True, transform=transform
	)

contranet_train_dataset = dataset.GTSRB(
	root_dir = './', train=True, transform=transform_contranet
	)

test_dataset = dataset.GTSRB(
	root_dir='./', train=False, transform=transform
	)
contranet_test_dataset = dataset.GTSRB(
	root_dir='./', train=False, transform=transform_contranet
)

train_loader = iter(torch.utils.data.DataLoader(
	train_dataset, 1,
	shuffle=False,
	drop_last=False)
	)
test_loader = iter(torch.utils.data.DataLoader(
	test_dataset, 1,
	shuffle=False,
	drop_last=False)
	)

contranet_train_loader = iter(torch.utils.data.DataLoader(
	contranet_train_dataset, 1,
	shuffle=False,
	drop_last=False)
	)

contranet_test_loader = iter(torch.utils.data.DataLoader(
	contranet_test_dataset, 1,
	shuffle=False,
	drop_last=False)
	)




large_image = 0
for i in range(len(train_dataset)):
	# import ipdb; ipdb.set_trace()
	image, label = next(train_loader)
	contranet_real_image, real_label = next(contranet_train_loader)
	

	# ax1 = fig.add_subplot(111)
	# ax2 = fig.add_subplot(221)
	

	long_side = min(image.shape[2], image.shape[3])
	if long_side > 20:
		# plt.figure(dpi=300)
		fig, axs = plt.subplots(3,1,sharex=False, sharey=True, dpi=600)
		print(long_side)
		print(image.shape[3])
		large_image = large_image + 1
		print("test190large_image_num++++++++++++++++++++++++++++++++++++++++:",large_image)
		image_pil = transforms.ToPILImage()(image[0]).convert('RGB')
		contranet_real_image_pil = transforms.ToPILImage()(contranet_real_image[0]).convert('RGB')
		latent_vector = encoder(contranet_real_image.cuda())
		z_mean, z_log_var, zs = vae(latent_vector)
		fake_image = Gen(zs.cuda(), real_label.cuda())
		print(zs.shape)
		print(real_label.shape)
		fake_image = torchvision.utils.make_grid(fake_image,nrow=1, padding=0, normalize=True, scale_each=True)
		fake_image_pil = transforms.ToPILImage()(fake_image.cpu()).convert('RGB')
		axs[1].spines['right'].set_visible(False)
		axs[1].spines['top'].set_visible(False)
		axs[1].spines['left'].set_visible(False)
		axs[1].spines['bottom'].set_visible(False)
		axs[1].spines['right'].set_visible(False)
		axs[2].spines['top'].set_visible(False)
		axs[2].spines['left'].set_visible(False)
		axs[2].spines['right'].set_visible(False)
		axs[2].spines['bottom'].set_visible(False)
		axs[1].imshow(contranet_real_image_pil)
		axs[2].imshow(fake_image_pil)
		axs[0].imshow(image_pil)
		# axs[1].set_xticks([])
		# axs[1].set_yticks([])
		# axs[2].set_xticks([])
		# axs[2].set_yticks([])

		plt.tight_layout()
		plt.show()
		plt.savefig("./high_resolution_GTSRB/"+str(i)+"_contranet_train_"+str(image.shape[2])+"_"+str(image.shape[3])+".png", dpi=600)
		plt.close()














