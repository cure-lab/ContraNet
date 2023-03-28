import os
import h5py as h5
import numpy as np
import random
from scipy import io
from PIL import ImageOps, Image

import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10, STL10, MNIST
from torchvision.datasets import ImageFolder
from  data_utils.load_dataset import *
import torch
import os
import pandas as pd
from torch.utils.data import Dataset
import numpy as np
from PIL import Image


class RandomCropLongEdge(object):
	"""
	this code is borrowed from https://github.com/ajbrock/BigGAN-PyTorch
	MIT License
	Copyright (c) 2019 Andy Brock
	"""
	def __call__(self, img):
		size = (min(img.size), min(img.size))
		# Only step forward along this edge if it's the long edge
		i = (0 if size[0] == img.size[0]
			else np.random.randint(low=0,high=img.size[0] - size[0]))
		j = (0 if size[1] == img.size[1]
			else np.random.randint(low=0,high=img.size[1] - size[1]))
		return transforms.functional.crop(img, i, j, size[0], size[1])

	def __repr__(self):
		return self.__class__.__name__


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


class LoadDataset(Dataset):
	def __init__(self, dataset_name, data_path, train, download, resize_size, hdf5_path=None, random_flip=False):
		super(LoadDataset, self).__init__()
		self.dataset_name = dataset_name
		self.data_path = data_path
		self.train = train
		self.download = download
		self.resize_size = resize_size
		self.hdf5_path = hdf5_path
		self.random_flip = random_flip
		self.norm_mean = [0.5,0.5,0.5]
		self.norm_std = [0.5,0.5,0.5]

		if self.hdf5_path is None:
			if self.dataset_name in ['cifar10', 'tiny_imagenet']:
				self.transforms = []
			elif self.dataset_name in ['imagenet','GTSRB', 'custom','MNIST']:
				if train:
					self.transforms = [RandomCropLongEdge(), transforms.Resize(self.resize_size)]
				else:
					self.transforms = [CenterCropLongEdge(), transforms.Resize(self.resize_size)]
		else:
			self.transforms = [transforms.ToPILImage()]

		if random_flip:
			self.transforms += [transforms.RandomHorizontalFlip()]

		self.transforms += [transforms.ToTensor()]#, transforms.Normalize(self.norm_mean, self.norm_std),self._noise_adder]
		self.transforms = transforms.Compose(self.transforms)

		self.load_dataset()
	# def _rescale(img):
	# 	return img * 2.0 - 1.0

	def _noise_adder(self,img):
		return torch.empty_like(img, dtype=img.dtype).uniform_(0.0, 1/128.0) + img



	# train_dataset = torchvision.datasets.MNIST(root=os.path.join(args.data_root,'train'),
	# 	train=True,
	# 	transform=transforms.Compose([#transforms.RandomChoice([
	# 		#transforms.RandomHorizontalFlip(p=0.5),
	# 		#transforms.RandomGrayscale(p=0.1),
	# 		#transforms.RandomVerticalFlip(p=0.5)
	# 		#]),
	# 		transforms.Resize(64), transforms.ToTensor(), _rescale, _noise_adder,
	# 		]),
	# 	download=True

	# 	)

	def load_dataset(self):
		if self.dataset_name == 'cifar10':
			if self.hdf5_path is not None:
				print('Loading %s into memory...' % self.hdf5_path)
				with h5.File(self.hdf5_path, 'r') as f:
					self.data = f['imgs'][:]
					self.labels = f['labels'][:]
			else:
				self.data = CIFAR10(root=os.path.join('data', self.dataset_name),
									train=self.train,
									download=self.download)

		elif self.dataset_name == 'MNIST':
			self.data = MNIST(root=os.path.join('data', self.dataset_name),
								train=self.train,
								download=True	

				)

		elif self.dataset_name == 'imagenet':
			if self.hdf5_path is not None:
				print('Loading %s into memory...' % self.hdf5_path)
				with h5.File(self.hdf5_path, 'r') as f:
					self.data = f['imgs'][:]
					self.labels = f['labels'][:]
			else:
				mode = 'train' if self.train == True else 'valid'
				root = os.path.join('data','ILSVRC2012', mode)
				self.data = ImageFolder(root=root)


		elif self.dataset_name == 'GTSRB':
			mode = 'train' if self.train == True else 'valid'
			self.root_dir = './GTSRB'
			self.sub_directory = 'trainingset' if self.train else 'testset'
			self.csv_file_name = 'training.csv' if self.train else 'test.csv'

			csv_file_path = os.path.join(
				self.root_dir, self.sub_directory, self.csv_file_name
				)

			self.csv_data = pd.read_csv(csv_file_path)


		elif self.dataset_name == "tiny_imagenet":
			if self.hdf5_path is not None:
				print('Loading %s into memory...' % self.hdf5_path)
				with h5.File(self.hdf5_path, 'r') as f:
					self.data = f['imgs'][:]
					self.labels = f['labels'][:]
			else:
				mode = 'train' if self.train == True else 'valid'
				root = os.path.join('data','TINY_ILSVRC2012', mode)
				self.data = ImageFolder(root=root)

		elif self.dataset_name == "custom":
			if self.hdf5_path is not None:
				print('Loading %s into memory...' % self.hdf5_path)
				with h5.File(self.hdf5_path, 'r') as f:
					self.data = f['imgs'][:]
					self.labels = f['labels'][:]
			else:
				mode = 'train' if self.train == True else 'valid'
				root = os.path.join('data','CUSTOM', mode)
				self.data = ImageFolder(root=root)
		else:
			raise NotImplementedError


	def __len__(self):
		if self.dataset_name == "GTSRB":
			num_dataset = len(self.csv_data)
			return num_dataset
		if self.hdf5_path is not None:
			num_dataset = self.data.shape[0]
		else:
			num_dataset = len(self.data)



		return num_dataset


	def __getitem__(self, index):
		if self.dataset_name == "GTSRB":
			img_path = os.path.join(self.root_dir, self.sub_directory, self.csv_data.iloc[index, 0])

			img = Image.open(img_path)

			classId = self.csv_data.iloc[index, 1]

			if self.transforms is not None:
				img = self.transforms(img)

			return img, classId
		
		if self.hdf5_path is None:
			img, label = self.data[index]
			img, label = self.transforms(img), int(label)
		else:
			img, label = np.transpose(self.data[index], (1,2,0)), int(self.labels[index])
			img = self.transforms(img)
		return img, label
