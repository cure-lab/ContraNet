# PyTorch StudioGAN: https://github.com/POSTECH-CVLab/PyTorch-StudioGAN
# The MIT License (MIT)
# See license file or visit https://github.com/POSTECH-CVLab/PyTorch-StudioGAN for details

# src/data_utils/load_dataset.py


import os
import h5py as h5
import numpy as np
import random
from PIL import Image
import pandas as pd

import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10, CIFAR100, MNIST
from torchvision.datasets import ImageFolder


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
             else np.random.randint(low=0, high=img.size[0] - size[0]))
        j = (0 if size[1] == img.size[1]
             else np.random.randint(low=0, high=img.size[1] - size[1]))
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
    def __init__(
            self, dataset_name, data_path, train, download, resize_size,
            hdf5_path=None, random_flip=False, norm=True):
        super(LoadDataset, self).__init__()
        self.dataset_name = dataset_name
        self.data_path = data_path
        self.train = train
        self.download = download
        self.resize_size = resize_size
        self.hdf5_path = hdf5_path
        self.random_flip = random_flip
        self.norm_mean = [0.5, 0.5, 0.5]
        self.norm_std = [0.5, 0.5, 0.5]

        if self.hdf5_path is None:
            if self.dataset_name in ['cifar10', 'cifar100']:
                self.transforms = []
            elif self.dataset_name in ["tiny_imagenet", 'MNIST']:
                self.transforms = [transforms.Resize(self.resize_size)]
            elif self.dataset_name in ['imagenet', 'gtsrb', 'custom']:
                if train:
                    self.transforms = [
                        RandomCropLongEdge(),
                        transforms.Resize(self.resize_size)]
                else:
                    self.transforms = [
                        CenterCropLongEdge(),
                        transforms.Resize(self.resize_size)]
        else:
            self.transforms = [transforms.ToPILImage()]

        if random_flip and self.dataset_name != "gtsrb":
            self.transforms += [transforms.RandomHorizontalFlip()]

        self.transforms += [transforms.ToTensor()]
        if norm:
            if self.dataset_name == "MNIST":
                self.transforms += [
                    transforms.Normalize(self.norm_mean[:1], self.norm_std[:1])
                ]
            else:
                self.transforms += [
                    transforms.Normalize(self.norm_mean, self.norm_std)
                ]
        self.transforms = transforms.Compose(self.transforms)
        print(self.transforms)

        self.load_dataset()

    def load_dataset(self):
        if self.dataset_name == 'cifar10':
            if self.hdf5_path is not None:
                print('Loading %s into memory...' % self.hdf5_path)
                with h5.File(self.hdf5_path, 'r') as f:
                    self.data = f['imgs'][:]
                    self.labels = f['labels'][:]
            else:
                self.data = CIFAR10(root=os.path.join(self.data_path),
                                    train=self.train,
                                    download=self.download)
        elif self.dataset_name == 'cifar100':
            if self.hdf5_path is not None:
                print('Loading %s into memory...' % self.hdf5_path)
                with h5.File(self.hdf5_path, 'r') as f:
                    self.data = f['imgs'][:]
                    self.labels = f['labels'][:]
            else:
                self.data = CIFAR100(root=os.path.join(self.data_path),
                                     train=self.train,
                                     download=self.download)
        elif self.dataset_name == 'MNIST':
            self.data = MNIST(
                root=os.path.join(self.data_path),
                train=self.train, download=self.download)
        elif self.dataset_name == 'imagenet':
            if self.hdf5_path is not None:
                print('Loading %s into memory...' % self.hdf5_path)
                with h5.File(self.hdf5_path, 'r') as f:
                    self.data = f['imgs'][:]
                    self.labels = f['labels'][:]
            else:
                mode = 'train' if self.train == True else 'valid'
                root = os.path.join('data', 'ILSVRC2012', mode)
                self.data = ImageFolder(root=root)

        elif self.dataset_name == 'gtsrb':
            self.root_dir = os.path.join(self.data_path, 'GTSRB')
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
                mode = 'train' if self.train == True else 'val/images'
                root = os.path.join(self.data_path, "tiny-imagenet-200", mode)
                self.data = ImageFolder(root=root)
                random.shuffle(self.data.samples, lambda: 0.1)
                self.data.targets = [s[1] for s in self.data.samples]

        elif self.dataset_name == "STL10":
            mode = 'test' if self.train == True else 'train'
            self.data = STL10(
                root=os.path.join('data', self.dataset_name),
                split=mode, download=True)

        elif self.dataset_name == "custom":
            if self.hdf5_path is not None:
                print('Loading %s into memory...' % self.hdf5_path)
                with h5.File(self.hdf5_path, 'r') as f:
                    self.data = f['imgs'][:]
                    self.labels = f['labels'][:]
            else:
                mode = 'train' if self.train == True else 'valid'
                root = os.path.join('data', 'CUSTOM', mode)
                self.data = ImageFolder(root=root)
        else:
            raise NotImplementedError

    def __len__(self):
        if self.dataset_name == "gtsrb":
            num_dataset = len(self.csv_data)
        elif self.hdf5_path is not None:
            num_dataset = self.data.shape[0]
        else:
            num_dataset = len(self.data)
        return num_dataset

    def __getitem__(self, index):
        if self.dataset_name == "gtsrb":
            img_path = os.path.join(
                self.root_dir, self.sub_directory, self.csv_data.iloc[index, 0]
            )
            img = Image.open(img_path)
            label = int(self.csv_data.iloc[index, 1])
            if self.transforms is not None:
                img = self.transforms(img)
            return img, label
        elif self.hdf5_path is None:
            img, label = self.data[index]
            img, label = self.transforms(img), int(label)
            if self.dataset_name == "MNIST":
                img = img.expand(3, img.shape[1], img.shape[2])
        else:
            img, label = \
                np.transpose(self.data[index], (1, 2, 0)), \
                int(self.labels[index])
            img = self.transforms(img)
        return img, label
