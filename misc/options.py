""" Options

This script is largely based on junyanz/pytorch-CycleGAN-and-pix2pix.

Returns:
    [argparse]: Class containing argparse
"""

import argparse
import os
import torch

# pylint: disable=C0103,C0301,R0903,W0622

class Options():
    """Options class

    Returns:
        [argparse]: argparse containing train and test options
    """

    def __init__(self):
        ##
        #
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        ##
        # Base
        
        self.parser.add_argument('--root', default='./data_tiny_imagenet/train/n01443537', help='path to dataset')
        self.parser.add_argument('--batchsize', type=int, default=64, help='input batch size')
        self.parser.add_argument('--workers', type=int, help='number of data loading workers', default=8)
        self.parser.add_argument('--droplast', action='store_true', default=True, help='Drop last batch size.')
        self.parser.add_argument('--isize', type=int, default=64, help='input image size.')
        self.parser.add_argument('--nc', type=int, default=3, help='input image channels')
        self.parser.add_argument('--nz', type=int, default=128, help='size of the latent z vector')
        self.parser.add_argument('--ngf', type=int, default=64)
        self.parser.add_argument('--ndf', type=int, default=64)
        self.parser.add_argument('--extralayers', type=int, default=0, help='Number of extra layers on gen and disc')
        self.parser.add_argument('--device', type=str, default='cuda', help='Device: gpu | cpu')
        self.parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        self.parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
        self.parser.add_argument('--name', type=str, default='experiment_name', help='name of the experiment')
        self.parser.add_argument('--model', type=str, default='ganomaly', help='chooses which model to use. ganomaly')
        self.parser.add_argument('--display_server', type=str, default="http://localhost", help='visdom server of the web display')
        self.parser.add_argument('--display_port', type=int, default=8097, help='visdom port of the web display')
        self.parser.add_argument('--display_id', type=int, default=0, help='window id of the web display')
        self.parser.add_argument('--display', action='store_true', help='Use visdom.')
        self.parser.add_argument('--outf', default='./output', help='folder to output images and model checkpoints')
        self.parser.add_argument('--manualseed', default=-1, type=int, help='manual seed')
        self.parser.add_argument('--anomaly_class', default='car', help='Anomaly class idx for mnist and cifar datasets')
        self.parser.add_argument('--proportion', type=float, default=0.1, help='Proportion of anomalies in test set.')
        self.parser.add_argument('--metric', type=str, default='roc', help='Evaluation metric.')

        ##
        # Train
        self.parser.add_argument('--print_freq', type=int, default=100, help='frequency of showing training results on console')
        self.parser.add_argument('--save_image_freq', type=int, default=100, help='frequency of saving real and fake images')
        self.parser.add_argument('--save_test_images', action='store_true', help='Save test images for demo.')
        self.parser.add_argument('--load_weights', action='store_true', help='Load the pretrained weights')
        self.parser.add_argument('--resume', default='', help="path to checkpoints (to continue training)")
        self.parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        self.parser.add_argument('--iter', type=int, default=0, help='Start from iteration i')
        self.parser.add_argument('--niter', type=int, default=15, help='number of epochs to train for')
        self.parser.add_argument('--niter_decay', type=int, default=100, help='# of iter to linearly decay learning rate to zero')#越大lr衰减月慢
        #self.parser.add_argument('--niter_decay', type=int, default=500, help='# of iter to linearly decay learning rate to zero')  # 越大lr衰减月慢
        self.parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        self.parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
        self.parser.add_argument('--w_bce', type=float, default=1, help='alpha to weight bce loss.')
        self.parser.add_argument('--w_rec', type=float, default=50, help='alpha to weight reconstruction loss')
        self.parser.add_argument('--w_enc', type=float, default=1, help='alpha to weight encoder loss')
        self.parser.add_argument('--CRITIC_ITERS', type=int, default=4, help='For WGAN and WGAN-GP, number of critic iters per gen iter')
        self.parser.add_argument('--CRITIC_ITERS2', type=int, default=5, help='For WGAN and WGAN-GP, number of critic iters per gen iter')
        self.parser.add_argument('--CRITIC_ITERS3', type=int, default=6, help='For WGAN and WGAN-GP, number of critic iters per gen iter')
        self.parser.add_argument('--num_train', type=int, default=5, help='For caltech coil100 dataset number')
        self.parser.add_argument('--ratio', type=int, default=0.5, help='For caltech coil100 dataset outliers number')
        self.parser.add_argument('--num_classes', type=int, default=9, help='For cvae clusters')

        self.isTrain = True
        self.opt = None
        self.number = None
        self.parser.add_argument('--w', nargs="?", default=0.5, type=float,
                        help='weight for the sum of the mapping loss function')
    def parse(self):
        """ Parse Arguments.
        """

        self.opt = self.parser.parse_known_args()[0]
      
        return self.opt
