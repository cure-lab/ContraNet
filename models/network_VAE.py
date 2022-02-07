import torch
import torch.nn as nn
import torch.nn.parallel
import sys

sys.path.append("..")


def weights_init(mod):
    """
    Custom weights initialization called on netG, netD and netE
    :param m:
    :return:
    """
    #print("mod=", mod)
    classname = mod.__class__.__name__
    if classname.find('Conv') != -1:
        mod.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        #print('BatchNorm initial')
        mod.weight.data.normal_(1.0, 0.02)
        mod.bias.data.fill_(0)


def weights_init_WD(mod):
    """
    Custom weights initialization called on netG, netD and netE
    :param m:
    :return:
    """
    #print("mod=", mod)
    classname = mod.__class__.__name__
    if classname.find('Conv') != -1:
        #mod.weight.data.normal_(0.0, 0.02)
        nn.init.xavier_normal_(mod.weight.data, gain=1)


def weights_init_info(mod):
    """
    Custom weights initialization called on netG, netD and netE
    :param m:
    :return:
    """
    classname = mod.__class__.__name__
    if classname.find('Conv') != -1:
        # mod.weight.data.normal_(0.0, 0.02)
        nn.init.xavier_normal_(mod.weight.data, gain=1)


###
class Encoder(nn.Module):
    """
    ENCODER NETWORK
    """
    # ndf是输出的channel个数

    def __init__(self, opt, n_extra_layers=0, add_final_conv=True, is_gan=False):
        super(Encoder, self).__init__()
        self.ngpu = opt.ngpu
        self.isize = opt.isize
        self.nc = opt.nc
        self.ndf = opt.ndf
        self.device = opt.device

        if is_gan:
            self.nz = 1
        else:
            self.nz = opt.nz
        assert self.isize % 16 == 0, "isize has to be a multiple of 16"

        main = nn.Sequential()  # model模型
        # input is nc x isize x isize

        # main.add_module('initial-conv',
        # nn.Conv2d(nc, ndf, 1, 1, 0, bias=False))  # （32+2×0-1）/1+1=32 #wgan-gp kernel是3
        # main.add_module('initial-relu',
        # nn.LeakyReLU(0.2, inplace=True))

        main.add_module(
            'initial-conv-{0}-{1}'.format(self.nc, self.ndf),
            nn.Conv2d(self.nc, self.ndf, 4, 2, 1, bias=False))  # （32+2×1-4）/2+1=16 #wgan-gp kernel是3###第一个ndf是nc
        main.add_module('initial-relu-{0}'.format(self.ndf),
                        nn.LeakyReLU(0.2, inplace=True))
        csize, cndf = self.isize / 2, self.ndf  # 图像的大小缩小两倍  channel数量不变 16对应64
        # self.netg.main.initial-relu-64

        # Extra layers
        for t in range(n_extra_layers):  # 没有额外的卷积层
            main.add_module('extra-layers-{0}-{1}-conv'.format(t, cndf),
                            nn.Conv2d(cndf, cndf, 3, 1, 1, bias=False))
            main.add_module('extra-layers-{0}-{1}-batchnorm'.format(t, cndf),
                            nn.BatchNorm2d(cndf))
            main.add_module('extra-layers-{0}-{1}-relu'.format(t, cndf),
                            nn.LeakyReLU(0.2, inplace=True))

        while csize > 4:  # 图像大于4的话就继续 16 8 4 一共新加两层卷积层
            in_feat = cndf
            out_feat = cndf * 2
            main.add_module('pyramid-{0}-{1}-conv'.format(in_feat, out_feat),
                            nn.Conv2d(in_feat, out_feat, 4, 2, 1, bias=False))
            main.add_module('pyramid-{0}-batchnorm'.format(out_feat),
                            nn.BatchNorm2d(out_feat))
            main.add_module('pyramid-{0}-relu'.format(out_feat),
                            nn.LeakyReLU(0.2, inplace=True))
            cndf = cndf * 2  # channel 变为2倍
            csize = csize / 2  # 图像缩小两倍

        # state size. K x 4 x 4 #最后一层卷积  一共四层卷积
        if add_final_conv:
            main.add_module(
                'final-{0}-{1}-conv'.format(cndf, 1),
                nn.Conv2d(cndf, self.nz, 4, 1, 0, bias=False))  # 图像大小现在已经小于4了 (（3）+2×0-4）/2+1=1  nz=100

        self.main = main
        # self.z_mean_calc = nn.Linear(self.nz, self.nz)  # 多加一层表示每个独立z的均值 可以不初始化
        # self.z_log_var_calc = nn.Linear(self.nz, self.nz)  # 多加一层表示每个独立z的方差 可以不初始化

    def forward(self, input):
        # print(input.shape)
        #print('encoder encoder encoder')
        latent_i = self.main(input)
        #z_mean = self.z_mean_calc(latent_i.view(-1, self.nz))
        #z_log_var = self.z_log_var_calc(latent_i.view(-1, self.nz))
        # epsilon = torch.randn(size=(z_mean.view(-1,self.nz).shape[0], self.nz)).to(self.device) #Sampling
        # latent_i_star = z_mean + torch.exp(z_log_var / 2) * epsilon  #Sampling
        return latent_i

##


class VAE(nn.Module):
    def __init__(self, opt, n_extra_layers=0, add_final_conv=True):
        super(VAE, self).__init__()
        self.ngpu = opt.ngpu
        self.isize = opt.isize
        self.nz = opt.nz
        self.nc = opt.nc
        self.ndf = opt.ndf
        self.device = opt.device
        assert self.isize % 16 == 0, "isize has to be a multiple of 16"

        self.z_mean_calc = nn.Linear(self.nz, self.nz)  # 多加一层表示每个独立z的均值 可以不初始化
        self.z_log_var_calc = nn.Linear(
            self.nz, self.nz)  # 多加一层表示每个独立z的方差 可以不初始化

    def forward(self, input):

        z_mean = self.z_mean_calc(input.view(-1, self.nz))
        z_log_var = self.z_log_var_calc(input.view(-1, self.nz))

        #stamp = self.get_stamp(target,opt)

        # 继续传播的部分
        z_mean_0 = z_mean  # * stamp
        z_log_var_0 = z_log_var  # * stamp
        epsilon = torch.randn(
            size=(z_mean_0.view(-1, self.nz).shape[0],
                  self.nz)).to(
            self.device)  # Sampling
        # Sampling
        latent_i_star = z_mean_0 + torch.exp(z_log_var_0 / 2) * epsilon

        # 不继续传播的部分 bias 可调。最初为 1
        #bias = 100
        #z_mean_flip = (bias-z_mean) * (1-stamp)
        #z_log_var_flip = (1-z_log_var) * (1-stamp)

        # 组合在一起返回
        z_mean_ret = z_mean_0  # + z_mean_flip
        z_log_var_ret = z_log_var_0  # + z_log_var_flip

        return z_mean_ret, z_log_var_ret, latent_i_star.type(torch.FloatTensor)


class Decoder(nn.Module):
    """
    DCGAN DECODER NETWORK
    """

    def __init__(self, opt, n_extra_layers=0):
        super(Decoder, self).__init__()
        self.ngpu = opt.ngpu
        self.isize = opt.isize
        self.nz = opt.nz
        self.nc = opt.nc
        self.ndf = opt.ndf
        self.ngf = opt.ngf
        self.device = opt.device
        assert self.isize % 16 == 0, "isize has to be a multiple of 16"

        cngf, tisize = self.ngf // 2, 4  # ngf=64  图像大小      32个channel对应4的图像大小
        while tisize != self.isize:
            cngf = cngf * 2
            tisize = tisize * 2

        main = nn.Sequential()
        # input is Z, going into a convolution
        main.add_module('initial-{0}-{1}-convt'.format(self.nz, cngf),
                        nn.ConvTranspose2d(self.nz, cngf, 4, 1, 0, bias=False))
        main.add_module('initial-{0}-batchnorm'.format(cngf),
                        nn.BatchNorm2d(cngf))
        main.add_module('initial-{0}-relu'.format(cngf),
                        nn.ReLU(True))

        csize, _ = 4, cngf
        while csize < self.isize // 2:
            main.add_module(
                'pyramid-{0}-{1}-convt'.format(cngf, cngf // 2),
                nn.ConvTranspose2d(cngf, cngf // 2, 4, 2, 1, bias=False))
            main.add_module('pyramid-{0}-batchnorm'.format(cngf // 2),
                            nn.BatchNorm2d(cngf // 2))
            main.add_module('pyramid-{0}-relu'.format(cngf // 2),
                            nn.ReLU(True))
            cngf = cngf // 2
            csize = csize * 2  # 配合前面

        # Extra layers
        for t in range(n_extra_layers):
            main.add_module('extra-layers-{0}-{1}-conv'.format(t, cngf),
                            nn.Conv2d(cngf, cngf, 3, 1, 1, bias=False))
            main.add_module('extra-layers-{0}-{1}-batchnorm'.format(t, cngf),
                            nn.BatchNorm2d(cngf))
            main.add_module('extra-layers-{0}-{1}-relu'.format(t, cngf),
                            nn.ReLU(True))

        main.add_module('final-{0}-{1}-convt'.format(cngf, self.nc),
                        nn.ConvTranspose2d(cngf, self.nc, 4, 2, 1, bias=False))
        main.add_module('final-{0}-tanh'.format(self.nc),
                        nn.Tanh())  # 逐元素
        self.main = main

    def forward(self, input):
        output = self.main(input)
        return output
