from torch import nn
import torch


class Encoder(nn.Module):
    def __init__(self, isize, nz, nc, ndf, add_final_conv=True):
        super(Encoder, self).__init__()
        encoder = nn.Sequential()
        encoder.add_module('initial-conv-{0}-{1}'.format(nc, ndf),
                           nn.Conv2d(nc, ndf, 4, 2, 1, bias=False))
        encoder.add_module('initial-relu-{0}'.format(ndf),
                           nn.LeakyReLU(0.2, inplace=True))
        csize, cndf = isize/2, ndf

        while csize > 4:
            in_feat = cndf
            out_feat = cndf * 2
            encoder.add_module('pyramid-{0}-{1}'.format(in_feat, out_feat),
                               nn.Conv2d(in_feat, out_feat, 4, 2, 1, bias=False)
                               )
            encoder.add_module('pyramid-{0}-batchnorm'.format(out_feat),
                               nn.BatchNorm2d(out_feat))
            encoder.add_module('pyramid-{0}-relu'.format(out_feat),
                               nn.LeakyReLU(0.2, inplace=True))
            cndf = cndf * 2
            csize = csize/2
        if add_final_conv:
            encoder.add_module('final-{0}-{1}-conv'.format(cndf, 1),
                               nn.Conv2d(cndf, nz, 4, 1, 0, bias=False))
        self.encoder = encoder

    def forward(self, input):
        output = self.encoder(input)
        return output


class VAE(nn.Module):
    def __init__(self, n_extra_layers=0, add_final_conv=True):
        super(VAE, self).__init__()

        self.isize = 32
        self.nz = 80
        self.nc = 3
        self.ndf = 64
        self.device = "cuda"

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

        return z_mean_ret, z_log_var_ret, latent_i_star.float()
