import torch
import logging
import torch.nn as nn
from .GANv2 import Generator, Encoder, VAE
from .model_utils import check_params


class ContraCGAN(nn.Module):
    def __init__(self, cfgs):
        super(ContraCGAN, self).__init__()
        assert check_params(cfgs, None)
        self.gen = Generator(
            cfgs.z_dim, cfgs.shared_dim, cfgs.img_size, cfgs.g_conv_dim,
            cfgs.g_spectral_norm, cfgs.attention,
            cfgs.attention_after_nth_gen_block, cfgs.activation_fn,
            cfgs.conditional_strategy, cfgs.num_classes, cfgs.g_init,
            cfgs.G_depth, False)
        self.encoder = Encoder(
            isize=cfgs.img_size, nz=cfgs.z_dim, nc=3, ndf=64)
        self.vae = VAE(isize=cfgs.img_size, nz=cfgs.z_dim)
        self.class_num = cfgs.num_classes

    def forward(self, data):
        img = data
        device = img.device
        n = img.size(0)
        img_gen_dict = {}
        latent_i = self.encoder(img)
        _, _, z = self.vae(latent_i)
        for classId in range(self.class_num):
            label = torch.LongTensor([classId] * n).to(device)
            img_gen_dict[classId] = self.gen(z, label)
        return img_gen_dict

    def resume_weights(self, cfgs, verbal=""):
        self.gen.load_state_dict(torch.load(cfgs.G_weights)['state_dict'])
        logging.info("{}Loaded gen from: {}".format(verbal, cfgs.G_weights))
        self.encoder.load_state_dict(torch.load(cfgs.E_weights)['state_dict'])
        logging.info("{}Loaded encoder from: {}".format(verbal, cfgs.E_weights))
        self.vae.load_state_dict(torch.load(cfgs.V_weights)['state_dict'])
        logging.info("{}Loaded vae from: {}".format(verbal, cfgs.V_weights))
