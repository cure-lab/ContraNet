import torch
import torch.nn as nn
import torch.nn.functional as F
from .generators.resnet64 import ResNetGenerator
from .generators.network_VAE import Encoder, VAE
from .model_utils import check_params
from .ContraCGAN import ContraCGAN

cfgs_flag = ["num_classes", "gen_num_features", "gen_dim_z",
             "gen_bottom_width", "gen_distribution", "isize", "nc", "ndf", "nz"]


class ContraCGANv1(ContraCGAN):
    def __init__(self, cfgs):
        # do not call super().__init__() here
        assert check_params(cfgs, None, cfgs_flag=cfgs_flag)
        nn.Module.__init__(self)
        _n_cls = cfgs.num_classes + 1
        self.gen = ResNetGenerator(
            cfgs.gen_num_features, cfgs.gen_dim_z, cfgs.gen_bottom_width,
            activation=F.relu, num_classes=_n_cls,
            distribution=cfgs.gen_distribution)
        self.encoder = Encoder(cfgs)
        self.vae = VAE(cfgs)
        self.class_num = cfgs.num_classes

    def resume_weights(self, cfgs):
        self.gen.load_state_dict(torch.load(cfgs.G_weights)['model'])
        print("Loaded gen from: {}".format(cfgs.G_weights))
        self.encoder.load_state_dict(torch.load(cfgs.E_weights)['model'])
        print("Loaded encoder from: {}".format(cfgs.E_weights))
        self.vae.load_state_dict(torch.load(cfgs.V_weights)['model'])
        print("Loaded vae from: {}".format(cfgs.V_weights))
