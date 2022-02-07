from abc import abstractmethod
import torch.nn as nn
import numpy as np
import random
from .model_utils import check_params
from .ContraCGAN import ContraCGAN
from torchvision.transforms import Normalize, Compose


class ContraNetDictBase(nn.Module):
    """ContraNet for cGAN + DML + classifier
    For inference Only.

    Args:
        cfgs: configuration for cGAN
        pars: configuration for DML
    """

    def __init__(self, cfgs, pars, classifier, cls_norm=None, model_norm=None):
        super(ContraNetDictBase, self).__init__()
        assert check_params(cfgs, pars=None)
        self.cGAN = ContraCGAN(cfgs)
        self.classifier = classifier
        if cls_norm is None:
            self.cls_norm = Compose([])
        else:
            self.cls_norm = Normalize(*cls_norm)
        self.model_norm = Normalize(*model_norm)
        [mean, var] = model_norm
        mean = np.array(mean)
        var = np.array(var)
        self.model_denorm = Normalize(mean=-mean / var, std=1. / var)
        self.debug = False
        self.ddebug = False
        self.return_fig = False
        self.fake_wrong = False
        self._thresh = None
        self._only_rej = False
        self._only_judge = False
        self.min_distance = False

    def forward_classifier(self, img):
        img_cls = self.cls_norm(img)
        logits_cls = self.classifier(img_cls)
        _, cls_pred = logits_cls.max(1)
        if self.fake_wrong:
            cls_pred = (cls_pred + random.randint(1, 9)) % 10
        return logits_cls, cls_pred

    def sort_results(self, out):
        return out.sort()
