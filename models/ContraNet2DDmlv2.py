from typing import List
import torch
import torch.nn as nn
from .ContraNet2D import ContraNet2D
from .ContraNet2 import ContraNet2


class ContraNet2DDmlv2(nn.Module):
    def __init__(self, cfgs, pars, classifier, cls_norm=None, model_norm=None):
        super(ContraNet2DDmlv2, self).__init__()
        self.classifier = classifier
        self.Dmodel = ContraNet2D(
            cfgs, pars, self.classifier, cls_norm, model_norm)
        self.DMmodel = ContraNet2(
            cfgs, pars, self.classifier, cls_norm, model_norm)
        self.cls_norm = self.Dmodel.cls_norm
        self.Dmodel.debug = True
        self.debug = False

    @property
    def only_rej(self):
        assert self.Dmodel._only_rej == self.DMmodel._only_rej
        return self.Dmodel._only_rej

    @only_rej.setter
    def only_rej(self, value: bool):
        self.Dmodel._only_rej = value
        self.DMmodel._only_rej = value

    @property
    def thresh(self):
        return [self.Dmodel._thresh, self.DMmodel._thresh]

    @thresh.setter
    def thresh(self, thresh: List):
        assert len(thresh) == 2
        self.Dmodel._thresh = thresh[0]
        self.DMmodel._thresh = thresh[1]

    def forward(self, img):
        _, dis_rej = self.Dmodel(img)
        dis_rej = dis_rej.to(img.device).byte()
        if self.debug:
            self.DMmodel.ddebug = True
            dm_logits, dm_rej, logits_cls = self.DMmodel((img, dis_rej))
        else:
            self.DMmodel.ddebug = False
            dm_logits = self.DMmodel((img, dis_rej))
        final_logits = dm_logits

        if self.debug:
            dm_rej = dm_rej.to(img.device)
            return final_logits, dis_rej, dm_rej, torch.logical_or(
                dis_rej, dm_rej), logits_cls
        else:
            return final_logits

    def load_classifier(self, path, key=None):
        checkpoint = torch.load(path)
        if key is not None:
            checkpoint = checkpoint[key]
        self.classifier.load_state_dict(checkpoint)
        print("Loaded classifier from: {}".format(path))
