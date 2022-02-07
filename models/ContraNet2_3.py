from .ContraNetDict import ContraNetDict
from .ContraNetDictH import ContraNetDictH
from .ContraNet2D import ContraNet2D
from .ContraNet2 import ContraNet2
from .ContraNet2dist import ContraNet2dist


class ContraNet2_3ssH(ContraNetDictH):
    def __init__(self, cfgs, pars, classifier, cls_norm=None, model_norm=None,
                 hard_drop=[0.12]):
        super(ContraNet2_3ssH, self).__init__(
            classifier, cls_norm, hard_drop=hard_drop)
        self.detector.add_module("D_model", ContraNet2D(
            cfgs, pars, self.classifier, cls_norm, model_norm))
        self.detector.add_module("DMmodel", ContraNet2(
            cfgs, pars, self.classifier, cls_norm, model_norm))
        self.detector.add_module("SSmodel", ContraNet2dist(
            cfgs, pars, self.classifier, cls_norm, model_norm, distance="ssim"))
