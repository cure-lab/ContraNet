import logging
from typing import List
import torch
import torch.nn as nn
from torchvision.transforms import Compose, Normalize
from models.ContraNet2dist import ContraNet2dist


class ContraNetDict(nn.Module):
    def __init__(self, classifier, cls_norm):
        super(ContraNetDict, self).__init__()
        self.classifier = classifier
        self.detector = nn.ModuleDict()
        if cls_norm is None:
            self.cls_norm = Compose([])
        else:
            self.cls_norm = Normalize(*cls_norm)

    def __len__(self):
        return len(self.detector)

    def keys(self):
        return self.detector.keys()

    @property
    def only_judge(self):
        return torch.ByteTensor([detector._only_judge
                                 for detector in self.detector.values()])

    @only_judge.setter
    def only_judge(self, value: bool):
        for d_name in self.detector:
            self.detector[d_name]._only_judge = value

    @property
    def only_rej(self):
        return torch.ByteTensor([detector._only_rej
                                 for detector in self.detector.values()])

    @only_rej.setter
    def only_rej(self, value: bool):
        for d_name in self.detector:
            self.detector[d_name]._only_rej = value
            self.detector[d_name].ddebug = value

    @property
    def thresh(self):
        return [detector._thresh for detector in self.detector.values()]

    @thresh.setter
    def thresh(self, thresh: List):
        assert len(thresh) == len(self.detector)
        for idx, d_name in enumerate(self.detector):
            self.detector[d_name]._thresh = thresh[idx]
            logging.info(
                "[detectorDict] {} using thresh={}".format(
                    d_name, thresh[idx]))

    def forward(self, img):
        if self.only_judge.all():
            score_dict = {}
            for d_name in self.detector:
                detector = self.detector[d_name]
                score = detector(img)
                score_dict[d_name] = score
            return score_dict

        if self.only_rej.all():
            rej_dict = {}
            all_rej = torch.ByteTensor([0] * img.shape[0])
            for d_name in self.detector:
                detector = self.detector[d_name]
                _, rej, logits_cls = detector(img)
                rej_dict[d_name] = rej
                all_rej = torch.logical_or(all_rej, rej)
            return rej_dict, all_rej, logits_cls
        else:
            raise NotImplementedError("Only support for only_rej mode.")

    def get_thresh(self, test_loader):
        thresh = []
        for d_name in self.detector:
            if isinstance(self.detector[d_name], ContraNet2dist):
                torch.random.manual_seed(1)  # manual seed after D and DM
            thresh.append(self.detector[d_name].get_thresh(test_loader))
            logging.info("[detectorDict] threshold for {} = {}".format(
                d_name, thresh[-1]))
        return thresh

    def load_classifier(self, path, key=None):
        checkpoint = torch.load(path)
        if key is not None:
            checkpoint = checkpoint[key]
        self.classifier.load_state_dict(checkpoint)
        logging.info("[detectorDict] Loaded classifier from: {}".format(path))
