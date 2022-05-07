from typing import List
import warnings
import torch
import random
from .pgdWrapper import Pgd
from .fgsmWrapper import Fgsm
from .baseWrapper import BaseWrapper


class PreGen(BaseWrapper):
    def __init__(self, net, criterion, cls_norm, load_paths: List, batch_size,
                 targeted=False, x_val_min=0, x_val_max=1, testAttack="Pgd"):
        super(PreGen, self).__init__(net, criterion=None, cls_norm=cls_norm,
                                     targeted=targeted, x_val_min=x_val_min,
                                     x_val_max=x_val_max)
        self.testAttack = eval(
            "{}(net, criterion, cls_norm)".format(testAttack))
        self.AEsample = {load_path: torch.load(
            load_path) for load_path in load_paths}
        warnings.warn("AE's are not aligned with input x when training.")
        temp_key = list(self.AEsample.keys())[0]
        self.index = list(range(len(self.AEsample[temp_key]["x_adv"])))
        self.batch_size = batch_size
        self.AEIter = self.batch()
        self.suss_stat = {load_path: {"suss": 0, "total": 0}
                          for load_path in load_paths}
        self.all_keys = load_paths

    def batch(self):
        random.shuffle(self.index)
        l = len(self.index)
        n = self.batch_size
        for ndx in range(0, l, n):
            this_batch = self.index[ndx:(ndx + n)]
            yield this_batch

    def attack(self, device):
        # load AE sample and give cls_pred
        try:
            batch_idx = next(self.AEIter)
        except StopIteration:
            self.AEIter = self.batch()
            batch_idx = next(self.AEIter)
        key = random.choice(self.all_keys)
        x_adv = self.AEsample[key]["x_adv"][batch_idx].to(device)
        label = self.AEsample[key]["y_ori"][batch_idx].to(device)
        with torch.no_grad():
            h_adv = self.net(self.cls_norm(x_adv))
            _, cls_pred = h_adv.max(1)
            suss = (cls_pred != label).sum().item()
            self.suss_stat[key]["suss"] += suss
            self.suss_stat[key]["total"] += x_adv.shape[0]
        return x_adv.detach(), cls_pred.detach(), label

    def adv_by_suss(self, x, y, device, training=False):
        if not training:
            return self.testAttack.adv_by_suss(x, y, device, training)
        else:
            x_adv, cls_pred, label = self.attack(device)
            fail_mask = torch.where(cls_pred == label)
            suss_mask = torch.where(cls_pred != label)
            return x_adv[suss_mask], cls_pred[suss_mask], label[suss_mask], \
                x_adv[fail_mask], label[fail_mask]

    def print_stat(self):
        for key in self.suss_stat:
            suss = self.suss_stat[key]["suss"]
            total = self.suss_stat[key]["total"]
            if total > 0:
                print("on file={}: total {:d}, suss {:d}, rate {:.2f}".
                      format(key, total, suss, suss / total * 100))
