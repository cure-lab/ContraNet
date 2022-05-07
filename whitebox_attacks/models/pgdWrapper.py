import torch
from .baseWrapper import BaseWrapper
import random


class Pgd(BaseWrapper):
    def __init__(self, net, criterion, cls_norm, iters=20, alpha=0.005,
                 targeted=False, max_eps=0.08, x_val_min=0, x_val_max=1):
        super(Pgd, self).__init__(net, criterion, cls_norm, targeted,
                                  x_val_min, x_val_max)
        self.max_eps = max_eps
        self.iters = iters
        self.alpha = alpha
        self.suss_stat = {i * 0.01: {"suss": 0, "total": 0}
                          for i in range(2, int(self.max_eps // 0.01) + 1)}

    def get_random_start(self, x0, epsilon):
        return x0 + (torch.rand_like(x0) * 2 * epsilon + epsilon)

    def project(self, x, x0, epsilon):
        return x0 + torch.clamp(x - x0, -epsilon, epsilon)

    def attack(self, x0, y, device, training=False):
        if x0.min() < self.x_val_min or x0.max() > self.x_val_max:
            ValueError("Input data should in the range of [{}, {}]".format(
                self.x_val_min, self.x_val_max
            ))
        self.net.eval()
        x0 = x0.to(device)
        y = y.to(device)
        if training:
            eps = random.randint(2, self.max_eps // 0.01) * 0.01
        else:
            eps = 0.03
        x = self.get_random_start(x0, eps)
        with torch.enable_grad():
            for _ in range(self.iters):
                x.requires_grad_()
                out_adv = self.net(self.cls_norm(x))
                self.net.zero_grad()
                loss = self.criterion(out_adv, y)
                loss.backward()

                x = x + self.alpha * x.grad.sign()
                x = self.project(x, x0, eps)
                x = torch.clamp(x, min=self.x_val_min,
                                max=self.x_val_max).detach_()
        x_adv = x.detach()
        with torch.no_grad():
            h_adv = self.net(self.cls_norm(x_adv))
            _, cls_pred = h_adv.max(1)
            if training:
                suss = (cls_pred != y).sum().item()
                self.suss_stat[eps]["suss"] += suss
                self.suss_stat[eps]["total"] += x0.shape[0]

        return x_adv.detach(), cls_pred.detach()

    def print_stat(self):
        for eps in self.suss_stat:
            suss = self.suss_stat[eps]["suss"]
            total = self.suss_stat[eps]["total"]
            if total > 0:
                print("on eps={:.2f}: total {:d}, suss {:d}, rate {:.2f}".
                      format(eps, total, suss, suss / total * 100))


if __name__ == "__main__":
    import sys
    sys.path.append(".")
    from models.densenet import *
    from misc.load_dataset import LoadDataset
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    classifier = densenet169()
    classifier.load_state_dict(
        torch.load("pretrain/classifier/densenet169.pt"))
    cls_norm = [(0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)]
    cri = torch.nn.CrossEntropyLoss()
    classifier = classifier.cuda()
    attacker = Pgd(classifier, cri, cls_norm)

    test_data = LoadDataset(
        "cifar10", "./dataset", train=False, download=False,
        resize_size=32, hdf5_path=None, random_flip=False, norm=False)
    test_loader = DataLoader(
        test_data, batch_size=16, shuffle=False, num_workers=4,
        pin_memory=True)

    suss_num, fail_num = 0, 0
    for idx, (img, classId) in tqdm(enumerate(test_loader)):
        x_adv_suss, pred_y_suss, _, x_adv_fail, pred_y_fail = \
            attacker.adv_by_suss(img, classId, "cuda", training=True)
        suss_num += x_adv_suss.shape[0]
        fail_num += x_adv_fail.shape[0]
        if idx > 10:
            break
    attacker.print_stat()
    print("suss: ", suss_num, "fail: ", fail_num)
