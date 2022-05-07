import torch
from .baseWrapper import BaseWrapper


class Fgsm(BaseWrapper):
    def __init__(self, net, criterion, cls_norm, targeted=False,
                 eps=0.03, x_val_min=0, x_val_max=1):
        super(Fgsm, self).__init__(net, criterion, cls_norm, targeted,
                                   x_val_min, x_val_max)
        self.eps = eps

    def attack(self, x, y, device, training=False):
        if x.min() < self.x_val_min or x.max() > self.x_val_max:
            ValueError("Input data should in the range of [{}, {}]".format(
                self.x_val_min, self.x_val_max
            ))
        self.net.eval()
        x = x.to(device)
        y = y.to(device)
        x.requires_grad_()
        grad = torch.zeros_like(x)
        with torch.enable_grad():
            norm_x = self.cls_norm(x)
            h_adv = self.net(norm_x)
            if self.targeted:
                cost = self.criterion(h_adv, y)
            else:
                cost = -self.criterion(h_adv, y)

        grad = torch.autograd.grad(cost, [x])[0].detach()

        grad = grad.sign()
        x_adv = x - self.eps * grad
        x_adv = torch.clamp(x_adv, self.x_val_min, self.x_val_max)

        with torch.no_grad():
            norm_x = self.cls_norm(x_adv)
            h_adv = self.net(norm_x)
            _, cls_pred = h_adv.max(1)

        return x_adv.detach(), cls_pred.detach()


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
    attacker = Fgsm(classifier, cri, cls_norm)

    test_data = LoadDataset(
        "cifar10", "./dataset", train=False, download=False,
        resize_size=32, hdf5_path=None, random_flip=False, norm=False)
    test_loader = DataLoader(
        test_data, batch_size=16, shuffle=False, num_workers=4,
        pin_memory=True)

    suss_num, fail_num = 0, 0
    for img, classId in tqdm(test_loader):
        x_adv_suss, pred_y_suss, x_adv_fail, pred_y_fail = \
            attacker.adv_by_suss(img, classId, "cuda")
        suss_num += x_adv_suss.shape[0]
        fail_num += x_adv_fail.shape[0]
    attacker.print_stat()
    print("suss: ", suss_num, "fail: ", fail_num)
