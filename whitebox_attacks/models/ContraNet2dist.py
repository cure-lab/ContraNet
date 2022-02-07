import logging
import torch
from lib.pytorch_ssim import ssim
from .ContraNetDictBase import ContraNetDictBase
from .model_utils import check_params, judge_thresh


class ContraNet2dist(ContraNetDictBase):
    """ContraNet for cGAN + DML + classifier
    For inference Only.

    Args:
        cfgs: configuration for cGAN
        pars: configuration for DML
    """

    def __init__(self, cfgs, pars, classifier, cls_norm=None, model_norm=None,
                 distance="L2"):
        super(ContraNet2dist, self).__init__(
            cfgs, pars, classifier, cls_norm=cls_norm, model_norm=model_norm)
        assert check_params(cfgs, pars=None)
        self.distance_name = distance
        if distance == "L2":
            self.distance = lambda x, y: torch.norm(x - y, 2, dim=(1, 2, 3))
            self.min_distance = True
        elif distance == "L1":
            self.distance = lambda x, y: torch.abs(x - y).sum(dim=(1, 2, 3))
            self.min_distance = True
        elif distance == "ssim":
            self.distance = lambda x, y: ssim(x, y, size_average=False)
            self.min_distance = False
        elif distance == "ssim-L2":
            self.distance = lambda x, y: ssim(
                x, y, size_average=False) - torch.norm(x - y, 2, dim=(1, 2, 3))
            self.min_distance = False
        else:
            raise NotImplementedError("not known distance: {}".format(distance))
        self.resume_weights(cfgs)
        self._thresh = None

    def judge_distance(self, l2_out):
        """True for pass, False for rejection.
        """
        if self._thresh is None:
            raise RuntimeError("You need to assign a threshold to judge")
        return judge_thresh(l2_out, self._thresh, self.min_distance)

    def get_distance(self, img_model, img_gen_dict, cls_pred, no_judge=False):
        """calculate L2 distance

        Args:
            img_model (torch.Tensor): input image batch
            img_gen_dict (dict): generated image dict
            cls_pred (torch.LongTensor): argmax from classifier output
        """
        img_gen_in = []
        for idx, pred_cls in enumerate(cls_pred):
            img_gen_in.append(img_gen_dict[pred_cls.item()][idx])
        img_gen_in = torch.stack(img_gen_in)
        distance = self.distance(img_gen_in, img_model)
        if no_judge:
            return distance, None
        else:
            dist_pred = self.judge_distance(distance)
            return distance, dist_pred

    def forward(self, img):
        """
        Returns:
            torch.Tensor: pred labels of size [N].
        """
        logits_cls, cls_pred = self.forward_classifier(img)

        img_model = self.model_norm(img)
        img_gen_dict = self.cGAN(img_model)
        if self._only_judge:
            distance, _ = self.get_distance(
                img_model, img_gen_dict, cls_pred, True)
            return distance
        distance, dist_pred = self.get_distance(
            img_model, img_gen_dict, cls_pred)

        rej = torch.zeros(img.shape[0])
        final_logits = []
        if not self._only_rej:
            raise NotImplementedError()
        for idx, dist_this in enumerate(dist_pred):  # im is a single image now.
            if dist_this != 1:
                rej[idx] = 1
        logging.info("[ContraNet2dist {}] Reject {} out of {} sampels".format(
            self.distance_name, rej.sum().item(), img.shape[0]))
        if self.return_fig:
            raise NotImplementedError()
        if self.debug or self.ddebug:
            if self.ddebug:
                return final_logits, rej, logits_cls
            else:
                return final_logits, rej
        else:
            raise NotImplementedError()

    def check_most(self, im, im_gen_dict):
        raise NotImplementedError()

    def get_judge_results(self, val_loader):
        all_dist = torch.Tensor()
        with torch.no_grad():
            for img, classId in val_loader:
                img = img.cuda()
                classId = classId.cuda()
                img_model = self.model_norm(img)
                img_gen_dict = self.cGAN(img_model)
                dist_out, _ = self.get_distance(
                    img_model, img_gen_dict, classId, no_judge=True)
                all_dist = torch.cat([all_dist, dist_out.cpu()])
        return all_dist

    def sort_results(self, out):
        if self.min_distance:
            out, inds = out.sort(descending=True)  # pos: max -> min
        else:
            out, inds = out.sort(descending=False)  # pos: min -> max
        return out, inds

    def get_thresh(self, val_loader, drop_rate=0.05):
        all_dist = self.get_judge_results(val_loader)
        all_dist, _ = self.sort_results(all_dist)
        thresh = all_dist[int(len(all_dist) * drop_rate)].item()
        return thresh

    def resume_weights(self, cfgs, pars=None):
        self.cGAN.resume_weights(cfgs, verbal="[ContraNet2dist {}] ".format(
            self.distance_name))

    def load_classifier(self, path, key='net'):
        checkpoint = torch.load(path)
        if key is not None:
            checkpoint = checkpoint[key]
        self.classifier.load_state_dict(checkpoint)
        logging.info("[ContraNet2dist {}] Loaded classifier from: {}".format(
            self.distance_name, path))
