import warnings
import logging
import torch
from .ContraNetDictBase import ContraNetDictBase
from .model_utils import check_params, concat, sub, judge_thresh
from .MobileNetV2 import MobileNetV2, MLP


class ContraNet2(ContraNetDictBase):
    """ContraNet for cGAN + DML + classifier
    For inference Only.

    Args:
        cfgs: configuration for cGAN
        pars: configuration for DML
    """

    def __init__(self, cfgs, pars, classifier, cls_norm=None, model_norm=None):
        super(ContraNet2, self).__init__(
            cfgs, pars, classifier, cls_norm=cls_norm, model_norm=model_norm)
        assert check_params(cfgs, pars)
        assert cfgs.num_classes == pars.class_num
        self.oridml = MobileNetV2(n_class=pars.feature_num)
        self.gendml = MobileNetV2(n_class=pars.feature_num)
        self.oridml.only_features()
        self.gendml.only_features()
        self.dml_lastchannel = self.oridml.last_channel
        if pars.feature_m:
            self.combine = sub
            self.mlpModel = MLP(
                self.dml_lastchannel, p=0.1, class_num=pars.class_num)
        else:
            self.mlpModel = MLP(
                self.dml_lastchannel * 2, p=0.1,
                class_num=pars.class_num)
            self.combine = concat
        self.class_num = pars.class_num
        self.resume_weights(cfgs, pars)
        self._thresh = 0.5

    def judge_mlp(self, mlp_out):
        """True for pass, False for rejection.
        """
        return judge_thresh(mlp_out, self._thresh, self.min_distance)

    def forward_mlp(self, img_model, img_gen_dict, cls_pred):
        img_gen_in = []
        for idx, pred_cls in enumerate(cls_pred):
            img_gen_in.append(img_gen_dict[pred_cls.item()][idx])
        img_gen_in = torch.stack(img_gen_in)
        pair = self.combine(
            self.oridml(img_model), self.gendml(img_gen_in)
        )
        mlp_out = self.mlpModel((pair, cls_pred))
        mlp_pred = self.judge_mlp(mlp_out[:, 1])
        return mlp_out, mlp_pred

    def forward(self, img):
        """
        Returns:
            torch.Tensor: pred labels of size [N].
        """
        if isinstance(img, tuple):
            (img, rej_pre) = img
        else:
            rej_pre = [False] * img.shape[0]
        logits_cls, cls_pred = self.forward_classifier(img)

        img_model = self.model_norm(img)
        img_gen_dict = self.cGAN(img_model)
        mlp_out, mlp_pred = self.forward_mlp(img_model, img_gen_dict, cls_pred)
        if self._only_judge:
            return mlp_out[:, 1]

        rej = torch.zeros(img.shape[0])
        final_logits = []
        mlp_out_refine = {}
        for idx, mlp_this in enumerate(mlp_pred):  # im is a single image now.
            if mlp_this == 1 and not rej_pre[idx]:
                final_logits.append(logits_cls[idx])
            else:
                if mlp_this != 1:
                    rej[idx] = 1
                if not self._only_rej:
                    im_gen_dict = {key: img_gen_dict[key][idx]
                                   for key in img_gen_dict}
                    if self.return_fig:
                        this_logits, this_mlp = self.check_most(
                            img_model[idx], im_gen_dict)
                        mlp_out_refine[idx] = this_mlp.cpu()
                    else:
                        this_logits = self.check_most(
                            img_model[idx], im_gen_dict)
                    final_logits.append(this_logits)
        if not self._only_rej:
            final_logits = torch.stack(final_logits)
        logging.info("[ContraNet2] Reject {} out of {} sampels".format(
            rej.sum().item(), img.shape[0]))
        if self.return_fig:
            for key in img_gen_dict:
                img_gen_dict[key] = img_gen_dict[key].cpu()
            dic = {
                "img_gen_dict": img_gen_dict,
                "cls_pred": cls_pred.cpu(),
                "final_logits": final_logits.cpu(),
                "mlp_out": mlp_out.cpu(),
                "mlp_out_refine": mlp_out_refine,
            }
            return dic
        if self.debug or self.ddebug:
            if self.ddebug:
                return final_logits, rej, logits_cls
            else:
                return final_logits, rej
        else:
            return final_logits

    def check_most(self, im, im_gen_dict):
        """Go through all labels and return the most possible one.

        Args:
            im (torch.Tensor): original image
            im_gen_dict (torch.Tensor): generated image

        Returns:
            torch.Tensor: pred label of shape [1]
        """
        device = im.device
        im_in, im_gen_in, label_in = [], [], []
        for label in range(self.class_num):
            im_gen_in.append(im_gen_dict[label])
            im_in.append(im)
            label_in.append(label)
        im_in = torch.stack(im_in)
        im_gen_in = torch.stack(im_gen_in)
        label_in = torch.LongTensor(label_in).to(device)
        pair = self.combine(self.oridml(im_in), self.gendml(im_gen_in))
        mlp_out = self.mlpModel((pair, label_in))  # size class_num x 2
        _, pred_label = mlp_out[:, 1].max(0)

        img_cls = self.cls_norm(self.model_denorm(
            im_gen_dict[pred_label.item()].unsqueeze(0)))
        logits_cls = self.classifier(img_cls)
        if self.return_fig:
            return logits_cls.squeeze(), mlp_out
        else:
            return logits_cls.squeeze()

    def get_judge_results(self, val_loader):
        mlp_out_pos = torch.Tensor()
        with torch.no_grad():
            for img, classId in val_loader:
                img = img.cuda()
                classId = classId.cuda()
                img_model = self.model_norm(img)
                img_gen_dict = self.cGAN(img_model)
                mlp_out, _ = self.forward_mlp(img_model, img_gen_dict, classId)
                mlp_out_pos = torch.cat([mlp_out_pos, mlp_out.cpu()[:, 1]])
        return mlp_out_pos

    def get_thresh(self, val_loader, drop_rate=0.05, return_list=False):
        if self._thresh != 0.5:
            warnings.warn(
                "You may have set thresh before calling this. get_thresh will use the given thresh")
        mlp_out_pos = self.get_judge_results(val_loader)
        mlp_out_pos_sort, _ = self.sort_results(mlp_out_pos)  # pos: min -> max
        thresh = mlp_out_pos_sort[int(len(mlp_out_pos_sort) * drop_rate)].item()
        if return_list:
            return thresh, mlp_out_pos
        else:
            return thresh

    def resume_weights(self, cfgs, pars):
        self.cGAN.resume_weights(cfgs, verbal="[ContraNet2] ")
        checkpoint = torch.load(pars.pretrain)
        self.oridml.load_state_dict(checkpoint['ori_state'])
        self.gendml.load_state_dict(checkpoint['gen_state'])
        self.mlpModel.load_state_dict(checkpoint['mlp_state'])
        logging.info("[ContraNet2] Loaded pretrain from: {}".format(
            pars.pretrain))

    def load_classifier(self, path, key='net'):
        checkpoint = torch.load(path)
        if key is not None:
            checkpoint = checkpoint[key]
        self.classifier.load_state_dict(checkpoint)
        logging.info("[ContraNet2] Loaded classifier from: {}".format(path))
