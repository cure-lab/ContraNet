import warnings
import logging
import torch
from .ContraNetDictBase import ContraNetDictBase
from .model_utils import check_params, cfgs_flag_common, judge_thresh
from .GANv2.big_resnet import Discriminator

pars_flag = ["class_num"]


class ContraNet2D(ContraNetDictBase):
    """ContraNet for cGAN + DML + classifier
    For inference Only.

    Args:
        cfgs: configuration for cGAN
        pars: configuration for DML
    """

    def __init__(self, cfgs, pars, classifier, cls_norm=None, model_norm=None):
        super(ContraNet2D, self).__init__(
            cfgs, pars, classifier, cls_norm=cls_norm, model_norm=model_norm)
        cfgs_flag = cfgs_flag_common
        assert check_params(cfgs, pars, pars_flag=pars_flag,
                            cfgs_flag=cfgs_flag)
        assert cfgs.num_classes == pars.class_num
        self.dis = Discriminator(
            cfgs.img_size, cfgs.d_conv_dim, cfgs.d_spectral_norm,
            cfgs.attention, cfgs.attention_after_nth_dis_block,
            cfgs.activation_fn, cfgs.conditional_strategy, cfgs.hypersphere_dim,
            cfgs.num_classes, cfgs.nonlinear_embed, cfgs.normalize_embed,
            cfgs.d_init, cfgs.D_depth, False)

        self.class_num = pars.class_num
        self.resume_weights(cfgs)
        self._thresh = 0.

    def judge_dis(self, dis_out):
        """True for pass, False for rejection.
        """
        return judge_thresh(dis_out, self._thresh, self.min_distance)

    def forward_dis(self, img_gen_dict, cls_pred):
        img_gen_in = []
        for idx, pred_cls in enumerate(cls_pred):
            img_gen_in.append(img_gen_dict[pred_cls.item()][idx])
        img_gen_in = torch.stack(img_gen_in)
        dis_out = self.dis(img_gen_in, cls_pred)
        dis_pred = self.judge_dis(dis_out)
        return dis_out, dis_pred

    def forward(self, img):
        """
        Returns:
            torch.Tensor: pred labels of size [N].
        """
        logits_cls, cls_pred = self.forward_classifier(img)

        img_model = self.model_norm(img)
        img_gen_dict = self.cGAN(img_model)
        # img_gen_in: generated img for predictied class, size: B x C x ...
        dis_out, dis_pred = self.forward_dis(img_gen_dict, cls_pred)
        if self._only_judge:
            return dis_out

        rej = torch.zeros(img.shape[0])
        final_logits = []
        dis_out_refine = {}
        if self.return_fig:
            dis_out_like = torch.zeros(dis_pred.shape[0], 2)
            # we only need [idx][:, 1].argmax() for dis_out_refine
            dis_out_refine = torch.zeros(dis_pred.shape[0], self.class_num, 2)
        for idx, dis_this in enumerate(dis_pred):  # im is a single image now.
            if dis_this:
                final_logits.append(logits_cls[idx])
                if self.return_fig:
                    dis_out_like[idx][1] = 1
            else:
                rej[idx] = 1
                if not self._only_rej:
                    im_gen_dict = {key: img_gen_dict[key][idx]
                                   for key in img_gen_dict}
                    im_gen_dict.pop(cls_pred[idx].item())
                    if self.return_fig:
                        this_logits, this_dis_label = self.check_most(
                            im_gen_dict, img_model.device)
                        dis_out_refine[idx][this_dis_label, 1] = 1
                    else:
                        this_logits = self.check_most(
                            im_gen_dict, img_model.device)
                    final_logits.append(this_logits)
                    if self.return_fig:
                        dis_out_like[idx][0] = 1
        if not self._only_rej:
            final_logits = torch.stack(final_logits)
        logging.info("[ContraNet2D] Reject {} out of {} sampels".format(
            rej.sum().item(), img.shape[0]))
        if self.return_fig:
            for key in img_gen_dict:
                img_gen_dict[key] = img_gen_dict[key].cpu()
            dic = {
                "img_gen_dict": img_gen_dict,
                "cls_pred": cls_pred.cpu(),
                "final_logits": final_logits.cpu(),
                "mlp_out": dis_out_like.cpu(),
                "mlp_out_refine": dis_out_refine,
            }
            return dic
        if self.ddebug:
            return final_logits, rej, logits_cls
        elif self.debug:
            return final_logits, rej
        else:
            return final_logits

    def check_most(self, im_gen_dict, device):
        """Go through all labels and return the most possible one.

        Args:
            im_gen_dict (torch.Tensor): generated image

        Returns:
            torch.Tensor: pred label of shape [1]
        """
        im_gen_in, label_in = [], []
        for label in im_gen_dict:
            im_gen_in.append(im_gen_dict[label])
            label_in.append(label)
        im_gen_in = torch.stack(im_gen_in)
        label_in = torch.LongTensor(label_in).to(device)
        # all data for each label except the rejection label
        dis_out = self.dis(im_gen_in, label_in)  # size class_num x 2
        pred_label = label_in[torch.argmax(dis_out).item()]

        img_cls = self.cls_norm(self.model_denorm(
            im_gen_dict[pred_label.item()].unsqueeze(0)))
        logits_cls = self.classifier(img_cls)
        if self.return_fig:
            return logits_cls.squeeze(), pred_label
        else:
            return logits_cls.squeeze()

    def get_judge_results(self, val_loader):
        dis_out_all = torch.Tensor()
        with torch.no_grad():
            for img, classId in val_loader:
                img = img.cuda()
                classId = classId.cuda()
                img_model = self.model_norm(img)
                img_gen_dict = self.cGAN(img_model)
                dis_out, _ = self.forward_dis(img_gen_dict, classId)
                dis_out_all = torch.cat([dis_out_all, dis_out.cpu()])
        return dis_out_all

    def get_thresh(self, val_loader, drop_rate=0.05, return_list=False):
        if self._thresh != 0.:
            warnings.warn(
                "You may have set thresh before calling this. get_thresh will use the given thresh")
        dis_out_all = self.get_judge_results(val_loader)
        dis_out_all_sort, _ = self.sort_results(dis_out_all)  # pos: min -> max
        thresh = dis_out_all_sort[int(len(dis_out_all_sort) * drop_rate)].item()
        if return_list:
            return thresh, dis_out_all
        else:
            return thresh

    def resume_weights(self, cfgs):
        self.cGAN.resume_weights(cfgs, verbal="[ContraNet2D] ")
        checkpoint = torch.load(cfgs.D_path)
        self.dis.load_state_dict(checkpoint['state_dict'])
        logging.info("[ContraNet2D] Loaded discriminator from: {}".format(
            cfgs.D_path))

    def load_classifier(self, path, key='net'):
        checkpoint = torch.load(path)
        if key is not None:
            checkpoint = checkpoint[key]
        self.classifier.load_state_dict(checkpoint)
        logging.info("[ContraNet2D] Loaded classifier from: {}".format(path))
