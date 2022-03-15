import torch
import torch.nn.functional as F

cfgs_flag_common = [
    "z_dim", "shared_dim", "img_size", "g_conv_dim", "g_spectral_norm",
    "attention", "attention_after_nth_gen_block", "activation_fn",
    "conditional_strategy", "num_classes", "g_init", "G_depth"]
cfgs_flag_pretrain = [
    "G_weights", "E_weights", "V_weights"]
pars_flag_common = ["feature_num", "feature_m", "class_num", "pretrain"]


def check_params(cfgs, pars, cfgs_flag=None, pars_flag=None):
    cfgs_flag = cfgs_flag_common + cfgs_flag_pretrain \
        if cfgs_flag is None else cfgs_flag
    pars_flag = pars_flag_common if pars_flag is None else pars_flag
    if cfgs is not None:
        for flag in cfgs_flag:
            assert hasattr(cfgs, flag), "no {} in cfgs".format(flag)
    if pars is not None:
        for flag in pars_flag:
            assert hasattr(pars, flag), "no {} in pars".format(flag)
    return True


def concat(feat_emb, feat_gen):
    pair = torch.cat([feat_emb, feat_gen], dim=1)
    return pair


def sub(feat_emb, feat_gen):
    pair = feat_emb - feat_gen
    return pair


def loss_hinge_dis(dis_out_real, dis_out_fake, test=False):
    if test:
        return torch.mean(F.relu(1. - dis_out_real)), \
            torch.mean(F.relu(1. + dis_out_fake))
    else:
        return torch.mean(F.relu(1. - dis_out_real)) + \
            torch.mean(F.relu(1. + dis_out_fake))


def judge_thresh(l2_out, thresh, min_distance=False):
    # True for pass, False for reject
        if min_distance:
            return (l2_out < thresh).long()
        else:
            return (l2_out > thresh).long()
