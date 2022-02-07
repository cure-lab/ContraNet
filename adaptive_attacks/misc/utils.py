import torch
import math
from itertools import chain
from misc.diff_aug import DiffAugment
import random
import warnings
import glob
import os
import logging
import numpy as np
import foolbox as fb


def make_logger(run_name, log_output):
    if log_output is not None:
        run_name = log_output.split('/')[-1].split('.')[0]
    logger = logging.getLogger()  # get and set root logger
    logger.propagate = False
    log_filepath = log_output if log_output is not None else os.path.join(
        'results/log', f'{run_name}.log')

    log_dir = os.path.dirname(os.path.abspath(log_filepath))
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    if not logger.handlers:  # execute only if logger doesn't already exist
        file_handler = logging.FileHandler(log_filepath, 'a', 'utf-8')
        stream_handler = logging.StreamHandler(os.sys.stdout)

        formatter = logging.Formatter(
            '[%(levelname)s] %(asctime)s > %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S')

        file_handler.setFormatter(formatter)
        stream_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)
        logger.setLevel(logging.INFO)
    return logger


def generate1(img, label, device, encoder, vae, gen, next=False, policy="",
              class_num=10):
    warnings.warn("Note: This generate is for {} classes".format(class_num))
    with torch.no_grad():
        img = img.to(device)
        img = DiffAugment(img, policy=policy)

        latent_i = encoder(img).to(device)

        y = label.to(device)
        _, _, z = vae(latent_i)
        img_pos = gen(z.to(device), y)
        img_pos = DiffAugment(img_pos, policy=policy)

        if next:
            Int_Modi = 1
        else:
            Int_Modi = random.randint(1, int(class_num - 1))
        wrong_y = ((y + Int_Modi) % int(class_num)).to(device)
        img_neg = gen(z.to(device), wrong_y)
        img_neg = DiffAugment(img_neg, policy=policy)

    return img, img_pos, img_neg, y, wrong_y


def gen_with_pn(img, pos_label, neg_label, device, encoder, vae, gen):
    with torch.no_grad():
        img = img.to(device)
        latent_i = encoder(img).to(device)
        _, _, z = vae(latent_i)

        y = pos_label.to(device)
        img_pos = gen(z.to(device), y)

        wrong_y = neg_label.to(device)
        img_neg = gen(z.to(device), wrong_y)
    return img, img_pos, img_neg, y, wrong_y


def gen_with_label(img, label, device, encoder, vae, gen):
    with torch.no_grad():
        img = img.to(device)
        latent_i = encoder(img).to(device)
        _, _, z = vae(latent_i)

        y = label.to(device)
        img_gen = gen(z.to(device), y)
    return img, img_gen, y


def inference_1_pair(mlpModel, pair, y, pos, test=False, policy=None):
    if policy is not None:
        pair = DiffAugment(pair, policy=policy)
    gt_pair = torch.zeros(pair.shape[0], dtype=torch.long)
    if pos:
        gt_pair += 1
    if test:
        mlpModel.eval()
    else:
        mlpModel.train()
    pred = mlpModel((pair, y))
    assert pred.shape[1] == 2
    return gt_pair, pred


def inference_2_mlp(mlpModel, feat_emb, feat, y, pos, test=False, policy=None):
    pair = torch.cat([feat_emb, feat], dim=1)
    gt_pair, pred = inference_1_pair(
        mlpModel, pair, y, pos, test=test, policy=policy)
    return gt_pair, pred


def inference_2_mlp_m(
        mlpModel, feat_emb, feat, y, pos, test=False, policy=None):
    raise NotImplementedError()


def inference_2_pair(mlpModel, pos_pair, neg_pair, y_pos, y_neg, test=False,
                     policy=None):
    if policy is not None:
        pos_pair = DiffAugment(pos_pair, policy=policy)
        neg_pair = DiffAugment(neg_pair, policy=policy)
    gt_pair = torch.cat([
        torch.ones(pos_pair.shape[0], dtype=torch.long),
        torch.zeros(neg_pair.shape[0], dtype=torch.long)
    ], dim=0)
    if test:
        mlpModel.eval()
    else:
        mlpModel.train()
    pos_pred = mlpModel((pos_pair, y_pos))
    neg_pred = mlpModel((neg_pair, y_neg))
    pred = torch.cat([pos_pred, neg_pred], dim=0)
    assert pred.shape[1] == 2
    return gt_pair, pred


def inference_4_mlp(mlpModel, feat_emb_pos, feat_emb_neg, feat_pos, feat_neg,
                    y, wrong_y, test=False, policy=None):
    pos_pair = torch.cat([feat_emb_pos, feat_pos], dim=1)
    neg_pair = torch.cat([feat_emb_neg, feat_neg], dim=1)
    gt_pair, pred = inference_2_pair(
        mlpModel, pos_pair, neg_pair, y, wrong_y, test=test, policy=policy)
    return gt_pair, pred


def inference_4_mlp_m(mlpModel, feat_emb_pos, feat_emb_neg, feat_pos, feat_neg,
                      y, wrong_y, test=False, policy=None):
    raise NotImplementedError()


def inference_mlp(mlpModel, feat_emb, feat_pos, feat_neg, y, wrong_y,
                  test=False, policy=None, aux_data=None):
    pos_pair = torch.cat([feat_emb, feat_pos], dim=1)
    neg_pair = torch.cat([feat_emb, feat_neg], dim=1)
    if aux_data is not None:
        aux_pair = torch.cat([
            aux_data["feat_aux_emb"],
            aux_data["feat_aux_pos"]
        ], dim=1)
        pos_pair = torch.cat([pos_pair, aux_pair], dim=0)
        y_pos = torch.cat([y, aux_data["aux_y"]], dim=0)
    else:
        y_pos = y
    gt_pair, pred = inference_2_pair(
        mlpModel, pos_pair, neg_pair, y_pos, wrong_y, test=test, policy=policy)
    return gt_pair, pred


def inference_mlp_m(mlpModel, feat_emb, feat_pos, feat_neg, y, wrong_y,
                    test=False, policy=None, aux_data=None):
    pos_pair = feat_emb - feat_pos
    neg_pair = feat_emb - feat_neg
    if aux_data is not None:
        aux_pair = aux_data["feat_aux_emb"] - aux_data["feat_aux_pos"]
        pos_pair = torch.cat([pos_pair, aux_pair], dim=0)
        y_pos = torch.cat([y, aux_data["aux_y"]], dim=0)
    else:
        y_pos = y
    gt_pair, pred = inference_2_pair(
        mlpModel, pos_pair, neg_pair, y_pos, wrong_y, test=test, policy=policy)
    return gt_pair, pred


def flatten_dict(init_dict):
    res_dict = {}
    if type(init_dict) is not dict:
        return res_dict

    for k, v in init_dict.items():
        if type(v) == dict:
            res_dict.update(flatten_dict(v))
        else:
            res_dict[k] = v
    return res_dict


def setattr_cls_from_kwargs(cls, kwargs):
    kwargs = flatten_dict(kwargs)
    for key in kwargs.keys():
        value = kwargs[key]
        setattr(cls, key, value)


def dict2clsattr(train_configs, model_configs):
    cfgs = {}
    for k, v in chain(train_configs.items(), model_configs.items()):
        cfgs[k] = v

    class cfg_container:
        pass
    cfg_container.train_configs = train_configs
    cfg_container.model_configs = model_configs
    setattr_cls_from_kwargs(cfg_container, cfgs)
    return cfg_container


def save_best(best_prec, prefix, acc, params, epoch, arch, results_dir,
              min_mode=False):
    on_save = acc <= best_prec if min_mode else acc >= best_prec
    if on_save:
        best_prec = acc
        for file in glob.glob(
                results_dir + '/{}_{}E*'.format(arch, prefix)):
            os.remove(file)
        torch.save(
            params, results_dir +
            '/{}_{}E{}V{:.2f}.pth'.format(arch, prefix, epoch, acc))
        print(">>>>>> Best saved <<<<<<")
    else:
        print(">>>>>> Best not change from {} <<<<<<".format(best_prec))
    return best_prec


def attack_helper(data_loader, model, fmodel, params, get_adv, prefix,
                  save_dir=None, save_prefix=None, save=False, no_test=False):
    dec_cor_batch = [0] * (len(params) + 1)
    cor_batch = [0] * len(params)
    dis_cor_batch = [0] * len(params)
    dm_cor_batch = [0] * len(params)
    total_number = 0
    total_adv_acc = [0] * len(params)
    if save:
        all_sample = {"x_ori": [], "y_ori": [],
                      "x_adv": {eps: [] for eps in params}}
        save_name = ""
    for idx, (img, classId) in enumerate(data_loader):
        if save:
            all_sample["x_ori"].append(img)
            all_sample["y_ori"].append(classId)
        img = img.cuda()
        classId = classId.cuda()
        total_number += img.shape[0]
        acc_of_classifier = fb.utils.accuracy(fmodel, img, classId)
        logging.info("cls acc of this batch is:{}, total num {}".format(
            acc_of_classifier, total_number))
        cls_pred = fmodel(img).argmax(axis=-1)
        cls_cor = (cls_pred == classId).byte().cpu()
        dec_cor_batch[-1] += cls_cor.sum().item()
        for i, param in enumerate(params):
            logging.info('==========param={}============'.format(param))

            x_adv = get_adv(param, idx, img, classId)
            if save:
                all_sample["x_adv"][param].append(x_adv.cpu())
            if not no_test:
                _, dis_rej, dm_rej, both_rej, logits_cls = model(x_adv)
                y_adv_cls = logits_cls.argmax(dim=1)
                should_rej = (y_adv_cls != classId).cpu()

                correct_rej = (should_rej == both_rej.cpu())
                cor_batch[i] += correct_rej.sum()

                correct_rej_dis = (should_rej == dis_rej.cpu())
                dis_cor_batch[i] += correct_rej_dis.sum()

                correct_rej_dm = (should_rej == dm_rej.cpu())
                dm_cor_batch[i] += correct_rej_dm.sum()

                detect_cor = torch.logical_and(cls_cor, torch.logical_or(
                    should_rej == 0, both_rej.cpu()
                ))
                dec_cor_batch[i] += detect_cor.sum().item()

                logging.info("groudtruth  :{}".format(classId.cpu()))
                logging.info("adv cls pred:{}".format(y_adv_cls.cpu()))
                logging.info("dis rej is  :{}".format(dis_rej.cpu()))
                logging.info("dm rej is   :{}".format(dm_rej.byte().cpu()))
                logging.info("both rej is :{}".format(both_rej.byte().cpu()))
                logging.info("acc_{}dis :{}".format(
                    prefix, np.array(dis_cor_batch) / total_number))
                logging.info("acc_{}dm  :{}".format(
                    prefix, np.array(dm_cor_batch) / total_number))
                logging.info("acc_{}    :{}".format(
                    prefix, np.array(cor_batch) / total_number))
                logging.info("acc_{}_dec:{}".format(
                    prefix, np.array(dec_cor_batch)[:-1] / dec_cor_batch[-1]))
            else:
                adv_acc = fb.utils.accuracy(fmodel, x_adv, classId)
                total_adv_acc[i] += adv_acc * x_adv.shape[0]
                logging.info("adv cls acc of this batch is:{}".format(adv_acc))
        logging.info("total adv cls acc is:{}".format(
            np.array(total_adv_acc) / total_number))
        if save:
            if save_name != "":
                os.remove(save_name)
            save_name = os.path.join(
                save_dir, "{}_{}.pt".format(save_prefix, total_number))
            torch.save(all_sample, save_name)
            if total_number > 2400:
                break


def test_clean(data_loader, model):
    dec_cor_batch = [0] * (len(model) + 2)
    cor_rej_batch = [0] * len(model)
    cor_batch = 0
    total_number = 0
    total_rej = 0
    for idx, (img, classId) in enumerate(data_loader):
        img = img.cuda()
        classId = classId.cuda()
        total_number += img.shape[0]

        rej_dict, all_rej, logits_cls = model(img)
        cls_pred = logits_cls.argmax(dim=1)
        should_rej = (cls_pred != classId).cpu()
        cls_cor = (cls_pred == classId).byte().cpu()
        dec_cor_batch[-1] += cls_cor.sum().item()
        total_rej += all_rej.sum().item()

        # robust acc: on clean is (TP+TN)/(P+N)
        # clean acc for classifier on all sample
        correct_rej = (should_rej == all_rej.cpu())
        cor_batch += correct_rej.sum()
        for idx, key in enumerate(rej_dict):
            correct_rej_temp = (should_rej == rej_dict[key].cpu())
            cor_rej_batch[idx] += correct_rej_temp.sum().item()

        # acc for (reject correctly classifier) / clean correct = FPR
        detect_cor2 = torch.logical_and(cls_cor, all_rej.cpu())
        dec_cor_batch[-2] += detect_cor2.sum().item()
        for idx, key in enumerate(rej_dict):
            detect_cor_temp = torch.logical_and(
                cls_cor, rej_dict[key].cpu())
            dec_cor_batch[idx] += detect_cor_temp.sum().item()

        rob_acc = np.array([cor_batch]) / total_number
        _FPR = np.array(dec_cor_batch)[:-1] / dec_cor_batch[-1]
        logging.info("groudtruth  :{}".format(classId.cpu()))
        logging.info("cls pred    :{}".format(cls_pred.cpu()))
        logging.info("all rej is  :{}".format(all_rej.byte().cpu()))
        logging.info("rob_acc self:{}".format(
            np.array(cor_rej_batch) / total_number))
        logging.info("rob_acc     :{}".format(rob_acc))
        logging.info("acc cls     :{}".format(
            np.array([dec_cor_batch[-1]]) / total_number))
        logging.info("detectors   :{}".format(rej_dict.keys()))
        logging.info("FPR         :{}".format(_FPR))
        logging.info("Rej clean   :{}".format(
            np.array(total_rej) / total_number))
    # final_results = (np.array(cor_rej_batch) / total_number).tolist()
    # final_results = "/".join(["{:.4f}".format(i) for i in final_results])
    rob_acc = "{:.4f}".format(rob_acc.tolist()[0])
    fpr = "\t".join(["{:.4f}".format(i) for i in _FPR.tolist()])
    return rob_acc, fpr


def attack_sample(all_sample, model, fmodel, prefix, batch_size=None):
    if batch_size is None:
        batch_size = all_sample["x_ori"][0].shape[0]
    x_ori = torch.cat(all_sample["x_ori"], dim=0)
    y_ori = torch.cat(all_sample["y_ori"], dim=0)
    parameters = all_sample["x_adv"].keys()
    x_adv_all = {key: torch.cat(all_sample["x_adv"][key], dim=0)
                 for key in all_sample["x_adv"]}

    tp_batch = [0] * len(parameters)
    tp_fn = [0] * len(parameters)
    single_tp_batch = {key: [0] * len(parameters) for key in model.keys()}
    single_incor_batch = {key: [0] * len(parameters) for key in model.keys()}
    incor_batch = [0] * len(parameters)
    total_number = 0
    rob_single_str = [["" for _ in model.keys()]
                      for _ in range(len(parameters))]
    tpr_single_str = [["" for _ in model.keys()]
                      for _ in range(len(parameters))]
    for idx in range(math.ceil(len(x_ori) / batch_size)):
        start = idx * batch_size
        end = start + batch_size
        img = x_ori[start:end].cuda()
        classId = y_ori[start:end].cuda()
        total_number += img.shape[0]
        acc_of_classifier = fb.utils.accuracy(fmodel, img, classId)
        logging.info("cls acc of this batch is:{}, total num {}".format(
            acc_of_classifier, total_number))
        cls_pred = fmodel(img).argmax(axis=-1)
        cls_cor = (cls_pred == classId).byte().cpu()
        for i, param in enumerate(parameters):
            logging.info('==========param={}============'.format(param))

            x_adv = x_adv_all[param][start:end].cuda()
            rej_dict, all_rej, logits_cls = model(x_adv)
            y_adv_cls = logits_cls.argmax(dim=1)
            should_rej = (y_adv_cls != classId).cpu()

            # attack suss rate: robust acc is
            # 1 - #(pass and incorrect samples)/#(all perturbed samples)
            # here is the #(pass and incorrect samples)
            incor_pass = torch.logical_and(should_rej, all_rej.cpu() == 0)
            incor_batch[i] += incor_pass.sum().item()
            for key in rej_dict:
                incorrect_pass_temp = torch.logical_and(
                    should_rej == 1, rej_dict[key].cpu() == 0)
                single_incor_batch[key][i] += incorrect_pass_temp.sum().item()

            # TPR: acc for (attack suss and reject) / attack suss
            detect_cor = torch.logical_and(
                torch.logical_and(cls_cor, should_rej), all_rej.cpu()
            )
            tp_batch[i] += detect_cor.sum().item()
            suss_attack = torch.logical_and(cls_cor, should_rej)
            tp_fn[i] += suss_attack.sum().item()
            for key in rej_dict:
                detect_cor_temp = torch.logical_and(
                    torch.logical_and(cls_cor, should_rej), rej_dict[key].cpu())
                single_tp_batch[key][i] += detect_cor_temp.sum().item()

            rob_acc = 1. - np.array(incor_batch) / total_number
            _TPR = np.array(tp_batch) / np.array(tp_fn)
            logging.info("groudtruth  :{}".format(classId.cpu()))
            logging.info("adv cls pred:{}".format(y_adv_cls.cpu()))
            logging.info("all rej is  :{}".format(all_rej.byte().cpu()))

            for ikey, key in enumerate(single_incor_batch):
                rob_this = 1 - np.array(single_incor_batch[key]) / total_number
                logging.info("rob_{}{}:{}".format(prefix, key, rob_this))
                rob_single_str[i][ikey] = "{:.4f}".format(rob_this[i])
            logging.info("robAcc_{} :{}".format(prefix, rob_acc))

            for ikey, key in enumerate(single_tp_batch):
                tpr_this = np.array(single_tp_batch[key]) / np.array(tp_fn)
                logging.info("TPR_{}{}:{}".format(prefix, key, tpr_this))
                tpr_single_str[i][ikey] = "{:.4f}".format(tpr_this[i])
            logging.info("TPR_{}    :{}".format(prefix, _TPR))
    rob_single = "Rob:\n"
    tpr_single = "Tpr:\n"
    for istr in range(len(rob_single_str)):
        rob_single += "\t".join(rob_single_str[istr]) + "\n"
        tpr_single += "\t".join(tpr_single_str[istr]) + "\n"
    logging.info("single statistics\n" + rob_single + tpr_single)
    rob_acc = "\t".join(["{:.4f}".format(i) for i in rob_acc.tolist()])
    tpr = "\t".join(["{:.4f}".format(i) for i in _TPR.tolist()])
    return rob_acc, tpr


def attack_sample_aa(data_loader, x_adv, model, classifier, batch_size):
    prefix = "_AA_"
    cls_adv_cor_num = 0
    tp_batch = [0]
    tp_fn = [0]
    single_tp_batch = {key: [0] for key in model.keys()}
    single_incor_batch = {key: [0] for key in model.keys()}
    incor_batch = [0]
    total_number = 0
    for idx, (img, classId) in enumerate(data_loader):
        img = img.cuda()
        classId = classId.cuda()
        x_adv_img = x_adv[total_number: total_number + batch_size].cuda()
        total_number += img.shape[0]

        cls_pred = classifier(img).argmax(axis=-1)
        cls_cor = (cls_pred == classId).byte().cpu()
        acc_of_classifier = cls_cor.sum().item() / len(cls_cor)
        logging.info("cls acc of this batch is:{}, total num {}".format(
            acc_of_classifier, total_number))

        rej_dict, all_rej, logits_cls = model(x_adv_img)
        y_adv_cls = logits_cls.argmax(dim=1)
        should_rej = (y_adv_cls != classId).cpu()
        cls_adv_cor_num += (should_rej == 0).sum().item()

        # attack suss rate: robust acc is
        # 1 - #(pass and incorrect samples)/#(all perturbed samples)
        # here is the #(pass and incorrect samples)
        incor_pass = torch.logical_and(should_rej, all_rej.cpu() == 0)
        incor_batch[0] += incor_pass.sum().item()
        for key in rej_dict:
            incorrect_pass_temp = torch.logical_and(
                should_rej == 1, rej_dict[key].cpu() == 0)
            single_incor_batch[key][0] += incorrect_pass_temp.sum().item()

        # TPR: acc for (attack suss and reject) / attack suss
        detect_cor = torch.logical_and(
            torch.logical_and(cls_cor, should_rej), all_rej.cpu()
        )
        tp_batch[0] += detect_cor.sum().item()
        suss_attack = torch.logical_and(cls_cor, should_rej)
        tp_fn[0] += suss_attack.sum().item()
        for key in rej_dict:
            detect_cor_temp = torch.logical_and(
                torch.logical_and(cls_cor, should_rej), rej_dict[key].cpu())
            single_tp_batch[key][0] += detect_cor_temp.sum().item()

        rob_acc = 1. - np.array(incor_batch) / total_number
        _TPR = np.array(tp_batch) / np.array(tp_fn)
        logging.info("groudtruth  :{}".format(classId.cpu()))
        logging.info("adv cls pred:{}".format(y_adv_cls.cpu()))
        logging.info("all rej is  :{}".format(all_rej.byte().cpu()))
        logging.info("acc cls adv :{}".format(cls_adv_cor_num / total_number))
        for key in single_incor_batch:
            logging.info(
                "rob_{}{}:{}".format(
                    prefix, key, 1 - np.array(
                        single_incor_batch[key]) / total_number))
        logging.info("robAcc_{} :{}".format(prefix, rob_acc))
        for key in single_tp_batch:
            logging.info("TPR_{}{}:{}".format(
                prefix, key, np.array(
                    single_tp_batch[key]) / np.array(tp_fn[0])))
        logging.info("TPR_{}    :{}".format(prefix, _TPR))
        if total_number >= 2400:
            break
    rob_acc = "\t".join(["{:.4f}".format(i) for i in rob_acc.tolist()])
    tpr = "\t".join(["{:.4f}".format(i) for i in _TPR.tolist()])
    return rob_acc, tpr
