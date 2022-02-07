from typing import Dict

import os
import torch
import numpy as np
from tqdm import tqdm
import logging
import multiprocessing as mp
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
from models.model_utils import judge_thresh
from models.ContraNetDict import ContraNetDict

_DEBUG = False
_MULTI_PROCESS = False if _DEBUG else True


def refine_fpr_tpr(fpr, tpr, all_combinations, drop_rate=0.05):
    """sort and check effective pair of fpr and tpr

    Args:
        fpr (List): points of fpr
        tpr (List): points of tpr

    Returns:
        fpr_refine, tpr_refine (List, List): fpr and tpr after refine
    """
    assert len(fpr) == len(tpr)
    # sort fpr
    fpr = torch.tensor(fpr)
    fpr, inds = fpr.sort()
    # change order of thresholds and tpr accordingly
    thresholds_sort = torch.tensor(all_combinations)[inds]
    tpr = torch.tensor(tpr)[inds]
    # 1. initialize
    fpr_refine, tpr_refine = [], []
    last_fpr = fpr[0]
    tpr_group = [tpr[0].item()]
    thresholds_group = [thresholds_sort[0].tolist()]
    for i in range(1, len(fpr)):
        if last_fpr >= fpr[i]:
            # 2a. accumulate same fpr
            tpr_group.append(tpr[i].item())
            thresholds_group.append(thresholds_sort[i].tolist())
        else:
            # 2b. determine best tpr
            tpr_max = np.max(tpr_group)
            tpr_max_ind = np.argmax(tpr_group)
            if tpr_refine == [] or tpr_max >= tpr_refine[-1]:
                fpr_refine.append(last_fpr.item())
                tpr_refine.append(tpr_max)
                # check if fpr reach drop rate
                if last_fpr <= drop_rate:
                    thresholds_final = thresholds_group[tpr_max_ind]
                    tpr_final = tpr_max
            tpr_group = [tpr[i].item()]
            last_fpr = fpr[i]
            thresholds_group = [thresholds_sort[i].tolist()]
    # 3. handle last group
    fpr_refine.append(last_fpr.item())
    tpr_refine.append(np.max(tpr_group))
    return fpr_refine, tpr_refine, thresholds_final, tpr_final


def tpr_fpr_single_attack(
        a_name, min_dists, this_score, this_y, all_combinations, d_names):
    P = this_y.sum()
    N = len(this_y) - P
    logging.info("[{}] P={}, N={}".format(a_name, P, N))
    fpr, tpr = [], []
    for thresh in tqdm(all_combinations):
        all_rej = torch.zeros_like(this_y)
        for idx, d_name in enumerate(d_names):
            this = judge_thresh(this_score[d_name], thresh[idx],
                                min_dists[d_name])
            all_rej = torch.logical_or(all_rej, 1 - this)
        # check fpr
        FP = torch.logical_and(this_y == 0, all_rej == 1).sum().item()
        # check tpr
        TP = torch.logical_and(this_y == 1, all_rej == 1).sum().item()
        fpr.append(FP / float(N))
        tpr.append(TP / float(P))
    return fpr, tpr


def tpr_fpr_process(a_name, min_dists, this_score, this_y, all_combinations,
                    d_names):
    logging.info("[{}] Start testing".format(a_name))
    fpr, tpr = tpr_fpr_single_attack(
        a_name, min_dists, this_score, this_y, all_combinations, d_names)
    # refine tpr and fpr
    fpr, tpr, final_thresh, tpr_final = refine_fpr_tpr(
        fpr, tpr, all_combinations)
    return {a_name: {"fpr": fpr, "tpr": tpr, "final_thresh": final_thresh,
                     "tpr_final": tpr_final}}


def plot_roc(score: Dict[str, Dict[str, torch.Tensor]],
             y: Dict[str, torch.LongTensor],
             thresholds: Dict[str, torch.Tensor], model: ContraNetDict,
             save_name: str):
    """plot roc curve.
    expect all clean samples can be classified correctly;
    expect all AE samples can attack successfully.

    Args:
        score (Dict[attack, Dict[d_name, score]]):
            dict of scores from each detector
        y (Dict[attack, torch.LongTensor]):
            1 for p/real AE sample; 0 for n/real clean sample
        thresholds (Dict[d_name, torch.Tensor]):
            dict of thresholds to each detecor
        model (ContraNetDict): detection dict model
        save_name (str): file to save
    """
    def merge(listA, listB):
        results = []
        for i in listA:
            for j in listB:
                if isinstance(i, list):
                    results.append(i + [j])
                else:
                    results.append([i, j])
        return results

    # get all combinations for thresholds
    d_names = list(thresholds.keys())
    all_combinations = [[i] for i in thresholds[d_names[0]].tolist()]
    for idx in range(len(d_names) - 1):
        all_combinations = merge(
            all_combinations, thresholds[d_names[idx + 1]].tolist())
    logging.info("Total {} combinations to test".format(len(all_combinations)))
    # test for each attack
    min_dists = {d_n: model.detector[d_n].min_distance for d_n in d_names}
    all_auc, final_thresholds = [], []
    pool = mp.Pool(6)
    procs = []
    results = {}
    for attack in score.keys():
        this_score = score[attack]
        this_y = y[attack]
        if _MULTI_PROCESS:
            procs.append(pool.apply_async(
                tpr_fpr_process,
                args=(attack, min_dists, this_score,
                      this_y, all_combinations, d_names)
            ))
        else:
            this_res = tpr_fpr_process(attack, min_dists, this_score,
                                       this_y, all_combinations, d_names)
            results.update(this_res)
    if _MULTI_PROCESS:
        for proc in procs:
            results.update(proc.get())
    # set random color before draw
    if len(score.keys()) > 10:
        colormap = plt.cm.nipy_spectral  # nipy_spectral, Set1, Paired
    else:
        colormap = plt.get_cmap("tab10")  # defualt color
    for idx, attack in enumerate(score.keys()):
        tpr = results[attack]["tpr"]
        fpr = results[attack]["fpr"]
        tpr_final = results[attack]["tpr_final"]
        final_thresh = results[attack]["final_thresh"]
        logging.info("[{}] thresh at drop rate: {}, tpr={:.4f}".format(
            attack, final_thresh, tpr_final))
        logging.info("[{}] {} points after refine".format(attack, len(fpr)))
        roc_auc = metrics.auc(fpr, tpr)
        all_auc.append(roc_auc)
        final_thresholds += final_thresh
        logging.info("[{}] roc_auc = {:.4f}".format(attack, roc_auc))
        logging.info("[{}] Done".format(attack))
        color = colormap(idx / len(score.keys())) if len(score.keys()) > 10 \
            else colormap(idx)
        plt.plot(fpr, tpr, label="{} auc={:.4f}".format(
            attack, roc_auc), color=color)

    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.title(os.path.basename(save_name).split(".")[0])
    plt.legend()
    plt.savefig(save_name)
    plt.close()
    return all_auc, final_thresholds


def collect_clean(data_loader, fmodel, model: ContraNetDict):
    score_dict = {key: torch.Tensor() for key in model.detector.keys()}
    for idx, (img, classId) in enumerate(data_loader):
        img = img.cuda()
        classId = classId.cuda()
        cls_pred = fmodel(img).argmax(axis=-1)
        cls_cor = (cls_pred == classId).byte().cpu()
        filtered_idx = torch.where(cls_cor)
        # restrict only to those correctly classified sample.
        if len(filtered_idx) == 0:
            continue
        score_dict_temp = model(img[filtered_idx])
        for key in score_dict:
            score_dict[key] = torch.cat(
                [score_dict[key], score_dict_temp[key].cpu()], dim=0)
        if _DEBUG and idx > 5:
            break
    return score_dict


def collect_sample(all_sample, model, fmodel, attack):
    loader = zip(all_sample["x_ori"], all_sample["y_ori"])
    parameters = all_sample["x_adv"].keys()
    score_dict = {"{}_{:.2f}".format(attack, param): {
        key: torch.Tensor() for key in model.detector.keys()
    } for param in parameters}
    for idx, (img, classId) in enumerate(loader):
        img = img.cuda()
        classId = classId.cuda()
        cls_pred = fmodel(img).argmax(axis=-1)
        cls_cor = (cls_pred == classId).byte().cpu()
        for i, param in enumerate(parameters):
            param_key = "{}_{:.2f}".format(attack, param)
            x_adv = all_sample["x_adv"][param][idx].cuda()
            y_adv_cls = fmodel(x_adv).argmax(axis=-1)
            attack_suss = (y_adv_cls != classId).cpu()
            filtered_idx = torch.where(torch.logical_and(cls_cor, attack_suss))
            # restrict only to those suss attack samples
            if len(filtered_idx[0]) == 0:
                continue
            score_dict_temp = model(x_adv[filtered_idx])
            for key in score_dict_temp:
                score_dict[param_key][key] = torch.cat(
                    [score_dict[param_key][key], score_dict_temp[key].cpu()],
                    dim=0)
        if _DEBUG and idx > 5:
            break
    return score_dict


def collect_sample_aa(data_loader, x_adv, model, classifier, batch_size,
                      model_name):
    score_dict = {model_name: {
        key: torch.Tensor() for key in model.detector.keys()}}
    total_number = 0
    for idx, (img, classId) in enumerate(data_loader):
        img = img.cuda()
        classId = classId.cuda()
        x_adv_img = x_adv[total_number: total_number + batch_size].cuda()
        total_number += img.shape[0]

        cls_pred = classifier(img).argmax(axis=-1)
        cls_cor = (cls_pred == classId).byte().cpu()

        y_adv_cls = classifier(x_adv_img).argmax(axis=-1)
        attack_suss = (y_adv_cls != classId).cpu()
        filtered_idx = torch.where(torch.logical_and(cls_cor, attack_suss))
        # restrict only to those suss attack samples
        if len(filtered_idx[0]) == 0:
            continue
        score_dict_temp = model(x_adv_img[filtered_idx])
        for key in score_dict_temp:
            score_dict[model_name][key] = torch.cat(
                [score_dict[model_name][key], score_dict_temp[key].cpu()],
                dim=0)
        if _DEBUG and idx > 5:
            break
        if total_number >= 2400:
            break
    return score_dict


def update_with_clean(score_dict, clean_dict):
    y_dict = {}
    for attack in score_dict:
        key = list(score_dict[attack].keys())[0]
        y_dict[attack] = torch.cat([
            torch.ones(score_dict[attack][key].shape, dtype=torch.long),
            torch.zeros(clean_dict[key].shape, dtype=torch.long)
        ], dim=0)
        for detector in score_dict[attack]:
            score_dict[attack][detector] = torch.cat([
                score_dict[attack][detector], clean_dict[detector]
            ], dim=0)
    return score_dict, y_dict
