import logging
import sys
sys.path.append(".")
sys.path.append("lib")

import json
import argparse
import os
import torch
import time
import numpy as np
import foolbox as fb
import glob
from tools.roc_utils import *
from models.ContraNet2_2 import *
from models.ContraNet2_3 import *
from models.ContraNet2_4 import *
from models.ContraNetDict import ContraNetDict
from models.resnet import *
from models.densenet import *
from models.mnist2layer import *
import misc.utils as utils
from misc.load_dataset import LoadDataset
from torch.utils.data import DataLoader
np.set_printoptions(precision=4, suppress=True, linewidth=120)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', default="ContraNet2_3ssH")
    parser.add_argument(
        '-t', '--test',
        default=["PGD", "PGDL2", "BIM", "BIML2", "DF", "DFL2", "CW", "CWinf",
                 "EADA", "EADAL1"],
        nargs='*', type=str)
    parser.add_argument('-c', '--config_path', type=str,
                        default='config/configsCifar10GDnoise112000B.json')
    parser.add_argument(
        '--feature_m', action='store_true', help='use the minus of inputs')
    parser.add_argument(
        '--pretrain',
        default='results/cifar10AEPgd2/MobileNetV2-dense.01E.2_112000cifar10CondPgd-2021-05-24-12-35-50/MobileNetV2_AEbestE91V97.48.pth',
        type=str, metavar='PATH',
        help='path to load pretrain checkpoint (default: none)')
    parser.add_argument('--batch_size', default=16, type=int,
                        metavar='N', help='mini-batch size')
    parser.add_argument('--save_dir', type=str,
                        default='results/log/whitebox')
    parser.add_argument('--load_dir', type=str, default="")
    parser.add_argument('--log_dir', type=str, default='results/log/roc')
    parser.add_argument('--csv_file', type=str, default=None)
    args = parser.parse_args()

    if args.config_path is not None:
        with open(args.config_path) as f:
            model_configs = json.load(f)
        train_configs = vars(args)
    else:
        raise NotImplementedError()
    cfgs = utils.dict2clsattr(train_configs, model_configs)
    args.dataset_name = cfgs.dataset_name
    args.data_path = cfgs.data_path
    args.img_size = cfgs.img_size
    args.class_num = cfgs.num_classes
    if args.dataset_name == "cifar10":
        args.feature_num = 12
        classifier = densenet169()
        cls_norm = [(0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)]
        cls_key = None
    elif args.dataset_name == "gtsrb":
        args.feature_num = 50
        classifier = ResNet18(num_classes=args.class_num)
        cls_norm = [(0.3337, 0.3064, 0.3171), (0.2672, 0.2564, 0.2629)]
        cls_key = "model"
    elif args.dataset_name == "MNIST":
        args.feature_num = 12
        classifier = Mnist2LayerNet()
        cls_norm = [(0.13, 0.13, 0.13), (0.31, 0.31, 0.31)]
        cls_key = "model"
    else:
        raise NotImplementedError()

    config_name = os.path.basename(args.config_path).split(".")[0]
    save_name = os.path.join(args.log_dir, "{}_roc.pdf".format(
        args.arch + config_name))
    date = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    logger = utils.make_logger(
        None, os.path.join(
            args.log_dir, args.arch + config_name + "_" + date + ".log"))
    logger.info(args)
    # config model
    model_norm = [(0.5, 0.5, 0.5), (0.5, 0.5, 0.5)]
    net = eval(args.arch)
    model = net(cfgs, args, classifier, cls_norm, model_norm)
    assert isinstance(model, ContraNetDict)
    model.load_classifier(cfgs.cls_path, key=cls_key)
    model.only_judge = True
    model.cuda()
    model.eval()        # This is important.

    # test data
    test_data = LoadDataset(
        args.dataset_name, args.data_path, train=False, download=False,
        resize_size=args.img_size, hdf5_path=None, random_flip=False,
        norm=False)
    test_loader = DataLoader(
        test_data, batch_size=args.batch_size, shuffle=False, num_workers=4,
        pin_memory=True)
    torch.random.manual_seed(1)

    # build attack
    bounds = (0, 1)
    preprocessing = dict(mean=cls_norm[0], std=cls_norm[1], axis=-3)
    fmodel = fb.PyTorchModel(classifier, bounds=bounds,
                             preprocessing=preprocessing)

    # attack
    print("Start test ...")

    torch.random.manual_seed(1)
    load_path = os.path.join(args.load_dir, "{}{}_results.pth".format(
        args.arch, config_name))
    if os.path.exists(load_path):
        load_file = torch.load(load_path)
        score_dict = load_file["score_dict"]
        score_dict_clean = load_file["score_dict_clean"]
        y_dict = load_file["y_dict"]
        logger.info("Using previous results from: {}".format(load_path))
    else:
        logger.info("This is run for all clean sample.")
        with torch.no_grad():
            score_dict_clean = collect_clean(test_loader, fmodel, model)
        score_dict = {}
        y_dict = {}
        for t in args.test:
            if t == "BIM":
                name = "BIMinf"
            elif t == "BIML2":
                name = "BIML2"
            elif t == "PGD":
                name = "PGDinf"
            elif t == "PGDL2":
                name = "PGDL2"
            elif t == "CW":
                name = "CW"
            elif t == "CWinf":
                name = "CWinf"
            elif t == "DF":
                name = "DFinf"
            elif t == "DFL2":
                name = "DFL2"
            elif t == "EADA":
                name = "EADA"
                prefix = "Eada"
            elif t == "EADAL1":
                name = "EADAL1"
                prefix = "EaL1"
            else:
                raise NotImplementedError()
            # reset seed after get threshold, simulate the behavior before.
            torch.random.manual_seed(1)
            load_name = glob.glob(os.path.join(
                args.save_dir, t, "{}_{}_*".format(args.dataset_name, name)))
            load_sample = torch.load(load_name[0])
            logger.info("Using generated sample from {}".format(load_name[0]))
            if t in ["PGD", "BIM"]:
                load_sample["x_adv"] = {
                    key: load_sample["x_adv"][key] for key in [0.03]
                    # key: load_sample["x_adv"][key] for key in [0.03, 0.1, 0.2]
                }
            if t in ["PGDL2", "BIML2"]:
                load_sample["x_adv"] = {
                    key: load_sample["x_adv"][key] for key in [1.0]
                    # key: load_sample["x_adv"][key] for key in [1.0, 4.0, 8.0]
                }
            if t == "CW":
                load_sample["x_adv"] = {
                    key: load_sample["x_adv"][key] for key in [0]}
            if t == "CWinf":
                load_sample["x_adv"] = {
                    key: load_sample["x_adv"][key] for key in [0]}
            with torch.no_grad():
                results = collect_sample(load_sample, model, fmodel, name)
            score_dict_temp, y_dict_temp = update_with_clean(
                results, score_dict_clean)
            score_dict.update(score_dict_temp)
            y_dict.update(y_dict_temp)
        torch.save({
            "score_dict": score_dict,
            "score_dict_clean": score_dict_clean,
            "y_dict": y_dict
        }, os.path.join(args.log_dir, "{}{}_results.pth".format(
            args.arch, config_name)))
    # check possible thresholds
    def sort_list(x, d_name): return model.detector[d_name].sort_results(x)[0]
    thresholds = {}
    # we only use these for each detector
    # 12 point
    # cut_point = [0, 0.01, 0.03, 0.05, 0.08, 0.1, 0.15, 0.2, 0.35, 0.5, 0.7, 1]
    # 20 points
    # cut_point = [0, 0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.25, 0.3,
    # 0.5, 0.7, 0.85, 0.9, 0.93, 0.96, 0.98, 0.99, 1]
    # 100 points
    cut_point = np.arange(0, 1, 0.01).tolist()
    logging.info("cut_point: {}".format(cut_point))
    for key in score_dict_clean:
        out_sorted = sort_list(score_dict_clean[key], key)
        # change position ratio to actual value
        inds = (torch.tensor(cut_point) * len(out_sorted)).long()
        inds[-1] -= 1
        thresholds[key] = out_sorted[inds]
    # call plot
    all_auc, final_thresholds = plot_roc(
        score_dict, y_dict, thresholds, model, save_name)
    if args.csv_file is not None:
        final_results = args.arch + "\t" + "\t".join(
            ["{:.4f}".format(i) for i in all_auc])
        final_results += "\tthresh\t" + "\t".join(
            ["{:.4f}".format(i) for i in final_thresholds])
        with open(args.csv_file, 'a') as f:
            f.write(final_results + "\n")
