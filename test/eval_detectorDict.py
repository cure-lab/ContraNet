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
from models.ContraNet2_3 import *
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
        default=["clean", "PGD", "PGDL2", "BIM", "BIML2", "DF", "DFL2", "CW",
                 "CWinf, EADA, EADAL1"],
        nargs='*', type=str)
    parser.add_argument('-c', '--config_path', type=str,
                        default='config/configsCifar10GDnoise112000B.json')
    parser.add_argument(
        '--feature_m', action='store_true', help='use the minus of inputs')
    parser.add_argument(
        '--pretrain',
        default='results/cifar10AEPgd2/MobileNetV2-dense.01E.2_112000cifar10CondPgd-2021-05-24-12-35-50/MobileNetV2_91V97.48.pth',
        type=str, metavar='PATH',
        help='path to load pretrain checkpoint (default: none)')
    parser.add_argument('--batch_size', default=16, type=int,
                        metavar='N', help='mini-batch size')
    parser.add_argument('--save_dir', type=str,
                        default='results/log/whitebox')
    parser.add_argument('--csv_file', type=str, default=None)
    parser.add_argument('--log_dir', type=str, default=None)
    parser.add_argument('--hard_drop', default=None, type=float, nargs='*',
                        help='hard drop rate')
    parser.add_argument('--drop_rate', default=[0.05, 0.05], type=float,
                        nargs='*', help='drop rate')
    parser.add_argument('--thresh', default=[], type=float, nargs='*',
                        help='preset threshold')
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
    if args.hard_drop is not None:
        config_name += "_".join(["{:.3f}".format(i)
                                 for i in args.drop_rate + args.hard_drop])
    if args.log_dir is None:
        args.log_dir = args.save_dir
    date = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    logger = utils.make_logger(None, os.path.join(
        args.log_dir, args.arch + config_name + "_" +
        "-".join(["{:.4f}".format(th) for th in args.thresh]) + "_" 
        + date + ".log"))
    logger.info(args)
    # config model
    model_norm = [(0.5, 0.5, 0.5), (0.5, 0.5, 0.5)]
    net = eval(args.arch)
    model = net(cfgs, args, classifier, cls_norm, model_norm,
                hard_drop=args.hard_drop)
    assert isinstance(model, ContraNetDict)
    model.load_classifier(cfgs.cls_path, key=cls_key)
    model.only_rej = True
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

    if args.thresh != []:
        model.thresh = args.thresh
    else:
        model.thresh = model.get_thresh(test_loader, drop_rate=args.drop_rate)

    # attack
    print("Start test ...")

    rob_acc = ""
    detect_rate = ""
    for t in args.test:
        if t == "clean":
            # reset seed after get threshold, simulate the behavior before.
            torch.random.manual_seed(1)
            logger.info("This is run for all clean sample.")
            acc, detect = utils.test_clean(test_loader, model)
        else:
            # reset seed after get threshold, simulate the behavior before.
            torch.random.manual_seed(1)
            if t == "BIM":
                name = "BIMinf"
                prefix = "bim_"
            elif t == "BIML2":
                name = "BIML2"
                prefix = "bim2"
            elif t == "PGD":
                name = "PGDinf"
                prefix = "pgd_"
            elif t == "PGDL2":
                name = "PGDL2"
                prefix = "pgd2"
            elif t == "CW":
                name = "CW"
                prefix = "_cw_"
            elif t == "CWinf":
                name = "CWinf"
                prefix = "cwif"
            elif t == "DF":
                name = "DFinf"
                prefix = "_df_"
            elif t == "DFL2":
                name = "DFL2"
                prefix = "dfL2"
            elif t == "EADA":
                name = "EADA"
                prefix = "Eada"
            elif t == "EADAL1":
                name = "EADAL1"
                prefix = "EaL1"
            else:
                raise NotImplementedError()
            load_name = glob.glob(os.path.join(
                args.save_dir, t, "{}_{}_*".format(args.dataset_name, name)))
            load_sample = torch.load(load_name[0])
            logger.info("Using generated sample from {}".format(load_name[0]))
            if t == "PGD":
                load_sample["x_adv"] = {
                    key: load_sample["x_adv"][key] for key in [0.03, 0.1, 0.2]}
            if t == "PGDL2":
                load_sample["x_adv"] = {
                    key: load_sample["x_adv"][key] for key in [1.0, 4.0, 8.0]}
            if t == "CW":
                load_sample["x_adv"] = {
                    key: load_sample["x_adv"][key] for key in [0]}
            acc, detect = utils.attack_sample(
                load_sample, model, fmodel, prefix, args.batch_size)
        rob_acc += acc + "\t"
        detect_rate += detect + "\t"
    logging.info("RobA:{}".format(rob_acc))
    logging.info("Rate:{}".format(detect_rate))
    if args.csv_file is not None:
        if args.hard_drop is not None:
            param = "\t".join([str(i) for i in args.drop_rate + args.hard_drop])
        else:
            param = "\t".join([str(i) for i in args.thresh])
        param = "\t" + param + "\t"
        robA_results = args.arch + param + rob_acc[:-1]
        rate_results = args.arch + param + detect_rate[:-1]
        with open(args.csv_file, 'a') as f:
            f.write(robA_results + "\n")
            f.write(rate_results + "\n")
