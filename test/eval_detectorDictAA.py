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
import misc.utils as utils
from lib.robustbench.utils import load_model
from misc.load_dataset import LoadDataset
from torch.utils.data import DataLoader
np.set_printoptions(precision=4, suppress=True, linewidth=120)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', default="ContraNet2_3ssH")
    parser.add_argument(
        '-m', '--model', default='Gowal2020Uncovering_70_16', type=str)
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
    # autoattack params
    parser.add_argument('--norm', type=str, default='Linf')
    parser.add_argument('--epsilon', type=float, default=8. / 255.)
    parser.add_argument('--attack', type=str,
                        default=['apgd-ce', 'apgd-t', 'fab-t', 'square'],
                        nargs='*')
    parser.add_argument('--version', type=str, default='standard')
    # end for autoattack params
    parser.add_argument(
        '--save_dir', type=str,
        default='results/log/autoattack/cifar10AEPgd295/E.2AEV97.48Dnoise112000B')
    parser.add_argument(
        '--csv_file', type=str,
        default="results/log/autoattack/cifar10AEPgd295/cifar.csv")
    parser.add_argument('--log_dir', type=str, default=None)
    parser.add_argument('--hard_drop', default=None, type=float, nargs='*',
                        help='hard drop rate')
    parser.add_argument('--drop_rate', default=[0.0], type=float,
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
        classifier = load_model(
            model_name=args.model, dataset=args.dataset_name,
            threat_model='Linf')
        cls_norm = None
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
        args.model + "_" + date + ".log"))
    logger.info(args)
    # config model
    model_norm = [(0.5, 0.5, 0.5), (0.5, 0.5, 0.5)]
    net = eval(args.arch)
    model = net(cfgs, args, classifier, cls_norm, model_norm,
                hard_drop=args.hard_drop)
    assert isinstance(model, ContraNetDict)
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

    if args.thresh != []:
        model.thresh = args.thresh
    else:
        model.thresh = model.get_thresh(test_loader, drop_rate=args.drop_rate)

    # attack
    print("Start test ...")

    rob_acc = ""
    detect_rate = ""
    # reset seed after get threshold, simulate the behavior before.
    torch.random.manual_seed(1)
    logger.info("This is run for all clean sample.")
    with torch.no_grad():
        acc, detect = utils.test_clean(test_loader, model)
    rob_acc += acc + "\t"
    detect_rate += detect + "\t"

    logger.info("This is run for all autoattack sample.")
    load_reg = 'results/log/autoattack/{}/{}_{}_*_eps_{:.5f}.pth'.format(
        args.model, "_".join(args.attack),
        args.version, args.epsilon)
    load_name = glob.glob(load_reg)
    if len(load_name) > 0:
        logger.info("Using generated sample from {}".format(load_name[0]))
        x_adv = torch.load(load_name[0])["adv_complete"]
        logger.info("loaded AE from: {}".format(load_name[0]))
    else:
        raise FileNotFoundError(load_reg)
    # reset seed after get threshold, simulate the behavior before.
    torch.random.manual_seed(1)
    with torch.no_grad():
        acc, detect = utils.attack_sample_aa(
            test_loader, x_adv, model, model.classifier, args.batch_size)
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
        robA_results = args.arch + "\t" + args.model + param + rob_acc[:-1]
        rate_results = args.arch + "\t" + args.model + param + detect_rate[:-1]
        with open(args.csv_file, 'a') as f:
            f.write(robA_results + "\n")
            f.write(rate_results + "\n")
