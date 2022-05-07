import logging
import sys
sys.path.append(".")
sys.path.append("lib")

import json
import argparse
import os
import torch
import time
import foolbox as fb
import numpy as np
from models.ContraNet2 import ContraNet2
from models.ContraNet2D import ContraNet2D
from models.ContraNet2DDmlv2 import ContraNet2DDmlv2
from models.resnet import *
from models.densenet import *
from models.mnist2layer import *
import misc.utils as utils
from misc.load_dataset import LoadDataset
from torch.utils.data import DataLoader
import glob
np.set_printoptions(precision=4, suppress=True, linewidth=120)


def attack_sample(all_sample, model, fmodel, prefix):
    def sample_adv(eps, idx, img, classId):
        return all_sample["x_adv"][eps][idx].cuda()
    loader = zip(all_sample["x_ori"], all_sample["y_ori"])
    utils.attack_helper(loader, model, fmodel, all_sample["x_adv"].keys(),
                        sample_adv, prefix)


def attack_gen(test_loader, fmodel, model, save_dir, prefix, name,
               epsilon_list, params, attack):
    def gen_adv(eps, idx, img, classId):
        adversary = attack(overshoot=eps, **params)
        _, x_adv, _ = adversary(fmodel, img, classId, epsilons=None)
        return x_adv
    utils.attack_helper(test_loader, model, fmodel, epsilon_list, gen_adv,
                        prefix, save_dir, name, save=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--arch', default='ContraNet2DDmlv2', type=str)
    parser.add_argument('--attack', type=str, help='attack type')
    parser.add_argument('-c', '--config_path', type=str,
                        default='./config/configsCifar10Dadv.json')
    parser.add_argument(
        '--feature_m', action='store_true', help='use the minus of inputs')
    parser.add_argument(
        '--pretrain',
        default='results/cifar10AEPgd/MobileNetV2-.01cifar10CondPgd-2021-04-30-21-47-22/MobileNetV2_AEbestE88V96.92.pth',
        type=str, metavar='PATH',
        help='path to load pretrain checkpoint (default: none)')
    parser.add_argument('--batch_size', default=8, type=int,
                        metavar='N', help='mini-batch size')
    parser.add_argument('--save_dir', type=str, default='results/log/whitebox/')
    parser.add_argument('--log_dir', type=str, default=None)
    parser.add_argument('--thresh', default=[], nargs='*', type=float,
                        help='preset threshold')
    args = parser.parse_args()

    if args.attack == "DF":
        adversary = fb.attacks.LinfDeepFoolAttack
        name = "DFinf"
        prefix = "_df_"
    elif args.attack == "DFL2":
        adversary = fb.attacks.L2DeepFoolAttack
        name = "DFL2"
        prefix = "dfL2"
    else:
        raise NotImplementedError()

    if args.config_path is not None:
        with open(args.config_path) as f:
            model_configs = json.load(f)
        train_configs = vars(args)
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
    args.save_dir = os.path.join(args.save_dir, args.attack)
    if args.log_dir is None:
        args.log_dir = args.save_dir
    date = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    logger = utils.make_logger(
        None, os.path.join(
            args.log_dir, config_name + date +
            "discriminator_foolbox_{}_white_box.log".format(name)))
    logger.info(args)
    # config model
    model_norm = [(0.5, 0.5, 0.5), (0.5, 0.5, 0.5)]
    net = eval(args.arch)
    model = net(cfgs, args, classifier, cls_norm, model_norm)
    if isinstance(model, ContraNet2DDmlv2) and args.thresh != []:
        model.thresh = args.thresh
        logger.info("set thresh to {}".format(model.thresh))
    if isinstance(model, ContraNet2DDmlv2):
        model.only_rej = True
    model.load_classifier(cfgs.cls_path, cls_key)
    model.cuda()
    model.eval()        # This is important.
    model.debug = True

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
    overshoot = [0.02]
    params = {
        "steps": 100,
        "candidates": args.class_num,
    }

    # attack
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    print("Start attack ...")

    load_name = glob.glob(os.path.join(
        args.save_dir, "{}_{}_*".format(args.dataset_name, name)))
    if len(load_name) > 0:
        load_sample = torch.load(load_name[0])
        logging.info("Using generated sample from {}".format(load_name[0]))
        attack_sample(load_sample, model, fmodel, prefix)
    else:
        load_sample = None
        attack_gen(test_loader, fmodel, model, args.save_dir, prefix,
                   "{}_{}".format(args.dataset_name, name), overshoot,
                   params, adversary)
