import sys
sys.path.append(".")
sys.path.append("lib")

from models.densenet import densenet169
import misc.utils as utils
from misc.load_dataset import LoadDataset
import models.MobileNetV2 as MobileNet
from datetime import datetime
from torch.utils.data import DataLoader
import argparse
from models.resnet import *
from models.mnist2layer import *
import json
import os
import torch
import torch.nn as nn
import models.GANv2 as GANv2
from models.pgdWrapper import Pgd
from models.fgsmWrapper import Fgsm
from models.preGenWrapper import PreGen
from torchvision.transforms import Normalize
from lib.robustbench.utils import load_model as load_AA_model

from torch.utils.tensorboard import SummaryWriter
import tensorflow as tf
import tensorboard as tb
tf.io.gfile = tb.compat.tensorflow_stub.io.gfile


def train(orimodel, genmodel, mlpModel, loss_func, device, loader, optim, epoch,
          encoder, vae, gen, inference_m, args, writer, policy="",
          cond_optim=None, attack=None, norm=None):
    mlpModel.train()
    orimodel.query_features()
    genmodel.query_features()
    if args.cond:
        orimodel.train()
        genmodel.train()
    else:
        orimodel.eval()
        genmodel.eval()
    total_loss, total_num = 0.0, 0
    for img, classId in loader:
        if norm is not None:
            img_ori = norm(img)
        else:
            img_ori = img
        img_ori, img_pos, img_neg, y, wrong_y = utils.generate1(
            img_ori, classId, device, encoder, vae, gen, policy=policy,
            class_num=args.class_num)
        aux_img = None
        if attack is not None:
            x_adv_suss, pred_y_suss, y_suss, x_adv_fail, y_fail = \
                attack.adv_by_suss(img, classId, device, training=True)
            if x_adv_suss.shape[0] > 0:
                x_adv_suss = norm(x_adv_suss)
                x_adv_ori, x_adv_pos, x_adv_neg, y_adv, wrong_y_adv = \
                    utils.gen_with_pn(
                        x_adv_suss, y_suss, pred_y_suss, device, encoder, vae, gen)
                img_ori = torch.cat([img_ori, x_adv_ori], dim=0)
                img_pos = torch.cat([img_pos, x_adv_pos], dim=0)
                img_neg = torch.cat([img_neg, x_adv_neg], dim=0)
                y = torch.cat([y, y_adv], dim=0)
                wrong_y = torch.cat([wrong_y, wrong_y_adv], dim=0)
            # This is for failed attack samples, i.e. y == pred_y
            if x_adv_fail.shape[0] > 0:
                x_adv_fail = norm(x_adv_fail)
                aux_img, aux_pos, aux_y = utils.gen_with_label(
                    x_adv_fail, y_fail, device, encoder, vae, gen)

        if args.cond:
            for opti in cond_optim:
                opti.zero_grad()
        optim.zero_grad()

        feat_emb, _ = orimodel(img_ori)
        feat_pos, _ = genmodel(img_pos)
        feat_neg, _ = genmodel(img_neg)
        if aux_img is not None:
            feat_aux_emb, _ = orimodel(aux_img)
            feat_aux_pos, _ = genmodel(aux_pos)
            aux_data = {
                "feat_aux_emb": feat_aux_emb,
                "feat_aux_pos": feat_aux_pos,
                "aux_y": aux_y,
            }
        else:
            aux_data = None
        gt_pair, pred = inference_m(
            mlpModel, feat_emb, feat_pos, feat_neg, y, wrong_y,
            aux_data=aux_data)
        loss = loss_func(pred, gt_pair.cuda())
        loss.backward()
        optim.step()
        if args.cond:
            for opti in cond_optim:
                opti.step()

        total_num += loader.batch_size
        total_loss += loss.item() * loader.batch_size
        train_lr = optim.param_groups[0]['lr']
        global train_iter
        if train_iter % 50 == 0 or args.tag.find("debug") != -1:
            print("E:[{}/{}], lr:{:.6f}, L:{:.4f}".format(
                epoch, args.epochs, train_lr, total_loss / total_num))
            if args.cond:
                print("DML lr: {}".format(
                    [opti.param_groups[0]['lr'] for opti in cond_optim]))
        writer.add_scalar('Train/lr', train_lr, train_iter)
        writer.add_scalar('Train/loss', loss.item(), train_iter)
        train_iter += 1
        if args.tag.find("debug") != -1:
            break
    if attack is not None:
        attack.print_stat()
    writer.add_scalar('Train/EAvgLoss', total_loss / total_num, epoch)
    return total_loss / total_num


def test(orimodel, genmodel, mlpModel, device, dataset, encoder, vae, gen,
         inference_m, writer, norm=None):
    global test_iter
    mlpModel.eval()
    orimodel.eval()
    genmodel.eval()
    orimodel.query_features()
    genmodel.query_features()
    pred_list, pos_pred_list, neg_pred_list = [], [], []
    with torch.no_grad():
        for img, classId in dataset:
            if norm is not None:
                img = norm(img)
            img, img_pos, img_neg, y, wrong_y = utils.generate1(
                img, classId, device, encoder, vae, gen, next=True,
                class_num=args.class_num)
            feat_emb, _ = orimodel(img)
            feat_pos, _ = genmodel(img_pos)
            feat_neg, _ = genmodel(img_neg)
            gt_pair, pred = inference_m(
                mlpModel, feat_emb, feat_pos, feat_neg, y, wrong_y, test=True)
            pred_y = torch.argmax(pred, dim=1).cpu()
            pred_list.append((pred_y == gt_pair).cpu())
            pos_pred_list.append(pred_y[torch.where(gt_pair == 1)] == 1)
            neg_pred_list.append(pred_y[torch.where(gt_pair == 0)] == 0)
    pred_list = torch.cat(pred_list)
    pos_pred_list = torch.cat(pos_pred_list)
    neg_pred_list = torch.cat(neg_pred_list)
    acc = torch.sum(pred_list) / len(pred_list)
    print("MLP acc: {}".format(acc))
    pos_acc = torch.sum(pos_pred_list) / len(pos_pred_list)
    neg_acc = torch.sum(neg_pred_list) / len(neg_pred_list)
    writer.add_scalar('TestEmb/mlpAcc', acc.item(), test_iter)
    writer.add_scalar('TestEmb/mlpAcc_pos', pos_acc.item(), test_iter)
    writer.add_scalar('TestEmb/mlpAcc_neg', neg_acc.item(), test_iter)
    test_iter += 1
    return acc


def cache_batch_AE(idx, dataset, attack, norm):
    global suss_AE_batch, fail_AE_batch
    if suss_AE_batch is None:
        print(">>>> test AE no cache, generating for the first time.")
        suss_AE_batch, fail_AE_batch = [], []
        for img, classId in dataset:
            x_adv_suss, pred_y_suss, x_adv_fail, pred_y_fail = \
                attack.adv_by_suss(img, classId, device)
            x_adv_suss = norm(x_adv_suss)
            x_adv_fail = norm(x_adv_fail)
            suss_AE_batch.append((x_adv_suss.cpu(), pred_y_suss.cpu()))
            fail_AE_batch.append((x_adv_fail.cpu(), pred_y_fail.cpu()))
    return suss_AE_batch[idx], fail_AE_batch[idx]


def testAE(orimodel, genmodel, mlpModel, device, dataset, encoder, vae, gen,
           writer, attack, norm, feature_m=False):
    global test_iter, suss_AE_batch, fail_AE_batch
    mlpModel.eval()
    orimodel.eval()
    genmodel.eval()
    orimodel.query_features()
    genmodel.query_features()
    if feature_m:
        inference_m = utils.inference_2_mlp_m
    else:
        inference_m = utils.inference_2_mlp
    pred_list, pos_pred_list, neg_pred_list = [], [], []
    with torch.no_grad():
        for idx, (img, classId) in enumerate(dataset):
            (x_adv_suss, pred_y_suss), (x_adv_fail, pred_y_fail) = \
                cache_batch_AE(idx, dataset, attack, norm)
            assert img.shape[0] == x_adv_suss.shape[0] + x_adv_fail.shape[0]
            if len(x_adv_suss) > 0:
                x_adv_neg, x_adv_neg_gen, y_neg = utils.gen_with_label(
                    x_adv_suss, pred_y_suss, device, encoder, vae, gen)
                feat_emb_neg, _ = orimodel(x_adv_neg)
                feat_neg, _ = genmodel(x_adv_neg_gen)
                neg_gt, neg_pred = inference_m(
                    mlpModel, feat_emb_neg, feat_neg, y_neg, pos=False,
                    test=True)
            else:
                neg_gt = torch.Tensor()
                neg_pred = torch.Tensor().to(device)
            if len(x_adv_fail) > 0:
                x_adv_pos, x_adv_pos_gen, y_pos = utils.gen_with_label(
                    x_adv_fail, pred_y_fail, device, encoder, vae, gen)
                feat_emb_pos, _ = orimodel(x_adv_pos)
                feat_pos, _ = genmodel(x_adv_pos_gen)
                pos_gt, pos_pred = inference_m(
                    mlpModel, feat_emb_pos, feat_pos, y_pos, pos=True,
                    test=True)
            else:
                pos_gt = torch.Tensor()
                pos_pred = torch.Tensor().to(device)
            gt_pair = torch.cat([pos_gt, neg_gt], dim=0)
            pred = torch.cat([pos_pred, neg_pred], dim=0)
            pred_y = torch.argmax(pred, dim=1).cpu()
            pred_list.append((pred_y == gt_pair).cpu())
            pos_pred_list.append(pred_y[torch.where(gt_pair == 1)] == 1)
            neg_pred_list.append(pred_y[torch.where(gt_pair == 0)] == 0)
    pred_list = torch.cat(pred_list)
    pos_pred_list = torch.cat(pos_pred_list)
    neg_pred_list = torch.cat(neg_pred_list)
    acc = torch.sum(pred_list) / len(pred_list)
    print("AEMLP acc: {}".format(acc))
    pos_acc = torch.sum(pos_pred_list) / len(pos_pred_list)
    neg_acc = torch.sum(neg_pred_list) / len(neg_pred_list)
    writer.add_scalar('TestEmb/AEmlpAcc', acc.item(), test_iter)
    writer.add_scalar('TestEmb/AEmlpAcc_pos', pos_acc.item(), test_iter)
    writer.add_scalar('TestEmb/AEmlpAcc_neg', neg_acc.item(), test_iter)
    return acc


test_iter = 0
train_iter = 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train MoCo on GTSRB')
    parser.add_argument('-a', '--arch', default='MobileNetV2', type=str)
    parser.add_argument('-c', '--config_path', type=str,
                        default='./config/configsCifar10.json')
    parser.add_argument('--tag', default='', type=str)
    # lr: 0.06 for batch 512 (or 0.03 for batch 256)
    parser.add_argument(
        '--lr', '--learning-rate', default=0.06, type=float, metavar='LR',
        help='initial learning rate', dest='lr')
    parser.add_argument(
        '--drop_p', default=0.1, type=float, metavar='LR',
        help='MLP drop rate', dest='drop_p')
    parser.add_argument(
        '--epochs', default=100, type=int, metavar='N',
        help='number of total epochs to run')
    parser.add_argument(
        '--feature_m', action='store_true', help='use the minus of inputs')
    parser.add_argument(
        '--useAE', type=str, default=None, help='add the AE to inputs')
    parser.add_argument(
        '--ae_epsilon', type=float, default=0.08, help='epsilon for AEs')
    parser.add_argument(
        '--AEPreGen', type=str, default=None, help='file to load pre_gen AEs')
    parser.add_argument(
        '--cond', action='store_true',
        help='After MLP pretrain, train end-to-end')
    parser.add_argument(
        '--device_id', default=[], nargs='*', type=int, help='cuda device ids')
    parser.add_argument('--cos', default=-1, type=int,
                        help='use cosine lr schedule')

    parser.add_argument('--batch_size', default=64, type=int,
                        metavar='N', help='mini-batch size')
    parser.add_argument('--wd', default=5e-4, type=float,
                        metavar='W', help='weight decay')
    # utils
    parser.add_argument(
        '--pretrain', default='', type=str, metavar='PATH',
        help='path to pretrain checkpoint (default: none)')
    parser.add_argument(
        '--resume', default='', type=str, metavar='PATH',
        help='path to latest checkpoint (default: none)')
    parser.add_argument(
        '--results_dir', default='./results', type=str, metavar='PATH',
        help='path to cache (default: none)')

    args = parser.parse_args()
    if args.cos == -1:
        args.cos = args.epochs
    if args.config_path is not None:
        with open(args.config_path) as f:
            model_configs = json.load(f)
        train_configs = vars(args)
    else:
        raise NotImplementedError
    cfgs = utils.dict2clsattr(train_configs, model_configs)
    args.dataset_name = cfgs.dataset_name
    args.data_path = cfgs.data_path
    args.img_size = cfgs.img_size
    args.class_num = cfgs.num_classes
    if args.dataset_name == "cifar10":
        args.feature_num = 12
    elif args.dataset_name == "MNIST":
        args.feature_num = 12
    elif args.dataset_name == "cifar100":
        args.feature_num = 120
    elif args.dataset_name == "tiny_imagenet":
        args.feature_num = 240
    elif args.dataset_name == "gtsrb":
        args.feature_num = 50

    # policy = "color,translation,cutout"
    policy = ""
    print("Using data augmentation policy: {}".format(policy))
    assert args.arch in ["MobileNetV2"]
    args.tag += cfgs.dataset_name
    if args.cond:
        args.tag += "Cond"
        assert args.pretrain != ""
    if args.useAE:
        assert args.useAE in ["Fgsm", "Pgd"], "only support Fgsm or Pgd attack"
        args.tag += args.useAE
        if args.AEPreGen is not None:
            args.tag += "_G"
    if args.tag != '':
        args.results_dir = os.path.join(
            args.results_dir, '{}-{}-'.format(args.arch, args.tag) +
            datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
    else:
        args.results_dir = os.path.join(
            args.results_dir, '{}-'.format(args.arch) +
            datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        )

    print(args)
    device = torch.device("cuda")
    print(' prepared models...')
    # initialize models.
    gen = GANv2.Generator(
        cfgs.z_dim, cfgs.shared_dim, cfgs.img_size, cfgs.g_conv_dim, cfgs.
        g_spectral_norm, cfgs.attention, cfgs.attention_after_nth_gen_block,
        cfgs.activation_fn, cfgs.conditional_strategy, cfgs.num_classes, cfgs.
        g_init, cfgs.G_depth, False)
    encoder = GANv2.Encoder(isize=cfgs.img_size, nz=cfgs.z_dim, nc=3, ndf=64)
    vae = GANv2.VAE(isize=cfgs.img_size, nz=cfgs.z_dim)

    gen.load_state_dict(torch.load(cfgs.G_weights)['state_dict'])
    print("gen loaded from: {}".format(cfgs.G_weights))
    encoder.load_state_dict(torch.load(cfgs.E_weights)['state_dict'])
    print("encoder loaded from: {}".format(cfgs.E_weights))
    vae.load_state_dict(torch.load(cfgs.V_weights)['state_dict'])
    print("vae loaded from: {}".format(cfgs.V_weights))

    gen.eval().cuda()
    encoder.eval().cuda()
    vae.eval().cuda()

    print(' preparing dataset...')
    norm = False if args.useAE else True
    train_data = LoadDataset(
        args.dataset_name, args.data_path, train=True, download=False,
        resize_size=args.img_size, hdf5_path=None, random_flip=True, norm=norm)
    test_data = LoadDataset(
        args.dataset_name, args.data_path, train=False, download=False,
        resize_size=args.img_size, hdf5_path=None, random_flip=False, norm=norm)
    if args.dataset_name == "tiny_imagenet":
        for key, value in train_data.data.class_to_idx.items():
            assert value == test_data.data.class_to_idx[key]

    train_loader = DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, num_workers=4,
        pin_memory=True, drop_last=True)
    test_loader = DataLoader(
        test_data, batch_size=args.batch_size, shuffle=False, num_workers=4,
        pin_memory=True)

    print(' preparing utils...')
    if args.feature_m:
        inference_m = utils.inference_mlp_m
    else:
        inference_m = utils.inference_mlp

    if args.useAE:
        print(' preparing attack...')
        weight_key = None
        if args.dataset_name == "cifar10":
            if hasattr(cfgs, "cls_name"):
                if cfgs.cls_name == "densenet":
                    classifier = densenet169()
                    weight_key = ""
                    cls_norm = [(0.4914, 0.4822, 0.4465),
                                (0.2023, 0.1994, 0.2010)]
                else:
                    classifier = load_AA_model(
                        model_name=cfgs.cls_name, dataset=args.dataset_name,
                        threat_model='Linf')
                    cls_norm = [(0., 0., 0.), (1., 1., 1.)]
                print("Classifier using: {}".format(cfgs.cls_name))
            else:
                classifier = ResNet50(num_classes=args.class_num)
                weight_key = "net"
                cls_norm = [(0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)]
        elif args.dataset_name == "gtsrb":
            classifier = ResNet18(num_classes=args.class_num)
            classifier.load_state_dict(torch.load(cfgs.cls_path)['model'])
            cls_norm = [(0.3337, 0.3064, 0.3171), (0.2672, 0.2564, 0.2629)]
        elif args.dataset_name == "MNIST":
            classifier = Mnist2LayerNet()
            classifier.load_state_dict(torch.load(cfgs.cls_path)['model'])
            cls_norm = [(0.13, 0.13, 0.13), (0.31, 0.31, 0.31)]
        else:
            raise NotImplementedError()
        if weight_key is "":
            classifier.load_state_dict(torch.load(cfgs.cls_path))
        elif weight_key is not None:
            classifier.load_state_dict(torch.load(cfgs.cls_path)[weight_key])
        print("Classifier loaded from: {}".format(cfgs.cls_path))
        classifier.cuda()
        cri = torch.nn.CrossEntropyLoss()

        if args.AEPreGen is None:
            attacker = eval("{}(classifier, cri, cls_norm, max_eps={})".format(
                args.useAE, args.ae_epsilon))
            print("Attacker loaded for: {} with eps={}".format(
                args.useAE, args.ae_epsilon))
        else:
            all_AE_paths = args.AEPreGen.strip(";").split(";")
            attacker = PreGen(classifier, cri, cls_norm,
                              all_AE_paths, args.batch_size)
            print("Attacker PreGen loaded from: {}".format(all_AE_paths))
        data_norm = Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    else:
        attacker = None
        data_norm = None

    print(' preparing model...')
    mlploss = nn.CrossEntropyLoss()
    device_ids = args.device_id
    orimodel = eval("MobileNet.{}(n_class={})".format(
        args.arch, args.feature_num))
    orimodel = orimodel.cuda()
    genmodel = eval("MobileNet.{}(n_class={})".format(
        args.arch, args.feature_num))
    genmodel = genmodel.cuda()

    if args.feature_m:
        mlpModel = MobileNet.MLP(
            orimodel.last_channel, p=args.drop_p, class_num=args.class_num)
    else:
        mlpModel = MobileNet.MLP(
            orimodel.last_channel * 2, p=args.drop_p, class_num=args.class_num)
    mlpModel = mlpModel.cuda()
    # define optimizer
    optimizer = torch.optim.SGD(
        mlpModel.parameters(), lr=args.lr, weight_decay=args.wd, momentum=0.9)

    # set aux opt and scheuler for cond
    if args.cond:
        ori_Optimizer = torch.optim.SGD(
            orimodel.parameters(),
            lr=args.lr, weight_decay=args.wd, momentum=0.9)

        gen_Optimizer = torch.optim.SGD(
            genmodel.parameters(),
            lr=args.lr, weight_decay=args.wd, momentum=0.9)
        cond_optim = [ori_Optimizer, gen_Optimizer]
    else:
        cond_optim = None

    # load model if resume or pretrain
    epoch_start = 1
    optimizer_loaded = False
    if args.resume is not '':
        load_path = args.resume
    elif args.pretrain is not '':
        load_path = args.pretrain
    else:
        load_path = None
    if load_path:
        checkpoint = torch.load(load_path)
        mlpModel.load_state_dict(checkpoint['mlp_state'])
        print('Loaded from: {}'.format(load_path))
        if args.resume:
            optimizer.load_state_dict(checkpoint['optimizer'])
            ori_Optimizer.load_state_dict(checkpoint["ori_optim"])
            gen_Optimizer.load_state_dict(checkpoint["gen_optim"])
            epoch_start = checkpoint['epoch'] + 1
            print("resuming optimizer from {}, will start from E{}".format(
                load_path, epoch_start))
        orimodel.load_state_dict(checkpoint['ori_state'])
        print("Loaded orimodel from: {}".format(load_path))
        genmodel.load_state_dict(checkpoint['gen_state'])
        print("Loaded genmodel from: {}".format(load_path))
    else:
        checkpoint = torch.load(cfgs.ori_path)
        orimodel.load_state_dict(checkpoint['state_dict'])
        print("Loaded orimodel from: {}".format(cfgs.ori_path))

        checkpoint = torch.load(cfgs.gen_path)
        genmodel.load_state_dict(checkpoint['state_dict'])
        print("Loaded genmodel from: {}".format(cfgs.gen_path))

    orimodel.eval()
    genmodel.eval()
    if args.resume:
        if args.cond:
            ori_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                ori_Optimizer, args.cos, eta_min=1e-6,
                last_epoch=epoch_start - 1
            )
            gen_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                gen_Optimizer, args.cos, eta_min=1e-6,
                last_epoch=epoch_start - 1
            )
        else:
            ori_lr_scheduler = None
            gen_lr_scheduler = None
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, args.cos, eta_min=1e-6,
            last_epoch=epoch_start - 1
        )
    else:
        if args.cond:
            epoch_start = 1
            ori_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                ori_Optimizer, args.cos, eta_min=1e-6
            )
            gen_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                gen_Optimizer, args.cos, eta_min=1e-6
            )
        else:
            ori_lr_scheduler = None
            gen_lr_scheduler = None
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, args.cos, eta_min=1e-6
        )

    # gpu parallel
    if len(device_ids) > 0:
        gen = torch.nn.DataParallel(gen, device_ids)
        encoder = torch.nn.DataParallel(encoder, device_ids)
        vae = torch.nn.DataParallel(vae, device_ids)
        orimodel = torch.nn.DataParallel(orimodel, device_ids)
        genmodel = torch.nn.DataParallel(genmodel, device_ids)
        mlpModel = torch.nn.DataParallel(mlpModel, device_ids)
        orimodel.query_features = orimodel.module.query_features
        genmodel.query_features = genmodel.module.query_features

    # logging
    results = {'train_loss': [], 'test_acc@1': []}
    if not os.path.exists(args.results_dir):
        os.mkdir(args.results_dir)
    # dump args
    with open(args.results_dir + '/args.json', 'w') as fid:
        json.dump(args.__dict__, fid, indent=2)

    writer = SummaryWriter(os.path.join(args.results_dir, "tensorboard"))
    # training loop
    best_prec_at_1 = 0
    best_AE_at_1 = 0
    train_iter = epoch_start * len(train_data) // args.batch_size
    test_iter = epoch_start
    suss_AE_batch, fail_AE_batch = None, None
    if load_path:
        if args.useAE:
            acc_AE = testAE(
                orimodel, genmodel, mlpModel, device, test_loader, encoder, vae,
                gen, writer, attacker, data_norm, feature_m=args.feature_m)
            best_AE_at_1 = acc_AE * 100
        acc = test(
            orimodel, genmodel, mlpModel, device, test_loader, encoder, vae,
            gen, inference_m, writer, data_norm)
        best_prec_at_1 = acc * 100

    print("Start Training ...")
    for epoch in range(epoch_start, args.epochs + 1):
        train(orimodel, genmodel, mlpModel, mlploss, device, train_loader,
              optimizer, epoch, encoder, vae, gen, inference_m, args, writer,
              policy, cond_optim=cond_optim, attack=attacker, norm=data_norm)
        lr_scheduler.step()
        if args.cond:
            ori_lr_scheduler.step()
            gen_lr_scheduler.step()
        params = {'epoch': epoch, 'optimizer': optimizer.state_dict(), }
        if device_ids == []:
            params['ori_state'] = orimodel.state_dict()
            params['gen_state'] = genmodel.state_dict()
            params['mlp_state'] = mlpModel.state_dict()
        else:
            params['ori_state'] = orimodel.module.state_dict()
            params['gen_state'] = genmodel.module.state_dict()
            params['mlp_state'] = mlpModel.module.state_dict()
        if args.cond:
            params['ori_optim'] = ori_Optimizer.state_dict()
            params['gen_optim'] = gen_Optimizer.state_dict()

        torch.save(params, args.results_dir + '/{}.pth'.format(args.arch))
        acc = test(orimodel, genmodel, mlpModel, device, test_loader, encoder,
                   vae, gen, inference_m, writer, data_norm)
        best_prec_at_1 = utils.save_best(best_prec_at_1, "best", acc * 100,
                                         params, epoch, args.arch,
                                         args.results_dir)
        if args.useAE:
            acc_AE = testAE(orimodel, genmodel, mlpModel, device, test_loader,
                            encoder, vae, gen, writer, attacker, data_norm,
                            feature_m=args.feature_m)
            best_AE_at_1 = utils.save_best(best_AE_at_1, "AEbest", acc_AE * 100,
                                           params, epoch, args.arch,
                                           args.results_dir)
        del params
