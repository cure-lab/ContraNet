# gpu_info = !nvidia-smi -i 0
# gpu_info = '\n'.join(gpu_info)
# print(gpu_info)

import misc.utils as utils
from misc.load_dataset import LoadDataset
import models.MobileNetV2 as MobileNet
from datetime import datetime
from torch.utils.data import DataLoader
import argparse
import json
import os
import torch
import glob
import numpy as np
import models.GANv2 as GANv2
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
from pytorch_metric_learning.regularizers import LpRegularizer
from pytorch_metric_learning.reducers import MeanReducer, ThresholdReducer
from pytorch_metric_learning import testers
from pytorch_metric_learning.distances import LpDistance
from lib.pytorch_metric_learning import losses, miners

from torch.utils.tensorboard import SummaryWriter
import tensorflow as tf
import tensorboard as tb
tf.io.gfile = tb.compat.tensorflow_stub.io.gfile


def train(dmlmodel, mining_func, loss_func, device, loader, optim, epoch,
          encoder, vae, gen, args, writer):
    dmlmodel.train()
    total_loss, total_num = 0.0, 0
    for img, classId in loader:
        optim.zero_grad()
        if args.genFig:
            img, img_pos, img_neg, y, wrong_y = utils.generate1(
                img, classId, device, encoder, vae, gen,
                class_num=args.class_num)
            feat_pos = dmlmodel(img_pos)
            feat_neg = dmlmodel(img_neg)

            embeddings = torch.cat([feat_pos, feat_neg], dim=0)
            labels = torch.cat([y, wrong_y], dim=0)
        else:
            img = img.to(device)
            feat_emb = dmlmodel(img)
            embeddings = feat_emb
            labels = classId.to(device)

        indices_tuple = mining_func(embeddings, labels)
        loss = loss_func(embeddings, labels, indices_tuple)

        loss.backward()
        optim.step()

        total_num += loader.batch_size
        total_loss += loss.item() * loader.batch_size
        train_lr = optim.param_groups[0]['lr']
        global train_iter
        if train_iter % 50 == 0 or args.tag.find("debug") != -1:
            print("E:[{}/{}], lr:{:.6f}, L:{:.4f} Ntrip:{}".format(
                epoch, args.epochs, train_lr, total_loss / total_num,
                mining_func.num_triplets))
        writer.add_scalar('Train/lr', train_lr, train_iter)
        writer.add_scalar('Train/loss', loss.item(), train_iter)
        writer.add_scalar('Train/Ntrip', mining_func.num_triplets, train_iter)
        train_iter += 1
        if args.tag.find("debug") != -1:
            break
    writer.add_scalar('Train/EAvgLoss', total_loss / total_num, epoch)
    return total_loss / total_num


def get_all_embeddings(dataset, model):
    tester = testers.BaseTester()
    return tester.get_all_embeddings(dataset, model)


def test(dmlmodel, accuracy_calculator, device, test_loader, encoder, vae, gen,
         args, writer):
    all_aug_pos, all_aug_neg = [], []
    emb_meta, neg_meta = [], []
    with torch.no_grad():
        for img, classId in test_loader:
            if args.genFig:
                _, img_pos, img_neg, y, wrong_y = utils.generate1(
                    img, classId, device, encoder, vae, gen, next=True,
                    class_num=args.class_num)
                augmented_pos = dmlmodel(img_pos)
                augmented_neg = dmlmodel(img_neg)
                all_aug_pos.append(augmented_pos.cpu())
                all_aug_neg.append(augmented_neg.cpu())
                neg_meta.append(wrong_y.cpu())
            else:
                img = img.to(device)
                feat_emb = dmlmodel(img)
                all_aug_pos.append(feat_emb.cpu())
                y = classId
            emb_meta.append(y.cpu())
    if args.genFig:
        neg_meta = torch.cat(neg_meta)
        all_aug_neg = torch.cat(all_aug_neg)
    all_aug_pos = torch.cat(all_aug_pos)
    emb_meta = torch.cat(emb_meta)

    print("Computing accuracy")
    num_sample = len(all_aug_pos) // 5
    if args.genFig:
        train_embeddings = torch.cat(
            [all_aug_pos[:num_sample*4], all_aug_neg[:num_sample*4]], dim=0
        ).cpu().numpy()
        train_labels = torch.cat(
            [emb_meta[:num_sample*4], neg_meta[:num_sample*4]], dim=0
        ).cpu().numpy()
        test_embeddings = torch.cat(
            [all_aug_pos[-num_sample:], all_aug_neg[-num_sample:]], dim=0
        ).cpu().numpy()
        test_labels = torch.cat(
            [emb_meta[-num_sample:], neg_meta[-num_sample:]], dim=0
        ).cpu().numpy()
    else:
        train_embeddings = all_aug_pos[:num_sample*4].cpu().numpy()
        train_labels = emb_meta[:num_sample*4].cpu().numpy()
        test_embeddings = all_aug_pos[-num_sample:].cpu().numpy()
        test_labels = emb_meta[-num_sample:].cpu().numpy()
    accuracies = accuracy_calculator.get_accuracy(
        test_embeddings,
        train_embeddings,
        np.squeeze(test_labels),
        np.squeeze(train_labels),
        False
    )
    print("Test set accuracy (Precision@1) = {}".format(
        accuracies["precision_at_1"]))
    global test_iter
    writer.add_scalar('Test/precision_at_1',
                      accuracies["precision_at_1"], test_iter)
    if test_iter % 5 == 0 or args.tag.find("debug") != -1:
        if args.genFig:
            meta_data = [['pos_gen', j.item()] for j in emb_meta] + \
                [['neg_gen', k.item()] for k in neg_meta]
            meta_header = ["type", "class"]
            all_emd = torch.cat([all_aug_pos, all_aug_neg], dim=0)
        else:
            meta_data = [j.item() for j in emb_meta]
            meta_header = None
            all_emd = all_aug_pos
        writer.add_embedding(
            all_emd, metadata=meta_data,
            global_step=test_iter,
            metadata_header=meta_header)
    test_iter += 1
    return accuracies["precision_at_1"]


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
        '--epochs', default=100, type=int, metavar='N',
        help='number of total epochs to run')
    parser.add_argument(
        '--genFig', action='store_true', help='use the minus of inputs')
    parser.add_argument(
        '--device_id', default=[], nargs='*', type=int, help='cuda device ids')
    parser.add_argument('--cos', default=-1, type=int,
                        help='use cosine lr schedule')
    parser.add_argument('--steps', default=[], nargs='*', type=int,
                        help='use cosine lr schedule')

    parser.add_argument('--batch_size', default=256, type=int,
                        metavar='N', help='mini-batch size')
    parser.add_argument('--wd', default=5e-4, type=float,
                        metavar='W', help='weight decay')
    # utils
    parser.add_argument(
        '--resume', default='', type=str, metavar='PATH',
        help='path to latest checkpoint (default: none)')
    parser.add_argument(
        '--results_dir', default='', type=str, metavar='PATH',
        help='path to cache (default: none)')

    args = parser.parse_args()
    if args.cos == -1:
        args.cos = args.epochs
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
    elif args.dataset_name == "MNIST":
        args.feature_num = 12
    elif args.dataset_name == "cifar100":
        args.feature_num = 120
    elif args.dataset_name == "tiny_imagenet":
        args.feature_num = 240
    elif args.dataset_name == "gtsrb":
        args.feature_num = 50

    assert args.arch in ["MobileNetV2"]
    args.tag += cfgs.dataset_name
    if args.genFig:
        args.tag += "Gen"
    if args.results_dir == '':
        if args.tag != '':
            args.results_dir = './pretrain/{}-{}-'.format(args.arch, args.tag) \
                + datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        else:
            args.results_dir = './pretrain/{}-'.format(args.arch) + \
                datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    print(args)
    print(' preparing models...')
    # initialize models.
    if args.genFig:
        gen = GANv2.Generator(
            cfgs.z_dim, cfgs.shared_dim, cfgs.img_size, cfgs.g_conv_dim,
            cfgs.g_spectral_norm, cfgs.attention,
            cfgs.attention_after_nth_gen_block, cfgs.activation_fn,
            cfgs.conditional_strategy, cfgs.num_classes, cfgs.g_init,
            cfgs.G_depth, False)
        encoder = GANv2.Encoder(isize=cfgs.img_size,
                                nz=cfgs.z_dim, nc=3, ndf=64)
        vae = GANv2.VAE(isize=cfgs.img_size, nz=cfgs.z_dim)

        gen.load_state_dict(torch.load(cfgs.G_weights)['state_dict'])
        encoder.load_state_dict(torch.load(cfgs.E_weights)['state_dict'])
        vae.load_state_dict(torch.load(cfgs.V_weights)['state_dict'])

        gen.eval().cuda()
        encoder.eval().cuda()
        vae.eval().cuda()
    else:
        gen = None
        encoder = None
        vae = None

    print(' preparing datasets...')
    train_data = LoadDataset(
        args.dataset_name, args.data_path, train=True, download=False,
        resize_size=args.img_size, hdf5_path=None, random_flip=True)
    test_data = LoadDataset(
        args.dataset_name, args.data_path, train=False, download=False,
        resize_size=args.img_size, hdf5_path=None, random_flip=False)
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
    device_ids = args.device_id
    dmlmodel = eval("MobileNet.{}(n_class={})".format(
        args.arch, args.feature_num))
    dmlmodel = dmlmodel.cuda()
    clsloss = losses.TripletMarginLoss(
        margin=0.7,
        swap=False,
        distance=LpDistance(p=2),
        reducer=ThresholdReducer(low=0),
        embedding_regularizer=LpRegularizer()
    )

    clsmining = miners.TripletMarginMiner(
        margin=1.0, distance=LpDistance(p=2), type_of_triplets="semihard")
    accuracy_calculator = AccuracyCalculator(include=("precision_at_1",), k=1)

    # define optimizer
    optimizer = torch.optim.SGD(
        dmlmodel.parameters(), lr=args.lr, weight_decay=args.wd, momentum=0.9)

    # load model if resume
    epoch_start = 1
    if args.resume is not '':
        checkpoint = torch.load(args.resume)
        dmlmodel.load_state_dict(checkpoint['state_dict'])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer'])
        epoch_start = checkpoint['epoch'] + 1
        print('Loaded from: {}'.format(args.resume))
        if args.steps != []:
            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, args.steps, gamma=0.1, last_epoch=epoch_start - 1
            )
        else:
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, args.cos, eta_min=1e-6
            )
            lr_scheduler.step(epoch_start - 1)
    else:
        if args.steps != []:
            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, args.steps, gamma=0.1,
            )
        else:
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, args.cos, eta_min=1e-6
            )

    if len(device_ids) > 0:
        if args.genFig:
            gen = torch.nn.DataParallel(gen, device_ids)
            encoder = torch.nn.DataParallel(encoder, device_ids)
            vae = torch.nn.DataParallel(vae, device_ids)
        dmlmodel = torch.nn.DataParallel(dmlmodel, device_ids)

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
    train_iter = epoch_start * len(train_data) // args.batch_size
    test_iter = epoch_start
    device = torch.device("cuda")

    print(' Start Training...')
    if args.resume is not '':
        test_iter -= 1
        best_prec_at_1 = test(dmlmodel, accuracy_calculator, device,
                              test_loader, encoder, vae, gen, args, writer)
    for epoch in range(epoch_start, args.epochs + 1):
        train(dmlmodel, clsmining, clsloss, device, train_loader,
              optimizer, epoch, encoder, vae, gen, args, writer)
        lr_scheduler.step()
        if device_ids == []:
            params = {'epoch': epoch, 'state_dict': dmlmodel.state_dict(),
                      'optimizer': optimizer.state_dict(), }
        else:
            params = {'epoch': epoch, 'optimizer': optimizer.state_dict(),
                      'state_dict': dmlmodel.module.state_dict(), }
        torch.save(params, args.results_dir + '/{}.pth'.format(args.arch))
        acc = test(dmlmodel, accuracy_calculator, device, test_loader, encoder,
                   vae, gen, args, writer)
        if acc >= best_prec_at_1:
            best_prec_at_1 = acc
            for file in glob.glob(
                    args.results_dir + '/{}_bestE*'.format(args.arch)):
                os.remove(file)
            torch.save(
                params, args.results_dir +
                '/{}_bestE{}acc{:.2f}.pth'.format(args.arch, epoch, acc*100))
            print(">>>>>> Best saved <<<<<<")
        else:
            print(">>>>>> Best not change from {} <<<<<<".format(best_prec_at_1))
        del params
