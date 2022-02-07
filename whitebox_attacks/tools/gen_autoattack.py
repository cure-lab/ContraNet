import sys
sys.path.append(".")
sys.path.append("lib")

import argparse
import os
import glob
import torch
from lib.autoattack import AutoAttack
from misc.load_dataset import LoadDataset
from torch.utils.data import DataLoader
from lib.robustbench.utils import load_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-m', '--model', default='Gowal2020Uncovering_70_16', type=str)
    parser.add_argument('--batch_size', default=16, type=int,
                        metavar='N', help='mini-batch size')
    parser.add_argument('--img_size', default=32, type=int)
    parser.add_argument('--data_path', default="./dataset",
                        type=str)
    parser.add_argument('--attack', type=str,
                        default=['apgd-ce', 'apgd-t', 'fab-t', 'square'],
                        nargs='*')
    parser.add_argument('--norm', type=str, default='Linf')
    parser.add_argument('--dataset_name', type=str,
                        default='cifar10')
    parser.add_argument('--epsilon', type=float, default=8. / 255.)
    parser.add_argument('--save_dir', type=str,
                        default='results/log/autoattack')
    parser.add_argument('--log_path', type=str,
                        default='{}_log_file.log')
    parser.add_argument('--version', type=str, default='standard')
    args = parser.parse_args()

    args.save_dir = os.path.join(args.save_dir, args.model)
    args.log_path = os.path.join(
        args.save_dir, args.log_path.format("_".join(args.attack)))

    # config model
    classifier = load_model(
        model_name=args.model, dataset=args.dataset_name,
        threat_model='Linf')
    classifier.cuda()
    classifier.eval()

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
    adversary = AutoAttack(classifier, norm=args.norm,
                           log_path=args.log_path,
                           eps=args.epsilon, version=args.version)
    adversary.attacks_to_run = args.attack
    print("This is {} AA {} for {} under {}".format(
        args.norm, args.version, args.attack, args.epsilon))

    l = [x for (x, _) in test_loader]
    x_test = torch.cat(l, 0)
    l = [y for (_, y) in test_loader]
    y_test = torch.cat(l, 0)

    # attack
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    print("Start attack ...")
    load_name = '{}/{}_{}_*_eps_{:.5f}.pth'.format(
                args.save_dir, "_".join(args.attack),
                args.version, args.epsilon)
    load_name = glob.glob(load_name)
    if len(load_name) > 0:
        x_adv = torch.load(load_name[0])["adv_complete"]
        save_name = load_name[0]
        print("continue from {}".format(save_name))
    else:
        x_adv = torch.Tensor()
        save_name = ""
    with torch.no_grad():
        for i in range(0, 10000, 100):
            if i < x_adv.shape[0]:
                print("skip gen for {}-{}".format(i, i + 100))
                continue
            x_adv_update = adversary.run_standard_evaluation(
                x_test[i:i + 100], y_test[i:i + 100], bs=args.batch_size)

            x_adv = torch.cat([x_adv, x_adv_update], dim=0)
            if save_name != "":
                os.remove(save_name)
            save_name = '{}/{}_{}_{}_eps_{:.5f}.pth'.format(
                args.save_dir, "_".join(args.attack),
                args.version, x_adv.shape[0],
                args.epsilon)
            torch.save({'adv_complete': x_adv}, save_name)
