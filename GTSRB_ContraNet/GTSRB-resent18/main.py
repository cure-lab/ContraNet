import os
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import datasets, transforms
import numpy as np
import sys
sys.path.append('../')
from model import Net
from mymodels.resnet import ResNet10APP
from utils import progress_bar
from gtsrb_dataloader import GTSRB
from data import data_transforms,data_jitter_hue,data_jitter_brightness,data_jitter_saturation,data_jitter_contrast,data_rotate,data_hvflip,data_shear,data_translate,data_center,data_hflip,data_vflip # data.py in the same folder

# Training settings
parser = argparse.ArgumentParser(description='PyTorch GTSRB example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 100)')
parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                    help='learning rate (default: 0.0001)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--size', '-s', default=1.0, type=float)
parser.add_argument('--model', '-m', default='ResNet10APP')

args = parser.parse_args()
model_name = args.model

torch.manual_seed(args.seed)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch


   
# Apply data transformations on the training images to augment dataset
print('==> Preparing data..')
# train_loader = torch.utils.data.DataLoader(
#    torch.utils.data.ConcatDataset([datasets.ImageFolder(args.data_dir + '/train',
#    transform=data_transforms),
#    datasets.ImageFolder(args.data_dir + '/train',
#    transform=data_jitter_brightness),datasets.ImageFolder(args.data_dir + '/train',
#    transform=data_jitter_hue),datasets.ImageFolder(args.data_dir + '/train',
#    transform=data_jitter_contrast),datasets.ImageFolder(args.data_dir + '/train',
#    transform=data_jitter_saturation),datasets.ImageFolder(args.data_dir + '/train',
#    transform=data_translate),datasets.ImageFolder(args.data_dir + '/train',
#    transform=data_rotate),datasets.ImageFolder(args.data_dir + '/train',
#    transform=data_hvflip),datasets.ImageFolder(args.data_dir + '/train',
#    transform=data_center),datasets.ImageFolder(args.data_dir + '/train',
#    transform=data_shear)]), batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)

train_loader = torch.utils.data.DataLoader(
  datasets.ImageFolder('./data/datasets'+'/train', transform=data_transforms),
  batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
   
val_loader = torch.utils.data.DataLoader(
    GTSRB('./data/datasets/', train=False, transform=data_transforms),
    batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)


# Neural Network and Optimizer
print('==> Building model..')
model = ResNet10APP(np.around(float(args.size),3), 43)
#model = ResNet18()

model = model.to(device)
if device == 'cuda':
    model = torch.nn.DataParallel(model)
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad,model.parameters()),lr=args.lr)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',patience=5,factor=0.5,verbose=True)

def train(epoch):
    print('\nEpoch: %d' % epoch)
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(train_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))


def test(epoch):
    global best_acc
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(val_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # scheduler.step(np.around(test_loss,2))
    
    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': model.module.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/' + args.model + '_' + np.around(str(args.size),3) + '.pth')
        best_acc = acc


for epoch in range(start_epoch, start_epoch+args.epochs):
    train(epoch)
    test(epoch)

