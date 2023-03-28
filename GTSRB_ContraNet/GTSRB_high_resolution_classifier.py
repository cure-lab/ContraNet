import matplotlib.pyplot as plt
import numpy as np 
import torch 
from torch import nn 
from torch import optim 
import torch.nn.functional as F 
from torchvision import datasets, transforms, models 
from torchsummary import summary

import warnings
warnings.filterwarnings("ignore")
from torch.utils.data import Dataset, DataLoader 
from torchvision import utils 
# import fiftyone as fo
# import fiftyone.zoo as foz 

import json
import os
import shutil

import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
#import tqdm
from tqdm import tqdm
import torch.nn as nn
import gtsrb_dataset as dataset

from torch.autograd import Variable

# TODO: refer to the hdf5 generation code
class CenterCropLongEdge(object):
  """Crops the given PIL Image on the long edge.
  Args:
      size (sequence or int): Desired output size of the crop. If size is an
          int instead of sequence like (h, w), a square crop (size, size) is
          made.
  """
  def __call__(self, img):
    """
    Args:
        img (PIL Image): Image to be cropped.
    Returns:
        PIL Image: Cropped image.
    """
    return transforms.functional.center_crop(img, min(img.size))

  def __repr__(self):
    return self.__class__.__name__

# norm_mean = [0.485, 0.456, 0.406]
# norm_std = [0.229, 0.224, 0.225]


data_transform = transforms.Compose([CenterCropLongEdge(), transforms.Resize(200), transforms.ToTensor(), transforms.Normalize((0.3337, 0.3064, 0.3171), ( 0.2672, 0.2564, 0.2629))])

# imagenet_cinic_dataset = datasets.ImageFolder(root='/research/dept6/yjyang/cinic/train', transform=data_transform)

# test_dataset = datasets.ImageFolder(root='/research/dept6/yjyang/cinic/test', transform=data_transform)

# train_dataset_loader = DataLoader(imagenet_cinic_dataset, batch_size=64, shuffle=True, num_workers=4)pp

# test_dataset_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)

train_dataset = dataset.GTSRB(
	root_dir = './', train=True, transform=data_transform
	)
test_dataset = dataset.GTSRB(
	root_dir='./', train=False, transform=data_transform
	)

train_dataset_loader = torch.utils.data.DataLoader(
	train_dataset, 64,
	shuffle=True,
	drop_last=False)
	
test_dataset_loader = torch.utils.data.DataLoader(
	test_dataset, 64,
	shuffle=True,
	drop_last=False)
	






# print("classes:",train_dataset_loader.dataset.classes)
print("total images:", len(train_dataset))

model = models.resnet50(pretrained=True)

# print(model)
for param in model.parameters():
	param.requires_grad = True 
model.fc = nn.Sequential(nn.Linear(2048, 512),
						nn.ReLU(),
						nn.Dropout(0.2),
						nn.Linear(512, 43),
						nn.LogSoftmax(dim=1))
print(model)

criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.003)
model.cuda()

epochs = 200
steps = 0
running_loss = 0
print_every = 200
train_losses, test_losses = [], []
print(summary(model, (3, 200, 200))) 
for epoch in range(epochs):
	for inputs, labels in train_dataset_loader:
		steps += 1
		inputs, labels = inputs.cuda(), labels.cuda()
		optimizer.zero_grad()
		logps = model.forward(inputs)
		loss = criterion(logps, labels)
		loss.backward()
		optimizer.step()
		running_loss += loss.item()

		if steps % print_every == 0:
			test_loss = 0
			accuracy = 0
			model.eval()
			with torch.no_grad():
				for inputs, labels in test_dataset_loader:
					inputs = inputs.cuda()
					labels = labels.cuda()
					logps = model.forward(inputs)
					batch_loss = criterion(logps, labels)
					test_loss += batch_loss.item()
					ps = torch.exp(logps)
					top_p, top_class = ps.topk(1, dim=1)
					equals = top_class==labels.view(*top_class.shape)
					accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
			train_losses.append(running_loss/len(train_dataset_loader))
			test_losses.append(test_loss/len(test_dataset_loader))
			print(f"Epoch {epoch+1}/{epochs} .. "
				f"Train loss: {running_loss/print_every:.3f}.."
				f"Test loss: {test_loss/len(test_dataset_loader):.3f}.. "
				f"Test accuracy: {accuracy/len(test_dataset_loader):.3f}")

			running_loss = 0
			model.train()
	torch.save(model.state_dict(), str(epoch)+"_200PREunfixstate_dict_gtsrb_classifier.pth")
	torch.save(model,str(epoch)+"_200PREunfixgtsrb_classifier.pth")
