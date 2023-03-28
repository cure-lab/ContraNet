# ** the largest test GTSRB image is 232 *266
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
import matplotlib.pyplot as plt
import os
import matplotlib.image as mpimg
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
from torch.autograd import Variable
# from options import Options
import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib as mpl 
mpl.use("Agg")
import os
from PIL import Image
import numpy
import matplotlib.colors as mcolors
import matplotlib.font_manager as font_manager
from matplotlib.ticker import FormatStrFormatter
import matplotlib.ticker as tck

from matplotlib import ticker
print(mpl.matplotlib_fname())
import torchvision.datasets as dset
from torchvision import utils 

import csv
import os
import shutil
try: 
	os.makedirs("./high_resolution_BTSDtrain")
except:
	pass
try:
	os.makedirs("./high_resolution_BTSDvaild")
except:
	pass 
try:
	os.makedirs("./high_resolution_BTSDtest")
except:
	pass

# csv_reader = csv.reader(open("./ImageNet2012_cifar10.csv"))
# path="/data/ssd/dataset/ImageNet/ILSVRC2012/train"
# to_path="/research/dept6/yjyang/cinic"






def ticks_format(value, index):
    """
    get the value and returns the value as:
       integer: [0,99]
       1 digit float: [0.1, 0.99]
       n*10^m: otherwise
    To have all the number of the same size they are all returned as latex strings
    """
    exp = np.floor(np.log10(value))
    base = value/10**exp
    
    if exp == 0 or exp == 1:   
        return '${0:d}$'.format(int(value))
    if exp == -1:
        return '${0:.1f}$'.format(value)
    else:
        return '${0:d}\\times10^{{{1:d}}}$'.format(int(base), int(exp))



mpl.rcParams.update(
    {
        'text.usetex': False,
        'font.family': 'stixgeneral',
        'mathtext.fontset': 'stix',
    }
)
mpl.rc('xtick', labelsize=9) 
mpl.rc('ytick', labelsize=9) 
# colors=list(mcolors.TABLEAU_COLORS.keys())

title_font = {'fontname':'Times New Roman', 'size':'12', 'color':'black', 'weight':'heavy',
              'verticalalignment':'bottom'} # Bottom vertical alignment for more space
axis_font = {'fontname':'Times New Roman', 'size':'11','style':'italic'}

# Set the font properties (for use in legend)   
font_path = '/research/dept6/yjyang/anaconda3/envs/rygaoMOCO/lib/python3.7/site-packages/matplotlib/mpl-data/fonts/ttf/times-new-roman.ttf'
font_prop = font_manager.FontProperties(fname=font_path, size=7)

transform = transforms.Compose([
		transforms.ToTensor(),
	])

# train_dataset = dataset.GTSRB(
# 	root_dir = './', train=True, transform=transform
# 	)
# test_dataset = dataset.GTSRB(
# 	root_dir='./', train=False, transform=transform
# 	)

train_dataset = dset.ImageFolder("./BelgiumTS_dataset/Training", transform=transform)
test_dataset = dset.ImageFolder("./BelgiumTS_dataset/Testing", transform=transform)


train_loader = iter(torch.utils.data.DataLoader(
	train_dataset, 1,
	shuffle=False,
	drop_last=False)
	)
test_loader = iter(torch.utils.data.DataLoader(
	test_dataset, 1,
	shuffle=False,
	drop_last=False)
	)

large_image = 0
BTSD_class_set = set([0,1,2,6,11,13,3,4,5,6,7,10,11,13,16,17,19,21,22,25,28,31,32,34,36,37,60,61])
# BTSD_class_set = set([0,1,2,6,11,13,3,4,5,6,7,10,11,13,16,17,19,21,22,25,28,31,32,34,36,37,60,61])
size_190_BTSD_class_set = [61,37,32,31,28,22,21,19,17,10, 4, 3,36,34,25,13, 0]
size_190_GTSRB_classset = [12,40, 2, 9,15,17,14,13,11,25,20,19,36,35,16,18,22]

# source_path = '/research/dept6/yjyang/SP2020/PyTorch-StudioGAN/GTSRB/high_resolution_BTSDtrain/'
source_path ='./high_resolution_BTSDtrain/'
# to_path = '/research/dept6/yjyang/SP2020/PyTorch-StudioGAN/GTSRB/high_resolution_BTSDtrain/GTSRB190_highresolution/'
to_path ='./high_resolution_BTSDtrain/GTSRB190_highresolution/'
write_path ='./GTSRB190_highresolution/'
try:
    os.makedirs(to_path)
    print("Create to_path!")
except:
    print("to_path exits!")


with open("GTSRB190_highresolution_labeling.csv", 'a+') as f:
    dirs = os.listdir(to_path)
    csv_writer = csv.writer(f)
    for dir_item in dirs:
        write_dir_path = os.path.join(write_path, dir_item)
        dir_path = os.path.join(to_path, dir_item)
        image_path_list = os.listdir(dir_path)
        for image_name in image_path_list:
            row = []
            image_path = os.path.join(write_dir_path,image_name)
            row.append(image_path)
            row.append(dir_item)
            csv_writer.writerow(row)

exit()






for i, BTSD_item in enumerate(size_190_BTSD_class_set):
    source_image_path = os.path.join(source_path, str(BTSD_item))
    destination_image_path = os.path.join(to_path, str(size_190_GTSRB_classset[i]))
    try:
        os.makedirs(destination_image_path)
    except:
        print(destination_image_path+"exist!")
        
    path_list = os.listdir(source_image_path)
    for image_path in path_list:
        shutil.copy(os.path.join(source_image_path, image_path), os.path.join(destination_image_path,image_path))

exit()



for i in range(len(test_dataset)):
	# import ipdb; ipdb.set_trace()
	# image, label = next(train_loader)
	image, label = next(test_loader)
	print(label.item())
	# if i == 63: exit()
	if label.item() in BTSD_class_set:
		long_side = min(image.shape[2], image.shape[3])
		if long_side > 190 and (long_side/max(image.shape[2], image.shape[3])>0.6):
			print(long_side)
			print(image.shape[3])
			large_image = large_image + 1
			print("test190large_image_num++++++++++++++++++++++++++++++++++++++++:",large_image)
			try:
				to_path = "./high_resolution_BTSDtrain/"
				os.makedirs(os.path.join(to_path, str(label.item())))
			except:
				print("dir exists!")
			# image_pil = transforms.ToPILImage()(image).convert('RGB')
			torchvision.utils.save_image(image, os.path.join(to_path, str(label.item()),str(i)+"test.png"))

# for image, label in train:
# 	source_image_path = os.path.join(path, line[0])
# 	try:
# 		os.makedirs(os.path.join(to_path, line[1]))
# 		print("make dir:",line[1])
# 	except:
# 		print(line[1]+"exists!")
# 	destination_image_path = os.path.join(to_path, line[1])
# 	try:
# 		path_list = os.listdir(source_image_path)
# 		for image_path in path_list:
# 			# import ipdb; ipdb.set_trace() 
# 			shutil.copy(os.path.join(source_image_path, image_path), os.path.join(destination_image_path, image_path))
# 	except:
# 		print("copy error")
# 	# fig= plt.figure(dpi=300)
# 	# ax1 = fig.add_subplot(111)

	# long_side = min(image.shape[2], image.shape[3])
	# if long_side > 190 and (long_side/max(image.shape[2], image.shape[3])>0.6):
	# 	print(long_side)
	# 	print(image.shape[3])
	# 	large_image = large_image + 1
	# 	print("test190large_image_num++++++++++++++++++++++++++++++++++++++++:",large_image)
	# 	# image_pil = transforms.ToPILImage()(image[0]).convert('RGB')
	# 	# ax1.imshow(image_pil)
	# 	# plt.tight_layout()
	# 	# plt.show()
	# 	# plt.savefig("./high_resolution_GTSRB/"+str(i)+"_trainset_"+str(long_side)+".png")
	# plt.close()
