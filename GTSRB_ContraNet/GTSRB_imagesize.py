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
print(mpl.get_cachedir())

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

train_dataset = dataset.GTSRB(
	root_dir = './', train=True, transform=transform
	)
test_dataset = dataset.GTSRB(
	root_dir='./', train=False, transform=transform
	)

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
for i in range(len(train_dataset)):
	# import ipdb; ipdb.set_trace()
	image, label = next(train_loader)
	fig= plt.figure(dpi=300)
	ax1 = fig.add_subplot(111)

	long_side = min(image.shape[2], image.shape[3])
	if long_side > 190:
		print(long_side)
		print(image.shape[3])
		large_image = large_image + 1
		print("test190large_image_num++++++++++++++++++++++++++++++++++++++++:",large_image)
		image_pil = transforms.ToPILImage()(image[0]).convert('RGB')
		ax1.imshow(image_pil)
		plt.tight_layout()
		plt.show()
		plt.savefig("./high_resolution_GTSRB/"+str(i)+"_trainset_"+str(long_side)+".png")
	plt.close()
