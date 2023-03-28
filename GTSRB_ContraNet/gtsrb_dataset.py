import torch
import os
import pandas as pd
from torch.utils.data import Dataset
import numpy as np
from PIL import Image

class GTSRB(Dataset):
    base_folder = 'GTSRB'

    def __init__(self, root_dir, train=False, transform=None):

        self.root_dir = root_dir

        self.sub_directory = 'trainingset' if train else 'testset'
        self.csv_file_name = 'training.csv' if train else 'test.csv'

        csv_file_path = os.path.join(
                root_dir, self.base_folder, self.sub_directory, self.csv_file_name
                )

        self.csv_data = pd.read_csv(csv_file_path)

        self.transform = transform




    def __len__(self):
        return len(self.csv_data)


    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir,self.base_folder, self.sub_directory,
                self.csv_data.iloc[idx, 0])

        img = Image.open(img_path)

        classId = self.csv_data.iloc[idx, 1]

        if self.transform is not None:
            img = self.transform(img)

        return img, classId
    
