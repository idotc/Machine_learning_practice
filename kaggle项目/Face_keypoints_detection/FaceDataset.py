import numpy as np
import pandas as pd
import cv2
import torch.utils.data as data
import torch
import os

class FaceDataset(data.Dataset):
    def __init__(self, path, flag):
        super(FaceDataset, self).__init__()
        self.path = path
        self.flag = flag
        self.root = os.path.join(self.path, 'train')
        file_name = os.path.join(self.path, self.flag+'.csv')
        self.label = pd.read_csv(file_name, header=None)
        self.label = np.array(self.label)
    def __getitem__(self, index):
        img_index = index
        if self.flag == 'val':
            img_index += 6999
        img_path = os.path.join(self.root, str(img_index) + '.jpg')
        img = cv2.imread(img_path)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_norm = img_gray.reshape((1, 96, 96)) / 255.0
        img_tensor = torch.from_numpy(img_norm)
        img_tensor = img_tensor.type('torch.FloatTensor')
        label = self.label[index]
        return img_tensor, label
    def __len__(self):
        return self.label.shape[0]
