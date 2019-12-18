import torch.utils.data as data
import torch
import numpy as np
import pandas as pd
import cv2
import os

class DigitDataset(data.Dataset):
    def __init__(self, root):
        super(DigitDataset, self).__init__()
        self.root = root
        img = pd.read_csv(self.root+'/dataset.csv', header=None, usecols=[0])
        label = pd.read_csv(self.root+'/dataset.csv', header=None, usecols=[1])
        self.imgs = np.array(img)[:, 0]
        self.labels = np.array(label)[:, 0]

    def __getitem__(self, index):
        imgname = self.imgs[index]
        img = cv2.imread(self.root+'/'+imgname)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_nomalized = img_gray.reshape((1, 28, 28)) / 255.0
        img_tensor = torch.from_numpy(img_nomalized)
        img_tensor = img_tensor.type("torch.FloatTensor")
        label = self.labels[index]
        return img_tensor, label

    def __len__(self):
        return self.imgs.shape[0]
