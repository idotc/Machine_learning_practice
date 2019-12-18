import torch
from torch.utils import data
import numpy as np
import pandas as pd
import cv2

class FaceDataset(data.Dataset):
    def __init__(self, root):
        super(FaceDataset, self).__init__()
        self.root = root
        pic = pd.read_csv(self.root+"/dataset.csv", header=None, usecols=[0])
        label = pd.read_csv(self.root+"/dataset.csv", header=None, usecols=[1])
        self.pic = np.array(pic)[:, 0]
        self.label = np.array(label)[:, 0]
    
    def __getitem__(self, index):
        face = cv2.imread(self.root+"/"+self.pic[index])
        face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        face_hist = cv2.equalizeHist(face_gray)
        face_normalized = face_hist.reshape((1, 48, 48)) / 255.0
        face_tensor = torch.from_numpy(face_normalized)
        face_tensor = face_tensor.type('torch.FloatTensor')
        label = self.label[index]
        return face_tensor, label
    
    def __len__(self):
        return self.pic.shape[0]
