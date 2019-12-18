import cv2
import numpy as np
import pandas as pd

data = pd.read_csv('train.csv')
imgs = data.iloc[:, 1:]
labels = data.iloc[:, 0]
