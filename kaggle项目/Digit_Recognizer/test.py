import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from DigitCNN import DigitCNN
from DigitDataset import DigitDataset

def test(model, test_dataset, batch_size):
    dataload = data.DataLoader(test_dataset, batch_size)
    result, num = 0, 0
    result_preds = []
    for imgs, labels in dataload:
        imgs = imgs.cuda()
        preds = model.forward(imgs)
        preds = preds.cpu()
        preds = np.argmax(preds.data.numpy(), axis=1)
        #result += np.sum((preds==labels.data.numpy()))
        #num += len(labels)
        result_preds.extend(preds)
    return result_preds

def main():
    #train_path = r"/media/root/515e7d3a-49ac-40be-ba58-fef9702d123c/work_record/digit_recognizer/digit-recognizer/digit-recognizer/1216/digit/train"
    test_path = r"/media/root/515e7d3a-49ac-40be-ba58-fef9702d123c/work_record/digit_recognizer/digit-recognizer/digit-recognizer/1216/digit/test"
    #train_dataset = DigitDataset(train_path)
    test_dataset = DigitDataset(test_path)
    gpus = [0]
    model = DigitCNN()
    model_paras = torch.load('digit_final_400.pkl')
    if isinstance(model_paras, torch.nn.DataParallel):
        model_paras = model_paras.module
    #net.load_state_dict(torch.load('model_net.pkl'), strict=False)
    #model.load_state_dict(torch.load("digit_final_400.pkl"))
    #model.load_state_dict(model_paras, strict=False)
    #model = torch.nn.DataParallel(model, device_ids=cfg.GPUS).cuda()
    #model = torch.nn.DataParallel(model, device_ids=gpus).cuda()
    model_paras = model_paras.eval()
    preds = test(model_paras, test_dataset, 512)
    index = [i for i in range(1, len(preds)+1)]
    preds_s = pd.Series(preds)
    index_s = pd.Series(index)
    df = pd.DataFrame()
    df["ImageId"] = index_s
    df["Label"] = preds_s
    df.to_csv("test_result.csv", index=None)

if __name__ == "__main__":
    main()
