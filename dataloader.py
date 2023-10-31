import nibabel as nib
import SimpleITK as sitk
import numpy as np
import cv2
import re
import random
from random import shuffle
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
from torch import optim
import torch.nn.functional as F


# 定义数据集
class MyDataset(Dataset):
    def __init__(self, image_path, label_path,train=False,transform=None):
        super(MyDataset, self).__init__()
        self.image_path = image_path
        self.image_file_list = sorted(os.listdir(image_path))
        self.label_path = label_path
        self.label_file_list = sorted(os.listdir(label_path))
        self.train = train
        self.transform = transform

    def __len__(self):
        return len(self.image_flie_list)

    def __getitem__(self, index):
        image_file = self.image_file_list[index]
        label_file = self.label_file_list[index]

        # 读取npy或nii并归一化【0,1】
        image = np.load(os.path.join(self.image_path,image_file))
        image_array = np.array(image,dtype=np.float32) / 255.0
        label = np.load(os.path.join(self.label_path,label_file))
        label_array = np.array(label,dtype=np.float32)

        # image = sitk.ReadImage(os.path.join(image_path,image_file))
        # image_array = sitk.GetArrayFromImage(image) / 255   ## 此时原本的数据格式由（H,W,D)变成了（D，H，W）
        # # image_array = image_array.astype(np.float32)
        # label = sitk.ReadImage(os.path.join(label_path,label_file))
        # label_array = sitk.GetArrayFromImage(label)

        # 预处理
        if self.transform is not None:
            image_array = self.transform(image_array)
            label_array = self.transform(label_array)

        # 转为tensor
        image_tensor = torch.from_numpy(image_array)
        label_tensor = torch.from_numpy(label_array)

        return image_tensor,label_tensor
