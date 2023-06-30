"""
Implementation for custom transforms and datasets
"""

import torch
import numpy as np
from torch.utils.data import Dataset
import os
import json
import shutil
import random

def split_test_train_dataset(folder_path):
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.endswith(('.json')):
                    json_path = os.path.join(root, file)
                    with open(json_path) as json_file:
                        json_data = json.load(json_file)
                if file.endswith(('.jpg')):
                    senario = file.split('_')[-2]
                    if senario == "5":
                        position = file.split('_')[-1].removesuffix(".jpg")
                        image_path = os.path.join(root, file)
                        if random.random() <= 0.2:
                            copy_folder_path = r"C:\Users\tsx10\PythonProjectsJupyter\TUM\FP\Images\trainning\data\raw\mini MNIST JPG\test"
                            file_name = str(json_data[position])
                            destination_folder = os.path.join(copy_folder_path, file_name)
                            shutil.copy(image_path, destination_folder)
                        else:
                            copy_folder_path = r"C:\Users\tsx10\PythonProjectsJupyter\TUM\FP\Images\trainning\data\raw\mini MNIST JPG\train"
                            file_name = str(json_data[position])
                            destination_folder = os.path.join(copy_folder_path, file_name)
                            shutil.copy(image_path, destination_folder)

class MyDataset(Dataset):
    def __init__(self, paths, labels, transform=None):
        """
        Overwrites the methods of the pytorch dataset class and defines
        how to return an image and a label

        Args:
            paths: list with image paths
            labels: list with labels
            transform: None or torchvision transforms to apply to image
        """

        self.paths = paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # load specific sample
        img = np.load(self.paths[idx], allow_pickle=True)
        label = self.labels[idx]

        # to tensor
        img = torch.tensor(img, dtype=torch.float32)
        if len(img.shape) == 3:
            img = torch.permute(img, (2, 0, 1))
        elif len(img.shape) == 2:
            img = torch.unsqueeze(img, dim=0)

        label = torch.tensor(label, dtype=torch.long)

        # apply transforms
        if self.transform is not None:
            img = self.transform(img)

        return img, label


class AddNoiseToTensor(object):
    def __init__(self, p, alpha):
        """
        implements the __call__ method of a torchvision transform operation

        Args:
            p: probability with which the transform should be applied
            alpha: scaling factor for the normal distributed noise
        """
        self.p = p
        self.a = alpha

    def __call__(self, x):

        # tip: use pytorch for random operations, since it handles management of random seeds in
        # workers of dataloader correctly. Numpy for example doesn't, which results in identical samples
        if torch.rand(1) < self.p:
            # add noise from normal distribution to tensor with a given probability
            return x + torch.randn_like(x) * self.a
        return x



