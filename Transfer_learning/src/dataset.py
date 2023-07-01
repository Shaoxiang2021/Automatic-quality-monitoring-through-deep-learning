import torch
from torch.utils.data import Dataset
from torchvision import transforms

from PIL import Image
import numpy as np
import os
import json
import shutil
import random

"""
split original dataset to data/raw

preprocessing scripts to move the data from
data/raw to data/processed.
"""

# original dataset split into train and test dataset
def split_dataset(folder_path:str, output_train_path:str, output_test_path:str):
    """
        the split function is only for specific forms.

        Args:
            folder_paht: path for original dataset, underfolder is sample ids
            output_train_path: path for output train-dataset
            output_test_path: path for outpupt test-dataset
    """
    for root, _, files in os.walk(folder_path):
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
                        copy_folder_path = output_test_path
                        file_name = str(json_data[position])
                        destination_folder = os.path.join(copy_folder_path, file_name)
                        shutil.copy(image_path, destination_folder)
                    else:
                        copy_folder_path = output_train_path
                        file_name = str(json_data[position])
                        destination_folder = os.path.join(copy_folder_path, file_name)
                        shutil.copy(image_path, destination_folder)

# convert images into numpy files and move them to processed folder
# caculate mean and standard, save them in json file with path for each numpy file
def preprocess_data(src_dir:str, out_dir:str):

    out_json = {
        "info": {
            "mean": np.zeros(3),
            "std": np.zeros(3),
        }
    }

    for split in ["train", "test"]:
        out_json[split] = []

        for label in os.listdir(f"{src_dir}/{split}"):
            print("processing", label)
            curr_out_dir = f"{out_dir}/{split}/{label}"
            os.makedirs(curr_out_dir, exist_ok=True)

            for file_name in os.listdir(f"{src_dir}/{split}/{label}"):
                # open image in greyscale mode
                img_frame = Image.open(f"{src_dir}/{split}/{label}/{file_name}")
                # convert to numpy and scale
                img_np = np.array(img_frame)/255.
                np.save(f"{curr_out_dir}/{file_name.replace('.jpg', '.npy')}", img_np)

                # append path and label
                out_json[split].append({
                    "path": f"{curr_out_dir}/{file_name.replace('.jpg', '.npy')}",
                    "label": int(label)
                })

                if split == "train":
                    # compute stats standardization and save them to json
                    mean = list()
                    std = list()
                    for i in range(img_np.shape[2]):
                        pixels = img_np[:, :, i].ravel()
                        mean.append(np.mean(pixels))
                        std.append(np.std(pixels))

                    out_json["info"]["mean"] += np.array(mean)
                    out_json["info"]["std"] += np.array(std)

    out_json["info"]["mean"] = list(out_json["info"]["mean"]/len(out_json["train"]))
    out_json["info"]["std"] = list(out_json["info"]["std"]/len(out_json["train"]))
    with open(out_dir + "/data.json", "w") as file:
        file.write(json.dumps(out_json, indent=4))

class MyData(Dataset):
    def __init__(self, json_path:str, train:bool, transform=None, aug=0):
        self.transform = transform
        self.aug = aug
        with open(json_path) as file:
            self.data_dic = json.load(file)
        self.mean = self.data_dic["info"]["mean"]
        self.std = self.data_dic["info"]["std"]

        if train is True:
            if self.aug == 0:
                self.paths = [sample["path"] for sample in self.data_dic["train"]]
                self.labels = [sample["label"] for sample in self.data_dic["train"]]
            elif 0 < self.aug < 1:
                self.paths = [sample["path"] for sample in self.data_dic["train"]]
                self.labels = [sample["label"] for sample in self.data_dic["train"]]
                random.seed(1)
                for _ in range(int(len(self.paths)*(1-aug))):
                    idx = random.randint(0, len(self.paths)-1)
                    self.paths.pop(idx)
                    self.labels.pop(idx)
            else:
                print("wrong aug input.")
        else:
            self.paths = [sample["path"] for sample in self.data_dic["test"]]
            self.labels = [sample["label"] for sample in self.data_dic["test"]]

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        # load specific sample
        img = np.load(self.paths[index], allow_pickle=True)
        label = self.labels[index]

        if self.aug != 0:
            img = self.augumentation(img)
        
        transformer_train = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=self.mean, std=self.std)])
        img = transformer_train(img)
        img = img.type(torch.float32)
        label = torch.tensor(label, dtype=torch.long)

        # apply transforms
        if self.transform is not None:
            img = self.transform(img)
        
        return img, label
    
    def __len__(self):
        return len(self.paths)
    
    @staticmethod
    def augumentation(img):
        img = Image.fromarray((img*255).astype(np.uint8))

        idx = random.randint(1, 4)
        if idx == 1:
            transformer = transforms.RandomRotation(degrees=10)
        elif idx == 2:
            transformer = transforms.GaussianBlur((7, 7), sigma=1)
        elif idx == 3:
            transformer = transforms.RandomAutocontrast(p=1)
        else:
            transformer = transforms.ColorJitter(brightness=1)
        img = transformer(img)

        img = np.array(img)/255
        return img
