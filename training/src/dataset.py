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

def split_evaluation_dataset(folder_path:str, output_evaluation_path:str):
    print("starting spliting data ...")
    for root, _, files in os.walk(folder_path):
         for file in files:
            if file.endswith(('.json')):
                json_path = os.path.join(root, file)
                with open(json_path) as json_file:
                    json_data = json.load(json_file)
            if file.endswith(('.jpg')):
                senario = file.split('_')[-2]
                if senario != "5":
                    position = file.split('_')[-1].removesuffix(".jpg")
                    image_path = os.path.join(root, file)
                    copy_folder_path = output_evaluation_path
                    file_name = str(json_data[position])
                    destination_folder = os.path.join(copy_folder_path, file_name)
                    shutil.copy(image_path, destination_folder)

def preprocess_evaluation_data(src_dir:str, out_dir:str):

    for label in os.listdir(f"{src_dir}"):
        print("processing", label)
        curr_out_dir = f"{out_dir}/{label}"
        os.makedirs(curr_out_dir, exist_ok=True)

        for file_name in os.listdir(f"{src_dir}/{label}"):
             # open image in greyscale mode
            img_frame = Image.open(f"{src_dir}/{label}/{file_name}")
            # convert to numpy and scale
            img_np = np.array(img_frame)/255.
            np.save(f"{curr_out_dir}/{file_name.replace('.jpg', '.npy')}", img_np)

def making_evaluation_json(src_dir:str, out_dir:str, szenario:list, filename):

    out_json = {
        "info": {
            "mean": [0.5464275974422526,
                     0.480967096142885,
                     0.4913222950300173],
            "std": [0.23247085631060224,
                    0.21001021511970114,
                    0.2134178785697123]
                }
            }
    out_json["evaluation"] = list()

    for label in os.listdir(f"{src_dir}"):
        print("processing", label)

        for file_name in os.listdir(f"{src_dir}/{label}"):
            
            szenario_index = int(file_name.split('_')[-2])
            if szenario_index in szenario:
            # append path and label
                out_json["evaluation"].append({
                    "path": f"{src_dir}/{label}/{file_name}",
                    "label": int(label)
            })

    with open(out_dir + "/" + filename + ".json", "w") as file:
        file.write(json.dumps(out_json, indent=4))

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
    with open(out_dir + "/data_train.json", "w") as file:
        file.write(json.dumps(out_json, indent=4))

class MyData(Dataset):
    def __init__(self, json_path:str, train:bool, evaluation=False, aug=False):
        self.aug = aug
        with open(json_path) as file:
            self.data_dic = json.load(file)
        self.mean = self.data_dic["info"]["mean"]
        self.std = self.data_dic["info"]["std"]

        if evaluation is True:
            self.paths = [sample["path"] for sample in self.data_dic["evaluation"]]
            self.labels = [sample["label"] for sample in self.data_dic["evaluation"]]
        else:
            if train is True:
                self.paths = [sample["path"] for sample in self.data_dic["train"]]
                self.labels = [sample["label"] for sample in self.data_dic["train"]]
            else:
                self.paths = [sample["path"] for sample in self.data_dic["test"]]
                self.labels = [sample["label"] for sample in self.data_dic["test"]]
        
        # AD Version 1
        self.tr_transform = transforms.Compose([transforms.ToTensor(), 
                                                transforms.RandomRotation(degrees=5), 
                                                transforms.ColorJitter(hue=0.1, saturation=0.1, brightness=0.7), 
                                                transforms.GaussianBlur((7, 7), sigma=(0.1, 0.5)), 
                                                transforms.Normalize(mean=self.mean, std=self.std)])
        # AD Version 2
        """self.tr_transform = transforms.Compose([transforms.ToTensor(), 
                                                transforms.RandomAffine(degrees=2),
                                                transforms.RandomHorizontalFlip(p=0.5),
                                                transforms.RandomVerticalFlip(p=0.5),
                                                transforms.RandomGrayscale(p=0.1),
                                                transforms.Normalize(mean=self.mean, std=self.std)])"""

        self.vl_transform = transforms.Compose([transforms.ToTensor(), 
                                                transforms.Normalize(mean=self.mean, std=self.std)])

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        # load specific sample
        img = np.load(self.paths[index], allow_pickle=True)
        label = self.labels[index]
        
        if self.aug is True:
            img = self.tr_transform(img)
        else:
            img = self.vl_transform(img)
        
        img = img.type(torch.float32)
        label = torch.tensor(label, dtype=torch.long)
        
        return img, label
    
    def __len__(self):
        return len(self.paths)
