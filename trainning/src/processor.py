"""
minimum working example:
performs classification on mini MNIST
"""

import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import json

from data import MyDataset, AddNoiseToTensor
from model import MyModel
from workflow import MyTrainer

import shutil
import os
from data import split_test_train_dataset
from preprocessing import preprocess_mini_mnist

class Processor():
    def __init__(self, USE_CUDA, parameter, learning_rate=0.0001, echos=100):
        self.USE_CUDA = USE_CUDA
        self.in_channels, self.num_classes, self.channels, self.num_blocks = parameter
        self.lr = learning_rate
        self.echos = echos
        self.logger_dic = dict()

    @staticmethod
    def delete_files_in_folder(folder_path):
        shutil.rmtree(folder_path)

    def reflash_data(self):
        self.delete_files_in_folder(r"C:\Users\tsx10\PythonProjectsJupyter\TUM\FP\Images\trainning\data")
        # prepare folder
        folder_names = ["test/0", "test/1", "train/0", "train/1"]
        for name in folder_names:
            os.makedirs("../data/processed/mini MNIST npy/" + name, exist_ok=True)
        for name in folder_names:
            os.makedirs("../data/raw/mini MNIST JPG/" + name, exist_ok=True)

    @staticmethod
    def generate_dataset():
        folder_path = r"C:\Users\tsx10\PythonProjectsJupyter\TUM\FP\Images\split_images_11.06.23"
        split_test_train_dataset(folder_path)
        preprocess_mini_mnist("../data/raw/mini MNIST JPG", out_dir="../data/processed/mini MNIST npy")

    def processing(self, times=5):
        
        self.logger_dic["info"] = {"in_channels": self.in_channels, "num_classes": self.num_classes, "channels": self.channels, "num_blocks": self.num_blocks, "learning_rate": self.lr, "echos": self.echos}
        for i in range(times):

            # self.reflash_data()
            # self.generate_dataset()

            with open(r"C:\Users\tsx10\PythonProjectsJupyter\TUM\FP\Images\trainning\data\processed\mini MNIST npy\data.json") as file:
                data = json.load(file)

            # get paths and labels
            tr_paths = [sample["path"] for sample in data["train"]]
            tr_labels = [sample["label"] for sample in data["train"]]
            vl_paths = [sample["path"] for sample in data["test"]]
            vl_labels = [sample["label"] for sample in data["test"]]

            # build data augmentations
            # (torchvision transforms also provides many different pre-built augmentation and conversion transforms)
            tr_transform = transforms.Compose([
                transforms.Normalize(mean=data["info"]["mean"], std=data["info"]["std"])
                # AddNoiseToTensor(p=0.5, alpha=0.01)
            ])

            vl_transform = transforms.Compose([
                transforms.Normalize(mean=data["info"]["mean"], std=data["info"]["std"])
            ])

            # build datasets and dataloaders
            # dataloaders handle batching of samples and multiprocessing (cores defined by num_workers)
            trainset = MyDataset(paths=tr_paths, labels=tr_labels, transform=tr_transform)
            valset = MyDataset(paths=vl_paths, labels=vl_labels, transform=vl_transform)
            trainloader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=0)
            valloader = DataLoader(valset, batch_size=1, shuffle=False, num_workers=0)

            # define model
            model = MyModel(self.in_channels, self.num_classes, self.channels, self.num_blocks)
            if self.USE_CUDA:
                model.cuda()  # .cuda() just pushes model to first available gpu device, alternative: model.to(<device_name>)

            # define optimizer after pushing model to device
            optimizer = torch.optim.Adam(model.parameters(), self.lr)
            criterion = torch.nn.CrossEntropyLoss(weight=None, reduction="mean")

            # define trainer
            t = MyTrainer(model=model,
                        trainloader=trainloader,
                        valloader=valloader,
                        optim=optimizer,
                        crit=criterion,
                        cuda=self.USE_CUDA)

            # let's gooo!
            logger = t.train(self.echos)

            model_folder_name = "_".join([str(self.in_channels), str(self.num_classes), str(self.channels), str(self.num_blocks), str(self.lr), str(self.echos)])
            path = "../saved_weights/" + model_folder_name
            os.makedirs(path, exist_ok=True)
            model_path = os.path.join(path, model_folder_name+"_{}.pt".format(i+1))
            t.save_model(model_path)

            self.logger_dic[str(i+1)] = logger

        json_path = os.path.join(path, model_folder_name + ".json")
        with open(json_path, 'w') as file:
            json.dump(self.logger_dic, file)
