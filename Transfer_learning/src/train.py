import torch
from torch import nn
from dataset import MyData
from torch.utils.data import DataLoader
from torch.utils.data import SubsetRandomSampler
from sklearn.model_selection import KFold
import numpy as np
import os
import json
from model import LoadModule

class MyTraniner(object):
    def __init__(self, model:str, batch_size:int, learning_rate:float, epoch:int, kfold=1, aug=0, cuda=False):
        
        self.seed = torch.cuda.manual_seed(1)
        self.aug = aug
        self.json_path = r"..\data\processed\data.json"
        self.loader = LoadModule(model)
        self.batch_size=batch_size
        self.kfold = kfold
        self.using_cuda(cuda)
        self.lr = learning_rate
        self.epoch = epoch
        self.loss_fn = nn.CrossEntropyLoss()
        self.log_path = "../logs/log_" + '_'.join([self.loader.name, str(self.lr), str(self.batch_size), str(self.epoch), str(self.cuda), str(self.kfold), str(self.aug)])

    def using_cuda(self, cuda):
        if cuda is True:
            if torch.cuda.is_available():
                self.cuda = True
        else:
            self.cuda = False

    def load_data(self): 
        self.train_data = MyData(json_path=self.json_path, train=True)
        if self.aug != 0:
            self.train_data = self.train_data + MyData(json_path=self.json_path, train=True, aug=self.aug)
        self.test_data = MyData(json_path=self.json_path, train=False)
        train_dataloader = DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True, num_workers=0, generator=self.seed)
        test_dataloader = DataLoader(self.test_data, batch_size=self.batch_size, shuffle=False, num_workers=0, generator=self.seed)
        return train_dataloader, test_dataloader

    def generate_model_path(self, fold_i=1):
        base_path = "../weights/model_"
        name = '_'.join([self.loader.name, str(self.lr), str(self.batch_size), str(self.epoch), str(self.cuda), str(self.aug)])
        os.makedirs(base_path + name, exist_ok=True)
        model_path = base_path + name + '/' + '_'.join([self.loader.name, str(self.lr), str(self.batch_size), str(self.epoch), str(self.cuda), str(self.aug)]) + ".pth"
        return model_path
    
    def generate_log_dic(self):
        log_dic = dict()
        log_dic["info"] = {"model": self.loader.name,
                           "learning rate": self.lr,
                           "batch size": self.batch_size,
                           "epoch": self.epoch,
                           "using cuda": self.cuda}
        if self.kfold == 1:
            log_dic["model accuracy"] = ""
            log_dic["log"] = dict()
        else:
            log_dic["model average accuracy"] = dict()
            log_dic["log"] = dict()
        return log_dic

    @staticmethod
    def generate_epoch_dic():
        epoch_dic = dict()
        epoch_dic= {"train loss": list(),
                    "train accuracy": list(),
                    "val loss": list(),
                    "val accuracy": list()}
        return epoch_dic
    
    @staticmethod
    def write_log(epoch_dic, train_loss, train_accuracy, val_loss, val_accuracy):
        epoch_dic["train loss"].append(train_loss)
        epoch_dic["train accuracy"].append(train_accuracy)
        epoch_dic["val loss"].append(val_loss)
        epoch_dic["val accuracy"].append(val_accuracy)
        return epoch_dic
    
    def save_log(self, log_dic):
        os.makedirs(self.log_path, exist_ok=True)
        log_name = '_'.join([self.loader.name, str(self.lr), str(self.batch_size), str(self.epoch), str(self.cuda), str(self.kfold)]) + ".json"
        log_path = os.path.join(self.log_path, log_name)
        with open(log_path, 'w') as file:
             json.dump(log_dic, file)

    def train_epoch(self):
        train_loss = 0
        pred_true = 0
        self.model.train()
        for data in self.trainloader:
            imgs, labels = data

            if self.cuda is True: 
                imgs=imgs.cuda() 
                labels=labels.cuda()

            self.optim.zero_grad()

            with torch.set_grad_enabled(True):
                outputs = self.model(imgs)
                loss = self.loss_fn(outputs, labels)
                loss.backward()
                self.optim.step()

            train_loss += loss.item()
            pred_true += (outputs.argmax(1)==labels).sum().item()

        return train_loss/len(self.trainloader), pred_true/len(self.trainloader.sampler)

    def validate_epoch(self):
        val_loss = 0
        pred_true = 0
        self.model.eval()
        with torch.no_grad():
            for data in self.valloader:
                imgs, labels = data
                    
                if self.cuda is True: 
                    imgs=imgs.cuda() 
                    labels=labels.cuda()

                self.optim.zero_grad()
                
                outputs = self.model(imgs)
                loss = self.loss_fn(outputs, labels)
                val_loss += loss.item()
                pred_true += (outputs.argmax(1)==labels).sum().item()

        return val_loss/len(self.valloader), pred_true/len(self.valloader.sampler)
    
    def easy_train(self):
        try:
            # initialization for model, loaders, optimizer and writer
            self.model = self.loader()
            self.trainloader, self.valloader = self.load_data()
            self.optim = torch.optim.Adam(params=self.model.parameters(), lr=self.lr)

            log_dic = self.generate_log_dic()
            epoch_dic = self.generate_epoch_dic()

            if self.cuda is True:
                self.model.cuda()
                self.loss_fn.cuda()

            for epoch in range(1, self.epoch+1):
                train_loss, train_accuracy = self.train_epoch()
                val_loss, val_accuracy = self.validate_epoch()
                print("epoch {0}: train loss: {1} train accuracy: {2} validation loss: {3} validation accuracy: {4}".format(epoch, round(train_loss, 3), round(train_accuracy, 3), round(val_loss, 3), round(val_accuracy, 3)))
                epoch_dic = self.write_log(epoch_dic, train_loss, train_accuracy, val_loss, val_accuracy)

            log_dic["log"] = epoch_dic
            log_dic["model accuracy"] = val_accuracy
            model_path = self.generate_model_path()
            self.save_model(model_path)
            self.save_log(log_dic)
        finally:
            torch.cuda.empty_cache()

    def k_fold_train(self):
        try:
            kfold = KFold(n_splits=self.kfold, shuffle=True)
            fold_loss = list()
            fold_accs = list()
            _, _ = self.load_data() # only train_data need
            log_dic = self.generate_log_dic()

            for fold_i, (train_ids, val_ids) in enumerate(kfold.split(self.train_data)):

                print("----- fold {} -----".format(fold_i+1))
                
                # initialization for model, loaders, optimizer and writer for each fold process
                self.model = self.loader()
                self.optim = torch.optim.Adam(params=self.model.parameters(), lr=self.lr)

                if self.cuda is True:
                    self.model.cuda()
                    self.loss_fn.cuda()

                epoch_dic = self.generate_epoch_dic()

                train_sampler = SubsetRandomSampler(train_ids, generator=self.seed)
                val_sampler = SubsetRandomSampler(val_ids, generator=self.seed)

                self.trainloader = DataLoader(dataset=self.train_data, batch_size=self.batch_size, sampler=train_sampler, generator=self.seed)
                self.valloader = DataLoader(dataset=self.train_data, batch_size=self.batch_size, sampler=val_sampler, generator=self.seed)

                for epoch in range(1, self.epoch+1):

                    train_loss, train_accuracy = self.train_epoch()
                    val_loss, val_accuracy = self.validate_epoch()
                    print("epoch {0}: train loss: {1} train accuracy: {2} validation loss: {3} validation accuracy: {4}".format(epoch, round(train_loss, 3), round(train_accuracy, 3), round(val_loss, 3), round(val_accuracy, 3)))
                    epoch_dic = self.write_log(epoch_dic, train_loss, train_accuracy, val_loss, val_accuracy)

                log_dic["log"][str(fold_i+1)] = epoch_dic
                fold_loss.append(val_loss)
                fold_accs.append(val_accuracy)
                # model_path = self.generate_model_path(fold_i)
                # self.save_model(model_path)
            fold_loss = np.array(fold_loss)
            fold_accs = np.array(fold_accs)
            log_dic["model average accuracy"] = {"fold loss mean": float(fold_loss.mean()), "fold loss std": float(fold_loss.std()), "fold acc mean": float(fold_accs.mean()), "fold acc std": float(fold_accs.std())}
            self.save_log(log_dic)
                    
        finally:
            torch.cuda.empty_cache()

    def train(self):
        if self.kfold == 1:
            self.easy_train()
        elif self.kfold < 1:
            print("kfold is not valid.")
        else:
            self.k_fold_train()

    def retrain(self):
        try:
            # initialization for model, loaders, optimizer and writer
            self.model = self.loader()
            self.trainloader, _ = self.load_data()
            self.optim = torch.optim.Adam(params=self.model.parameters(), lr=self.lr)

            if self.cuda is True:
                self.model.cuda()
                self.loss_fn.cuda()

            for epoch in range(1, self.epoch+1):
                train_loss, train_accuracy = self.train_epoch()
                print("epoch {0}: train loss: {1} train accuracy: {2}".format(epoch, round(train_loss, 3), round(train_accuracy, 3)))
            model_path = self.generate_model_path()
            self.save_model(model_path)

        finally:
            torch.cuda.empty_cache()

    def accuracy(self):
        model_path = self.generate_model_path()
        self.model = self.loader()
        self.model.load_state_dict(torch.load(model_path))
        json_path = model_path.removesuffix(".pth") + ".json"

        if self.cuda is True:
                    self.model.cuda()
                    self.loss_fn.cuda()

        _, self.valloader = self.load_data()
        self.optim = torch.optim.Adam(params=self.model.parameters(), lr=self.lr)
        val_loss, val_accuracy = self.validate_epoch()

        accuracy_dic = dict()
        accuracy_dic["info"] = self.generate_log_dic()["info"]
        accuracy_dic["accuracy"] = {"test loss": val_loss, "test accuracy": val_accuracy}

        with open(json_path, 'w') as file:
             json.dump(accuracy_dic, file)
             
        print("model --- test loss: " + str(val_loss) + " test accuracy: " + str(val_accuracy))

    def save_model(self, model_path):
        torch.save(self.model.state_dict(), model_path)
