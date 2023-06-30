from torch.utils.tensorboard import SummaryWriter
import torch
from torch import nn
from dataset import MyData
from torch.utils.data import DataLoader
from torch.utils.data import SubsetRandomSampler
from sklearn.model_selection import KFold
import numpy as np
import os

class MyTraniner(object):
    def __init__(self, model, batch_size:int, learning_rate:float, epoch:int, cuda=False):

        self.json_path = r"..\data\processed\data.json"
        self.batch_size=batch_size
        self.model = model
        self.trainloader, self.valloader = self.load_data()
        self.testloader = self.valloader
        self.using_cuda(cuda)

        self.lr = learning_rate
        self.epoch = epoch
        self.loss_fn = nn.CrossEntropyLoss()
        self.optim = torch.optim.Adam(params=self.model.parameters(), lr=self.lr)

        self.log_path = "../logs/log_" + '_'.join([self.model._get_name(), str(self.lr), str(self.batch_size), str(self.epoch), str(self.cuda)])
        self.writer = SummaryWriter(log_dir=self.log_path)

        if self.cuda is True:
            self.model.cuda()
            self.loss_fn.cuda()

    def using_cuda(self, cuda):
        if cuda is True:
            if torch.cuda.is_available():
                self.cuda = True
        else:
            self.cuda = False

    def load_data(self): 
        self.train_data = MyData(json_path=self.json_path, train=True)
        self.val_data = MyData(json_path=self.json_path, train=False)
        train_dataloader = DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True, num_workers=0)
        val_dataloader = DataLoader(self.val_data, batch_size=self.batch_size, shuffle=False, num_workers=0)
        return train_dataloader, val_dataloader

    # --- trainning with train- and testdataset ---
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
        for data in self.valloader:
            imgs, labels = data
                
            if self.cuda is True: 
                imgs=imgs.cuda() 
                labels=labels.cuda()

            self.optim.zero_grad()
            with torch.no_grad():
                outputs = self.model(imgs)
                loss = self.loss_fn(outputs, labels)
            val_loss += loss.item()
            pred_true += (outputs.argmax(1)==labels).sum().item()
        return val_loss/len(self.valloader), pred_true/len(self.valloader.sampler)
    
    def train(self):
        try:
            for epoch in range(1, self.epoch+1):
                train_loss, train_accuracy = self.train_epoch()
                val_loss, val_accuracy = self.validate_epoch()
                print("epoch {0}: train loss: {1} train accuracy: {2} validation loss: {3} validation accuracy: {4}".format(epoch, round(train_loss, 3), round(train_accuracy, 3), round(val_loss, 3), round(val_accuracy, 3)))
                self.writer.add_scalars("Model loss", {"train loss":train_loss, "validation loss":val_loss}, epoch)
                self.writer.add_scalars("Model accuracy", {"train accuracy":train_accuracy, "validation accuracy":val_accuracy}, epoch)
            model_path = "../weights/model_" + '_'.join([self.model._get_name(), str(self.lr), str(self.batch_size), str(self.epoch), str(self.cuda)]) + ".pth"
            self.save_model(model_path)
        finally:
            # self.save_graph()
            self.writer.close()
            torch.cuda.empty_cache()

    # --- trainning with k-fold, train-, validation- and testdataset ---
    def k_fold_train(self, k):
        try:
            kfold = KFold(n_splits=k)
            fold_loss = list()
            fold_accs = list()

            for fold_i, (train_ids, val_ids) in enumerate(kfold.split(self.train_data)):
                self.writer = SummaryWriter(log_dir=self.log_path)

                train_sampler = SubsetRandomSampler(train_ids)
                val_sampler = SubsetRandomSampler(val_ids)

                self.trainloader = DataLoader(dataset=self.train_data, batch_size=self.batch_size, sampler=train_sampler)
                self.valloader = DataLoader(dataset=self.train_data, batch_size=self.batch_size, sampler=val_sampler)

                for epoch in range(1, self.epoch+1):
                    train_loss, train_accuracy = self.train_epoch()
                    val_loss, val_accuracy = self.validate_epoch()
                    print("epoch {0}: train loss: {1} train accuracy: {2} validation loss: {3} validation accuracy: {4}".format(epoch, round(train_loss, 3), round(train_accuracy, 3), round(val_loss, 3), round(val_accuracy, 3)))
                    self.writer.add_scalars("Model loss kfold {}".format(fold_i), {"train loss" :train_loss, "validation loss":val_loss}, epoch)
                    self.writer.add_scalars("Model accuracy kfold {}".format(fold_i), {"train accuracy":train_accuracy, "validation accuracy":val_accuracy}, epoch)
                fold_loss.append(val_loss)
                fold_accs.append(val_accuracy)
                model_path = "../weights/model_" + '_'.join([self.model._get_name(), str(self.lr), str(self.batch_size), str(self.epoch), str(self.cuda)]) + '/' + '_'.join([self.model._get_name(), str(self.lr), str(self.batch_size), str(self.epoch), str(self.cuda)]) + ".pth"
                os.makedirs("../weights/model_" + '_'.join([self.model._get_name(), str(self.lr), str(self.batch_size), str(self.epoch), str(self.cuda)]), exist_ok=True)
                self.save_model(model_path)
            fold_loss = np.array(fold_loss)
            fold_accs = np.array(fold_accs)
            return [fold_loss.mean, fold_loss.std, fold_accs.mean, fold_accs.std]
                    
        finally:
            self.writer.close()
            torch.cuda.empty_cache()

    def save_model(self, model_path):
        torch.save(self.model.state_dict(), model_path)

    def save_graph(self):
        for data in self.valloader:
            imgs, labels = data
            img = imgs[0]
            if img is not None:
                break
        self.writer.add_graph(self.model, img)
