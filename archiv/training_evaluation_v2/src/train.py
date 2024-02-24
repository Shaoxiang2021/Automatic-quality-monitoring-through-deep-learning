import torch
from torch.nn import CrossEntropyLoss
from dataset import MyData
from torch.utils.data import DataLoader
from torch.utils.data import SubsetRandomSampler
from sklearn.model_selection import KFold
import numpy as np
import os
import json
from model import LoadModule
from pytorchtools import EarlyStopping
from torch.optim import lr_scheduler, Adam
#  from torch.utils.tensorboard import SummaryWriter

class MyTraniner(object):
    def __init__(self, config):

        self.json_path = r"..\data\processed\data_train.json"
        
        self.seed = torch.cuda.manual_seed(1)
        self.loader = LoadModule(config["model"])
        self.loss_fn = CrossEntropyLoss()

        self.aug = config["aug"]
        self.batch_size = config["batch_size"]
        self.kfold = config["kfold"]
        self.lr = config["learning_rate"]
        self.epoch = config["epoch"]
        self.early_stopping = config["early_stopping"]

        self.using_cuda(config["cuda"])

    def using_cuda(self, cuda):
        if cuda is True:
            if torch.cuda.is_available():
                self.cuda = True
        else:
            self.cuda = False

    def load_data(self): 
        self.train_data = MyData(json_path=self.json_path, train=True, aug=self.aug)
        self.test_data = MyData(json_path=self.json_path, train=False)
        train_dataloader = DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True, num_workers=8, generator=self.seed, pin_memory=True)
        test_dataloader = DataLoader(self.test_data, batch_size=self.batch_size, shuffle=False, num_workers=8, generator=self.seed, pin_memory=True)
        return train_dataloader, test_dataloader

    def generate_model_path(self):
        base_path = "../weights"
        os.makedirs(base_path, exist_ok=True)
        model_path = os.path.join(base_path, '_'.join(["model", self.loader.name, str(self.lr), str(self.batch_size), str(self.epoch), str(self.cuda), str(self.aug), str(self.early_stopping)]) + ".pth")
        return model_path
    
    def generate_log_dic(self):
        log_dic = dict()
        log_dic["info"] = {"model": self.loader.name,
                           "learning rate": self.lr,
                           "batch size": self.batch_size,
                           "epoch": self.epoch,
                           "augumentation": self.aug,
                           "early stopping": self.early_stopping,
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
        base_path = "../logs"
        os.makedirs(base_path, exist_ok=True)
        log_name = '_'.join([self.loader.name, str(self.lr), str(self.batch_size), str(self.epoch), str(self.cuda), str(self.aug), str(self.early_stopping), str(self.kfold)]) + ".json"
        log_path = os.path.join(base_path, log_name)
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
        """val_loss = 0
        # pred_true = 0

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

        return val_loss/len(self.valloader), pred_true/len(self.valloader.sampler)"""

        """val_loss = 0
        total_correct = 0
        total_samples = 0
        confusion_matrix = torch.zeros(2, 2)  # Assuming binary classification

        self.model.eval()
        with torch.no_grad():
            for data in self.valloader:
                imgs, labels = data
                    
                if self.cuda is True: 
                    imgs = imgs.cuda() 
                    labels = labels.cuda()

                outputs = self.model(imgs)
                loss = self.loss_fn(outputs, labels)
                val_loss += loss.item()

                predicted_labels = outputs.argmax(dim=1)
                total_correct += (predicted_labels == labels).sum().item()
                total_samples += labels.size(0)

                for i in range(len(labels)):
                    confusion_matrix[labels[i], predicted_labels[i]] += 1

        accuracy = total_correct / total_samples
        return val_loss/len(self.valloader), accuracy, confusion_matrix"""

        val_loss = 0
        TP = 0
        FP = 0
        FN = 0
        TN = 0
        total_samples = 0
        total_correct = 0

        self.model.eval()
        with torch.no_grad():
            for data in self.valloader:
                imgs, labels = data
                    
                if self.cuda is True: 
                    imgs = imgs.cuda() 
                    labels = labels.cuda()

                outputs = self.model(imgs)
                loss = self.loss_fn(outputs, labels)
                val_loss += loss.item()

                predicted_labels = outputs.argmax(dim=1)
                total_correct += (predicted_labels == labels).sum().item()
                total_samples += labels.size(0)

                for i in range(len(labels)):
                    if labels[i] == 0:
                        if predicted_labels[i] == 0:
                            TP += 1
                        else:
                            FN += 1
                    elif labels[i] == 1:
                        if predicted_labels[i] == 0:
                            FP += 1
                        else:
                            TN += 1

        accuracy = total_correct/total_samples
        return val_loss/len(self.valloader), accuracy, TP, FP, FN, TN
    
    def train_early_stopping(self):
        try:
            # initialization for model, loaders, optimizer and writer
            self.model = self.loader()
            self.trainloader, self.valloader = self.load_data()
            self.optim = Adam(params=self.model.parameters(), lr=self.lr)
            self.scheduler = lr_scheduler.StepLR(self.optim, step_size=5, gamma=0.9)

            if self.early_stopping is True:
                early_stopping = EarlyStopping(patience=5)

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

                self.scheduler.step()

                if self.early_stopping is True:
                    if early_stopping(self.model, val_loss) is True:
                        self.model = early_stopping.best_model
                        print("early stopping at echo {}".format(epoch))
                        break

            log_dic["log"] = epoch_dic
            log_dic["model accuracy"] = val_accuracy
            model_path = self.generate_model_path()
            self.save_model(model_path)
            self.save_log(log_dic)
        finally:
            torch.cuda.empty_cache()

    def direct_train(self):
        try:

            # writer = SummaryWriter(log_dir="..\logs")

            # initialization for model, loaders, optimizer and writer
            self.model = self.loader()
            _, _ = self.load_data()
            torch.manual_seed(1)
            train_data, validation_data = torch.utils.data.random_split(self.train_data, [int(0.8*len(self.train_data)), len(self.train_data)-int(0.8*len(self.train_data))])
            self.trainloader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True, num_workers=8, generator=self.seed, pin_memory=True)
            self.valloader = DataLoader(validation_data, batch_size=self.batch_size, shuffle=False, num_workers=8, generator=self.seed, pin_memory=True)

            self.optim = Adam(params=self.model.parameters(), lr=self.lr)

            log_dic = self.generate_log_dic()
            epoch_dic = self.generate_epoch_dic()

            if self.cuda is True:
                self.model.cuda()
                self.loss_fn.cuda()

            print("Start training ...")

            for epoch in range(1, self.epoch+1):
                train_loss, train_accuracy = self.train_epoch()
                val_loss, val_accuracy = self.validate_epoch()
                print("epoch {0}: train loss: {1} train accuracy: {2} validation loss: {3} validation accuracy: {4}".format(epoch, round(train_loss, 3), round(train_accuracy, 3), round(val_loss, 3), round(val_accuracy, 3)))
                epoch_dic = self.write_log(epoch_dic, train_loss, train_accuracy, val_loss, val_accuracy)

                # writer.add_scalar('Loss/Train', train_loss, epoch)
                # writer.add_scalar('Accuracy/Train', train_accuracy, epoch)
                # writer.add_scalar('Loss/Validation', val_loss, epoch)
                # writer.add_scalar('Accuracy/Validation', val_accuracy, epoch)

            log_dic["log"] = epoch_dic
            log_dic["model accuracy"] = val_accuracy
            # model_path = self.generate_model_path()
            # self.save_model(model_path)
            self.save_log(log_dic)

            # writer.close()
        finally:
            torch.cuda.empty_cache()

    def retrain(self):
        try:

            # initialization for model, loaders, optimizer and writer
            self.model = self.loader()
            self.trainloader, _ = self.load_data()

            self.optim = Adam(params=self.model.parameters(), lr=self.lr)

            if self.cuda is True:
                self.model.cuda()
                self.loss_fn.cuda()

            print("Start training ...")

            for epoch in range(1, self.epoch+1):
                print("epoch {} is completed ...".format(epoch))
                self.train_epoch()

            model_path = self.generate_model_path()
            self.save_model(model_path)

        finally:
            torch.cuda.empty_cache()

    def k_fold_train(self):
        try:
            kfold = KFold(n_splits=self.kfold, shuffle=True)
            fold_loss = list()
            fold_accs = list()
            _, _ = self.load_data() # only train_data needed
            log_dic = self.generate_log_dic()

            for fold_i, (train_ids, val_ids) in enumerate(kfold.split(self.train_data)):

                print("----- fold {} -----".format(fold_i+1))
                
                # initialization for model, loaders, optimizer and writer for each fold process
                self.model = self.loader()
                self.optim = Adam(params=self.model.parameters(), lr=self.lr)

                if self.cuda is True:
                    self.model.cuda()
                    self.loss_fn.cuda()

                epoch_dic = self.generate_epoch_dic()

                train_sampler = SubsetRandomSampler(train_ids, generator=self.seed)
                val_sampler = SubsetRandomSampler(val_ids, generator=self.seed)

                self.trainloader = DataLoader(dataset=self.train_data, batch_size=self.batch_size, sampler=train_sampler, generator=self.seed, num_workers=8, pin_memory=True)
                self.valloader = DataLoader(dataset=self.train_data, batch_size=self.batch_size, sampler=val_sampler, generator=self.seed, num_workers=8, pin_memory=True)

                for epoch in range(1, self.epoch+1):

                    train_loss, train_accuracy = self.train_epoch()
                    val_loss, val_accuracy = self.validate_epoch()
                    print("epoch {0}: train loss: {1} train accuracy: {2} validation loss: {3} validation accuracy: {4}".format(epoch, round(train_loss, 3), round(train_accuracy, 3), round(val_loss, 3), round(val_accuracy, 3)))
                    epoch_dic = self.write_log(epoch_dic, train_loss, train_accuracy, val_loss, val_accuracy)

                log_dic["log"][str(fold_i+1)] = epoch_dic
                fold_loss.append(val_loss)
                fold_accs.append(val_accuracy)
            fold_loss = np.array(fold_loss)
            fold_accs = np.array(fold_accs)
            log_dic["model average accuracy"] = {"fold loss mean": float(fold_loss.mean()), "fold loss std": float(fold_loss.std()), "fold acc mean": float(fold_accs.mean()), "fold acc std": float(fold_accs.std())}
            self.save_log(log_dic)
                    
        finally:
            torch.cuda.empty_cache()

    def train(self):
        if self.kfold is False:
            self.direct_train()
        elif self.kfold <= 1:
            print("kfold is not valid, k value must greater than 1.")
        else:
            self.k_fold_train()

    def validation(self, model_path=None):
        if model_path is None:
            model_path = self.generate_model_path()
        self.model = self.loader()
        self.model.load_state_dict(torch.load(model_path))
        json_path = model_path.removesuffix(".pth") + ".json"

        if self.cuda is True:
            self.model.cuda()
            self.loss_fn.cuda()

        _, self.valloader = self.load_data()
        self.optim = Adam(params=self.model.parameters(), lr=self.lr)
        val_loss, val_accuracy, TP, FP, FN, TN = self.validate_epoch()

        accuracy_dic = dict()
        accuracy_dic["info"] = self.generate_log_dic()["info"]
        accuracy_dic["accuracy"] = {"test loss": val_loss, "test accuracy": val_accuracy}

        with open(json_path, 'w') as file:
             json.dump(accuracy_dic, file)
             
        print("model --- test loss: " + str(val_loss) + " test accuracy: " + str(val_accuracy))
        return val_accuracy, TP, FP, FN, TN
    
    def evaluation(self, json_file, model_path=None):
        if model_path is None:
            model_path = self.generate_model_path()
        self.model = self.loader()
        self.model.load_state_dict(torch.load(model_path))

        json_path = "evaluation_analyse/" + "analyse" + json_file.removeprefix("evaluation") 

        if self.cuda is True:
            self.model.cuda()
            self.loss_fn.cuda()

        self.evaluation_data = MyData(json_path="../data/processed/" + json_file, train=False, evaluation=True)
        self.valloader = DataLoader(self.evaluation_data, batch_size=self.batch_size, shuffle=False, num_workers=8, generator=self.seed, pin_memory=True)

        self.optim = Adam(params=self.model.parameters(), lr=self.lr)
        val_loss, val_accuracy, TP, FP, FN, TN = self.validate_epoch()

        accuracy_dic = dict()
        accuracy_dic["info"] = self.generate_log_dic()["info"]
        accuracy_dic["accuracy"] = {"test loss": val_loss, "test accuracy": val_accuracy}

        with open(json_path, 'w') as file:
             json.dump(accuracy_dic, file)
             
        print("model --- test loss: " + str(val_loss) + " test accuracy: " + str(val_accuracy))
        return val_accuracy, TP, FP, FN, TN

    def save_model(self, model_path):
        torch.save(self.model.state_dict(), model_path)
