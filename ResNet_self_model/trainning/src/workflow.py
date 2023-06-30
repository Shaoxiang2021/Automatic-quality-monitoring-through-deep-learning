"""
implements the trainer class for a more structured workflow
"""

import torch


class MyTrainer(object):
    def __init__(self, model, trainloader, valloader, optim, crit, cuda=False):
        """
        Implements the training and validation workflow for a given model, dataloaders,
        etc.
        """

        self._model = model
        self._trainloader = trainloader
        self._valloader = valloader
        self._optim = optim
        self._crit = crit
        self._cuda = cuda

    def train(self, epochs):

        # to save training logs (keys must match those returned by train_epoch an val_epoch)
        logger = {
            "tr loss": [],
            "vl loss": [],
            "tr acc.": [],
            "vl acc.": []
        }

        # loop through epochs
        for ep in range(1, epochs+1):

            tr_res = self.train_epoch()
            for k, v in tr_res.items():
                logger[k].append(v)

            vl_res = self.validate_epoch()
            for k, v in vl_res.items():
                logger[k].append(v)

            # print stats
            print("-"*150)
            print(f"epoch {ep} | ", end="")
            for k, v in logger.items():
                print(f"{k}: {round(v[-1], 4)} | ", end="")
            print()

        return logger

    def train_epoch(self):
        run_loss_ep = 0.0
        run_preds_ep = []
        run_targets_ep = []

        # set model to training mode, since e.g. batchnorm layers have different behaviour
        self._model.train()
        for i, (x, y) in enumerate(self._trainloader):
            if self._cuda:
                x, y = x.cuda(), y.cuda()

            # zero gradients in optimizer
            self._optim.zero_grad()

            # process x, compute loss, backpropagate it and perform optimization step
            with torch.set_grad_enabled(True):
                y_hat = self._model(x)
                loss = self._crit(y_hat, y)
                loss.backward()
                self._optim.step()

            _, preds = torch.max(y_hat, 1)  # get class prediction from probabilities
            run_loss_ep += loss.item()  # .item() converts to default python datatype on cpu
            run_preds_ep.extend(preds.tolist())
            run_targets_ep.extend(y.tolist())

        return {
            "tr loss": run_loss_ep / len(self._trainloader),
            "tr acc.": self.compute_accuracy(run_preds_ep, run_targets_ep)
        }

    def validate_epoch(self):
        run_loss_ep = 0.0
        run_preds_ep = []
        run_targets_ep = []

        self._model.eval()
        for i, (x, y) in enumerate(self._valloader):
            if self._cuda:
                x, y = x.cuda(), y.cuda()

            self._optim.zero_grad()
            with torch.no_grad():
                y_hat = self._model(x)
                loss = self._crit(y_hat, y)

            _, preds = torch.max(y_hat, 1)
            run_loss_ep += loss.item()
            run_preds_ep.extend(preds.tolist())
            run_targets_ep.extend(y.tolist())

        return {
            "vl loss": run_loss_ep / len(self._valloader),
            "vl acc.": self.compute_accuracy(run_preds_ep, run_targets_ep)
        }

    def compute_accuracy(self, predictions, targets):
        correct = sum([predictions[i] == targets[i] for i in range(len(predictions))])
        return correct/len(predictions)

    def save_model(self, path):
        torch.save(self._model.state_dict(), path)
