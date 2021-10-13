
import tqdm

import torch


class Trainer(object):
    def __init__(self,
                 classifier,
                 optimizer,
                 loss,
                 device,
                 progress_print=0,
                 save_checkpoint_path=None,
                 load_checkpoint=None):

        self.device = device
        self.classifier = classifier.to(device)
        self.optimizer = optimizer
        self.loss = loss
        self.progress_print = progress_print

        self.checkpoint_path = save_checkpoint_path
        self.logs = {"prec1": [0.0], "prec5": [0.0], "loss": [0.0],
                     "eval_prec1":[0.0], "eval_prec5":[0.0]}

        self.epoch = 0
        self.num_steps = 0

        if load_checkpoint:
            self._load_checkpoint(load_checkpoint)


    def _train_model(self, train_loader):
        self.classifier.train()

        total = 0
        correct = 0
        for batch_index, (batch, label) in enumerate(tqdm.tqdm(train_loader)):
            if self.progress_print != 0 and self.num_steps % self.progress_print == 0:
                self._print_progress()

            batch, label = batch.to(self.device), label.to(self.device)

            logits = self.classifier(batch)

            loss = self.loss(logits, label)

            _, logits_index = torch.max(logits, 1)
            total += label.size(0)
            correct += (logits_index == label).float().sum()

            self.logs["loss"].append(loss.item())

            self.optimizer.zero_grad()
            loss.backward()

            self.optimizer.step()

            self.num_steps += 1

        acc = 100 * correct / total
        self.logs["acc"].append(acc)
        self._print_progress()

    def _eval_model(self, test_loader):
        self.classifier.eval()

        eval_acc = 0

        total = 0
        correct = 0

        for batch_index, (batch, label) in enumerate(tqdm.tqdm(test_loader)):

            batch, label = batch.to(self.device), label.to(self.device)
            logits = self.classifier(batch)

            _, logits_index = torch.max(logits, 1)
            total += label.size(0)
            correct += (logits_index == label).float().sum()

        eval_acc = 100 * correct / total

        self.logs["eval_acc"].append(eval_acc)

        print("Accuracy: {}".format(self.logs["eval_acc"][-1]))

    def train(self, train_loader, test_loader, epochs):

        for epoch in range(self.epoch, epochs):

            print(f"\n<Epoch: {epoch}> -------")

            print("<Train Step> ------")
            self._train_model(train_loader)

            print("<Evaluate Step> ------")
            self._eval_model(test_loader)

            print("<Save checkpoint> ------")
            self._save_checkpoint()

    def _print_progress(self):
        print("Accuracy: {}".format(self.logs["acc"][-1]))
        print("Loss: {}".format(self.logs["loss"][-1]))

    def _save_checkpoint(self):
        if self.checkpoint_path is None:
            return
        checkpoint = {
            "epoch": self.epoch,
            "num_steps": self.num_steps,
            "classifier": self.classifier.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        torch.save(checkpoint, self.checkpoint_path)

    def _load_checkpoint(self, checkpoint_path):
        print("=> loading checkpoint '{}'".format(checkpoint_path))

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.epoch = checkpoint["epoch"]
        self.num_steps = checkpoint["num_steps"]

        self.classifier.load_state_dict(checkpoint["classifier"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])

        print("=> loaded checkpoint '{}' (epoch {})"
              .format(checkpoint_path, checkpoint['epoch']))
