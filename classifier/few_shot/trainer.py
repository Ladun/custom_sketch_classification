
import tqdm
import os

import torch


class Trainer(object):
    def __init__(self,
                 classifier,
                 optimizer,
                 scheduler,
                 loss,
                 device,
                 progress_print=0,
                 save_checkpoint_path=None,
                 load_checkpoint=None):

        self.device = device
        self.classifier = classifier.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss = loss
        self.progress_print = progress_print

        self.checkpoint_path = save_checkpoint_path
        head, tail = os.path.split(save_checkpoint_path)
        _, ext = os.path.splitext(tail)
        self.best_path = f"{head}/best{ext}"

        self.logs = {"acc": [0.0], "loss": [0.0],
                     "eval_acc":[0.0]}

        self.epoch = 0
        self.num_steps = 0
        self.best_acc = 0

        if load_checkpoint:
            self._load_checkpoint(load_checkpoint)

    def _train_model(self, train_loader):
        self.classifier.train()

        total = 0
        correct = 0
        for batch_index, (batch) in enumerate(tqdm.tqdm(train_loader)):
            if self.progress_print != 0 and self.num_steps % self.progress_print == 0:
                self._print_progress()

            support_images = batch[0].to(self.device)
            support_labels = batch[1].to(self.device)
            query_images = batch[2].to(self.device)
            query_labels = batch[3].to(self.device)

            scores = self.classifier(support_images, support_labels, query_images)

            loss = self.loss(scores, query_labels)

            _, logits_index = torch.max(scores, 1)
            total += query_labels.size(0)
            correct += (logits_index == query_labels).float().sum()

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

        total = 0
        correct = 0

        for batch_index, (batch) in enumerate(tqdm.tqdm(test_loader)):

            support_images = batch[0].to(self.device)
            support_labels = batch[1].to(self.device)
            query_images = batch[2].to(self.device)
            query_labels = batch[3].to(self.device)

            scores = self.classifier(support_images, support_labels, query_images)

            _, logits_index = torch.max(scores, 1)
            total += query_labels.size(0)
            correct += (logits_index == query_labels).float().sum()

        eval_acc = 100 * correct / total
        if eval_acc > self.best_acc:
            self._save_checkpoint(self.best_path)

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
            self._save_checkpoint(self.checkpoint_path)

            self.scheduler.step()

    def _print_progress(self):
        print("Accuracy: {}".format(self.logs["acc"][-1]))
        print("Loss: {}".format(self.logs["loss"][-1]))

    def _save_checkpoint(self, path):
        if path is None:
            return
        checkpoint = {
            "epoch": self.epoch,
            "num_steps": self.num_steps,
            "best_acc": self.best_acc,
            "classifier": self.classifier.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict()
        }
        torch.save(checkpoint, path)

    def _load_checkpoint(self, checkpoint_path):
        print("=> loading checkpoint '{}'".format(checkpoint_path))

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.epoch = checkpoint["epoch"]
        self.num_steps = checkpoint["num_steps"]
        self.best_acc = checkpoint["best_acc"]

        self.classifier.load_state_dict(checkpoint["classifier"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.scheduler.load_state_dict(checkpoint["scheduler"])

        print("=> loaded checkpoint '{}' (epoch {})"
              .format(checkpoint_path, checkpoint['epoch']))
