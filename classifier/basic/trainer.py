
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

    def _accuracy(self, logits, label, topk=(1,)):
        maxk = max(topk)
        batch_size = label.size(0)
        _, pred = logits.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(label.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

    def _train_model(self, train_loader):
        self.classifier.train()

        for batch_index, (batch, label) in enumerate(tqdm.tqdm(train_loader)):
            if self.progress_print != 0 and self.num_steps % self.progress_print == 0:
                self._print_progress()

            batch, label = batch.to(self.device), label.to(self.device)

            logits = self.classifier(batch)

            loss = self.loss(logits, label)

            prec1, prec5 = self._accuracy(logits, label, topk=(1, 5))
            self.logs["prec1"].append(prec1)
            self.logs["prec5"].append(prec5)
            self.logs["loss"].append(loss.item())

            self.optimizer.zero_grad()
            loss.backward()

            self.optimizer.step()

            self.num_steps += 1

    def _eval_model(self, test_loader):
        self.classifier.eval()

        eval_prec1 = 0
        eval_prec5 = 0

        for batch_index, (batch, label) in enumerate(tqdm.tqdm(test_loader)):
            if self.progress_print != 0 and self.num_steps % self.progress_print == 0:
                self._print_progress()

            batch, label = batch.to(self.device), label.to(self.device)
            logits = self.classifier(batch)

            prec1, prec5 = self._accuracy(logits, label, topk=(1, 5))
            eval_prec1 += prec1
            eval_prec5 += prec5

        eval_prec1 = eval_prec1 / len(test_loader)
        eval_prec5 = eval_prec5 / len(test_loader)

        self.logs["eval_prec1"].append(eval_prec1)
        self.logs["eval_prec5"].append(eval_prec5)

        print("Prec@1: {}".format(self.logs["prec1"][-1]))
        print("Prec@5: {}".format(self.logs["prec5"][-1]))

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
        print("Prec@1: {}".format(self.logs["prec1"][-1]))
        print("Prec@5: {}".format(self.logs["prec5"][-1]))
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
