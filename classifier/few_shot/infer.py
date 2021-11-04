import tqdm

import argparse
import os
import numpy as np
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import matplotlib.pyplot as plt

from trainer import Trainer
from model import PrototypicalNet
from torchvision.models import resnet18
from datasets import TaskSampler, InferenceDataset

'''
python classifier/few_shot/infer.py \
        --infer \
        --infer_data datasets/data/few_shot \
        --load_checkpoint classifier/few_shot/checkpoint/best_few_shot.ckpt \
'''


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_way', type=int, default=5,
                        help='n_way')
    parser.add_argument('--n_shot', type=int, default=5,
                        help='n_shot')
    parser.add_argument('--n_query', type=int, default=10,
                        help='number of query')
    parser.add_argument('--use_cpu', action='store_true',
                        help='enables CPU training')
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed (default: 1)')
    parser.add_argument('--load_checkpoint', type=str,
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--test_data', default='', type=str,
                        help='test dataset directory path')
    parser.add_argument('--infer_data', default='', type=str,
                        help='infer dataset directory path')
    parser.add_argument('--infer', action='store_true',
                        help='do infer')

    args = parser.parse_args()
    args.device = torch.device("cuda" if not args.use_cpu and torch.cuda.is_available() else "cpu")

    set_seed(args.seed)

    print('<Parsed arguments>')
    for k, v in vars(args).items():
        print('{}: {}'.format(k, v))
    print('')

    kwargs = {'num_workers': 4, 'pin_memory': True} if not args.use_cpu and torch.cuda.is_available() else {}

    # --------- Model ---------
    backbone = resnet18(pretrained=True)
    backbone.fc = nn.Identity()
    model = PrototypicalNet(backbone)

    # --------- Load Model ----------
    checkpoint = torch.load(args.load_checkpoint, map_location=args.device)
    model.load_state_dict(checkpoint["classifier"])

    # --------- Train ---------
    if args.infer:
        infer(args, model)
    else:
        evaluate_with_loader(args, model, kwargs)


def evaluate_with_loader(args, model, kwargs):

    # --------- Dataset ---------
    test_dataset = datasets.ImageFolder(
        args.val_data,
        transform=transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor()
        ])
    )
    test_dataset.labels = test_dataset.targets
    test_sampler = TaskSampler(
        test_dataset,
        n_way=args.n_way, n_shot=args.n_shot, n_query=args.n_query,
        n_tasks=args.n_val_episode
    )
    test_loader = DataLoader(
        test_dataset,
        batch_sampler=test_sampler,
        collate_fn=test_sampler.episodic_collate_fn,
        **kwargs
    )

    model.eval()

    total = 0
    correct = 0

    for batch_index, (batch) in enumerate(tqdm.tqdm(test_loader)):
        support_images = batch[0].to(args.device)
        support_labels = batch[1].to(args.device)
        query_images = batch[2].to(args.device)
        query_labels = batch[3].to(args.device)

        scores = model(support_images, support_labels, query_images)

        _, logits_index = torch.max(scores, 1)
        total += query_labels.size(0)
        correct += (logits_index == query_labels).float().sum()

    eval_acc = 100 * correct / total

    print("Accuracy: {}".format(eval_acc))


def infer(args, model):
    model.eval()

    # -------------- Load Data -----------------
    infer_data = InferenceDataset('datasets/data/few_shot_test',
                                  transforms=transforms.Compose([
                                        transforms.Resize([224, 224]),
                                        transforms.Grayscale(num_output_channels=3),
                                        transforms.ToTensor()
                                    ]),
                                  n_way=3,
                                  n_shot=5)
    data_loader = DataLoader(infer_data, batch_size=1)

    np.set_printoptions(precision=6, suppress=True)
    for batch_index, (batch) in enumerate(data_loader):
        support_images = batch[0].to(args.device).squeeze(0)
        support_labels = batch[1].to(args.device).squeeze(0)
        query_images = batch[2].to(args.device)
        true_class = batch[3]
        query_name = batch[4]

        scores = model(support_images, support_labels, query_images)

        _, logits_index = torch.max(scores, 1)

        softmax = nn.Softmax(dim=-1)(scores)

        print(f"[{batch_index}] query image {query_name}: {true_class[logits_index.item()]}, {softmax.detach().numpy() * 100}")


if __name__ == "__main__":
    main()
