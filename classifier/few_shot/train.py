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
from datasets import TaskSampler

'''
python classifier/basic/train.py \
        --train_data datasets/tu_berlin/data/train \
        --val_data datasets/tu_berlin/data/val \
        --save_checkpoint classifier/basic/checkpoint/test.ckpt \

'''


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10,
                        help='input batch size for training')
    parser.add_argument('--n_episode', type=int, default=2000,
                        help='number of episode for training')
    parser.add_argument('--n_val_episode', type=int, default=100,
                        help='number of episode for testing')
    parser.add_argument('--n_way', type=int, default=5,
                        help='n_way')
    parser.add_argument('--n_shot', type=int, default=5,
                        help='n_shot')
    parser.add_argument('--n_query', type=int, default=10,
                        help='number of query')
    parser.add_argument('--learning_rate', type=float, default=2e-4,
                        help='learning rate (default: 2e-4)')
    parser.add_argument('--use_cpu', action='store_true',
                        help='enables CPU training')
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed (default: 1)')
    parser.add_argument('--save_checkpoint', type=str,
                        help='path to save checkpoint ')
    parser.add_argument('--load_checkpoint', default='', type=str,
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--train_data', type=str,
                        help='train dataset directory path')
    parser.add_argument('--val_data', type=str,
                        help='val dataset directory path')
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd', 'adamw'])
    args = parser.parse_args()
    args.device = torch.device("cuda" if not args.use_cpu and torch.cuda.is_available() else "cpu")

    print('<Parsed arguments>')
    for k, v in vars(args).items():
        print('{}: {}'.format(k, v))
    print('')

    kwargs = {'num_workers': 4, 'pin_memory': True} if not args.use_cpu and torch.cuda.is_available() else {}

    # --------- Dataset ---------
    train_dataset = datasets.ImageFolder(
        args.train_data,
        transform=transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.Grayscale(num_output_channels=3),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])
    )
    train_dataset.labels = train_dataset.targets
    train_sampler = TaskSampler(
        train_dataset,
        n_way=args.n_way, n_shot=args.n_shot, n_query=args.n_query,
        n_tasks=args.n_episode
    )
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        collate_fn=train_sampler.episodic_collate_fn,
        **kwargs
    )

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

    # --------- Model ---------
    backbone = resnet18(pretrained=True)
    backbone.fc = nn.Identity()
    model = PrototypicalNet(backbone)

    # --------- Train ---------
    loss = nn.CrossEntropyLoss()
    if args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    elif args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate)
    else:
        optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    trainer = Trainer(classifier=model,
                      optimizer=optimizer,
                      scheduler=scheduler,
                      loss=loss,
                      device=args.device,
                      save_checkpoint_path=args.save_checkpoint,
                      load_checkpoint=args.load_checkpoint)

    trainer.train(train_loader, test_loader, args.epochs)


if __name__ == "__main__":
    main()
