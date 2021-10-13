
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

from trainer import Trainer
from model import SketchANet, ResNetBase
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
    parser.add_argument('--batch_size', type=int, default=64,
                        help='input batch size for training')
    parser.add_argument('--test-batch-size', type=int, default=64, metavar='N',
                        help='input batch size for testing (default: 10)')
    parser.add_argument('--epochs', type=int, default=2000,
                        help='input batch size for training')
    parser.add_argument('--learning_rate', type=float, default=2e-4,
                        help='learning rate (default: 2e-4)')
    parser.add_argument('--weight_decay', type=float, default=0.0005,
                        help='Adm weight decay (default: 0.0005)')
    parser.add_argument('--use_cpu', action='store_true',
                        help='enables CPU training')
    parser.add_argument('--classes', type=int, default=250,
                        help='number of classes (default: 250)')
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
            transforms.Resize([256, 256]),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(45),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])
    )
    test_dataset = datasets.ImageFolder(
        args.val_data,
        transform=transforms.Compose([
            transforms.Resize([256, 256]),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            #transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                # std=[0.229, 0.224, 0.225])
        ])
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.test_batch_size, shuffle=True, **kwargs
    )
    # --------- Model ---------
    # model = SketchANet(num_classes=args.classes)
    model = ResNetBase(num_classes=args.classes)

    # --------- Train ---------
    loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.99), weight_decay=args.weight_decay)

    trainer = Trainer(classifier=model,
                      optimizer=optimizer,
                      loss=loss,
                      device=args.device,
                      save_checkpoint_path=args.save_checkpoint,
                      load_checkpoint=args.load_checkpoint)

    trainer.train(train_loader, test_loader, args.epochs)


if __name__ == "__main__":
    main()
