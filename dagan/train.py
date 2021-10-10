
import os
import random
import argparse

import numpy as np
import torch
import torch.optim as optim
import torchvision.transforms as transforms

from dagan_trainer import DaganTrainer
from discriminator import Discriminator
from feature_extractor import FeatureExtractor
from generator import Generator
from dataset import create_dagan_dataloader


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


def get_dagan_args():
    parser = argparse.ArgumentParser(
        description="Use this script to train a dagan.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        help="Filepath for dataset on which to train dagan. File should be .npy format with shape "
        "(num_classes, samples_per_class, height, width, channels).",
    )
    parser.add_argument(
        "--final_model_path", type=str, help="Filepath to save final dagan model."
    )
    parser.add_argument(
        "--batch_size",
        nargs="?",
        type=int,
        default=32,
        help="batch_size for experiment",
    )
    parser.add_argument(
        "--img_size",
        nargs="?",
        type=int,
        help="Dimension to scale images when training. "
        "Useful when model architecture expects specific input size. "
        "If not specified, uses img_size of data as passed.",
    )
    parser.add_argument(
        "--num_training_classes",
        nargs="?",
        type=int,
        default=300,
        help="Number of classes to use for training.",
    )
    parser.add_argument(
        "--num_val_classes",
        nargs="?",
        type=int,
        default=45,
        help="Number of classes to use for validation.",
    )
    parser.add_argument(
        "--epochs",
        nargs="?",
        type=int,
        default=50,
        help="Number of epochs to run training.",
    )
    parser.add_argument(
        "--init_epochs",
        nargs="?",
        type=int,
        default=15,
        help="Number of epochs to run content training.",
    )
    parser.add_argument(
        "--save_checkpoint_path",
        nargs="?",
        type=str,
        help="Filepath to save intermediate training checkpoints.",
    )
    parser.add_argument(
        "--load_checkpoint_path",
        nargs="?",
        type=str,
        help="Filepath of intermediate checkpoint from which to resume training.",
    )
    parser.add_argument(
        "--dropout_rate",
        type=float,
        nargs="?",
        default=0.5,
        help="Dropout rate to use within network architecture.",
    )
    parser.add_argument(
        "--content_loss_weight",
        type=float,
        nargs="?",
        default=10,
        help="content_loss_weight",
    )
    parser.add_argument(
        "--suppress_generations",
        action="store_true",
        help="If specified, does not show intermediate progress images.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2021,
        help="random seed",
    )
    parser.add_argument(
        "--use_cpu",
        action="store_true",
        help="random seed",
    )
    return parser.parse_args()


def main():
    args = get_dagan_args()

    print('<Parsed arguments>')
    for k, v in vars(args).items():
        print('{}: {}'.format(k, v))
    print('')

    set_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() and not args.use_cpu else "cpu")

    # Load dataset
    raw_data = np.load(args.dataset_path).copy()
    in_channels = raw_data.shape[-1]
    img_size = args.img_size or raw_data.shape[2]

    # Exception check
    final_generator_dir = os.path.dirname(args.final_model_path) or os.getcwd()
    if not os.access(final_generator_dir, os.W_OK):
        raise ValueError(args.final_generator_dir + " is not a valid filepath.")

    if args.num_training_classes + args.num_val_classes > raw_data.shape[0]:
        raise ValueError(
            "Expected at least %d classes but only had %d."
            % (args.num_training_classes + args.num_val_classes, raw_data.shape[0])
        )

    # ----------------------- About dataset -----------------------
    # Define transform
    mid_pixel_value = np.max(raw_data) / 2
    train_transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(
                (mid_pixel_value,) * in_channels, (mid_pixel_value,) * in_channels
            ),
        ]
    )

    # Create dataloader
    train_dataloader = create_dagan_dataloader(
        raw_data, args.num_training_classes, train_transform, args.batch_size
    )
    display_transform = train_transform

    # Get validation dataset
    val_data = raw_data[args.num_training_classes: args.num_training_classes + args.num_val_classes]
    flat_val_data = val_data.reshape(
        (val_data.shape[0] * val_data.shape[1], *val_data.shape[2:])
    )

    # ----------------------- About Model -----------------------
    # Define generator and discriminator
    g = Generator(dim=img_size, channels=in_channels, dropout_rate=args.dropout_rate)
    d = Discriminator(dim=img_size, channels=in_channels * 2, dropout_rate=args.dropout_rate)
    feature_extractor = FeatureExtractor(feature_network='resnet-101')

    g_opt = optim.AdamW(g.parameters(), lr=0.0001, betas=(0.0, 0.9))
    d_opt = optim.AdamW(d.parameters(), lr=0.0001, betas=(0.0, 0.9))

    # ----------------------- About Train -----------------------
    trainer = DaganTrainer(
        generator=g,
        discriminator=d,
        feature_extractor=feature_extractor,
        gen_optimizer=g_opt,
        dis_optimizer=d_opt,
        batch_size=args.batch_size,
        device=device,
        critic_iterations=5,
        content_loss_weight=args.content_loss_weight,
        print_every=75,
        num_tracking_images=10,
        save_checkpoint_path=args.save_checkpoint_path,
        load_checkpoint_path=args.load_checkpoint_path,
        display_transform=display_transform,
        should_display_generations=not args.suppress_generations,
    )

    # Do train
    trainer.train(data_loader=train_dataloader, epochs=args.epochs, init_epochs=args.init_epochs, val_images=flat_val_data)

    # Save final generator model
    torch.save(trainer.g, args.final_model_path)


if __name__ == "__main__":
    main()
