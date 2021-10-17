from tqdm import tqdm
import os
import shutil
import random

import argparse


def main(args):
    # unzip tu-berlin sketch datasets file into 'tu_berlin/data/'
    train_dir = args.train
    val_dir = args.val
    path_list = []
    with open(os.path.join(train_dir, "filelist.txt"), 'r') as f:
        path_list = f.readlines()

    dic = {}
    count = 0
    # Get All image data path
    for path_name in tqdm(path_list):
        path_name = path_name.replace('\n', '')
        file_dir, file_name = path_name.split('/')
        count += 1
        if file_dir in dic:
            dic[file_dir].append(file_name)
        else:
            dic[file_dir] = [file_name]

    # split train and validation
    if args.split_by_class:
        keys = dic.keys()
        keys = random.sample(keys, args.val_size_for_class)

        for key in keys:
            src_dir = os.path.join(train_dir, key)

            shutil.move(src_dir, val_dir)
    else:
        val_item_per_class = args.val_item_per_class
        for key in dic:
            images = dic[key]
            print(f"{key}: {len(images)}")

            src_dir = os.path.join(train_dir, key)
            tar_dir = os.path.join(val_dir, key)
            if not os.path.exists(tar_dir):
                os.makedirs(tar_dir)

            idxs = random.sample(range(len(images)), val_item_per_class)
            for idx in idxs:
                shutil.move(os.path.join(src_dir, images[idx]), os.path.join(tar_dir, images[idx]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, default="datasets/tu_berlin/data/train")
    parser.add_argument("--val", type=str, default="datasets/tu_berlin/data/val")
    parser.add_argument("--val_item_per_class", type=int, default=15)
    parser.add_argument("--val_size_for_class", type=int, default=15)
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--split_by_class", action="store_true",
                        help="if TRUE, split data by class\n else, split data each class")

    args = parser.parse_args()

    random.seed(args.seed)
    main(args)

