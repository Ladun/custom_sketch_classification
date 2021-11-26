from tqdm import tqdm
import os
import shutil
import random

import argparse

import cv2

from utils import convert_line_to_dotted_line

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
    print("Make dictionary----")
    for path_name in tqdm(path_list):
        path_name = path_name.replace('\n', '')
        file_dir, file_name = path_name.split('/')
        count += 1
        if file_dir in dic:
            dic[file_dir].append(file_name)
        else:
            dic[file_dir] = [file_name]

    if args.convert_dot_line:
        print("Do convert_dot_line----")
        for file_dir in dic:
            images_name = dic[file_dir]
            length = len(images_name)
            convert_indexes = random.sample(range(length), length // 2)

            for idx in convert_indexes:
                path = os.path.join(train_dir, file_dir, images_name[idx])
                img = cv2.imread(path)
                img = convert_line_to_dotted_line(img)
                cv2.imwrite(path, img)

    # split train and validation
    print("Split train and validation----")
    if args.split_by_class:
        keys = random.sample(dic.keys(), args.val_size_for_class)

        for key in keys:
            src_dir = os.path.join(train_dir, key)
            tar_dir = os.path.join(val_dir, key)

            print(f"{src_dir} to {tar_dir}")
            shutil.move(src_dir, tar_dir)
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
            print(f"{idxs} to {tar_dir}")
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
    parser.add_argument("--convert_dot_line", action="store_true",
                        help="Change the line to a dotted line.")

    args = parser.parse_args()

    random.seed(args.seed)
    main(args)

