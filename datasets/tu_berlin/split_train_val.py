from tqdm import tqdm
import os
import shutil

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

    val_item_per_class = args.val_size
    # split train and validation
    for key in dic:
        images = dic[key]
        print(f"{key}: {len(images)}")

        src_dir = os.path.join(train_dir, key)
        tar_dir = os.path.join(val_dir, key)
        if not os.path.exists(tar_dir):
            os.makedirs(tar_dir)

        idx = 0
        for _ in range(val_item_per_class):
            shutil.move(os.path.join(src_dir, images[idx]), os.path.join(tar_dir, images[idx]))
            idx += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, default="datasets/tu_berlin/data/train")
    parser.add_argument("--val", type=str, default="datasets/tu_berlin/data/val")
    parser.add_argument("--val_size", type=int, default=15)

    main()

