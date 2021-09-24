import os
import glob
import urllib.request
import numpy as np
import argparse
import random


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def get_classes(args):

    with open(args.classes_file) as f:
        classes = f.readlines()

    idx = np.random.sample(len(classes), args.max_classes)
    classes = classes[idx]

    return classes


def download(args):
    base = 'https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/'

    classes = get_classes(args)

    for i, c in enumerate(classes):
        cls_url = c.replace('_', '%20')
        path = base + cls_url +'.npy'
        try:
            urllib.request.urlretrieve(path, 'data/ ' + c +'.npy')
            print(f"[{i + 1}/{args.max_classes}] Save {cls_url} class, path: {path}")
        except:
            print(f"[{i + 1}/{args.max_classes}] {cls_url} is not exist")


def convert_all_data(args):
    all_files = glob.glob(os.path.join(args.store_dir, "*.npy"))

    #initialize variables
    x = np.empty([0, 784])
    y = np.empty([0])
    class_names = []

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--store_dir", type=str, default="data/")
    parser.add_argument("--dataset_file", type=str, default="data/",
                        help="")
    parser.add_argument("--classes_file", type=str, default="categories.txt",
                        help="")
    parser.add_argument("--max_classes", type=int, default=100,
                        help="maximum number of classes")
    parser.add_argument("--max_data_per_class", type=int, default=5000,
                        help="maximum number of data per class")
    parser.add_argument("--seed", type=int, default=2021)

    args = parser.parse_args()

    # Set seed
    set_seed(args.seed)

    # if npy files is not exist, download files
    dir_empty = False
    if not os.path.exists(args.store_dir):
        os.makedirs(args.store_dir)
        dir_empty = True
    else: # if directory is empty
        if not os.listdir(args.store_dir):
            dir_empty = True

    if dir_empty:
        download(args)






