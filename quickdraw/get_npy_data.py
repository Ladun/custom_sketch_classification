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

    idx = np.random.choice(len(classes), args.max_classes, replace=False)
    ret = []
    for i in idx:
        ret.append(classes[i][:-1])

    return ret


def download(args):
    print("<download> --------------")

    base = 'https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/'

    classes = get_classes(args)

    for i, c in enumerate(classes):
        cls_url = c.replace(' ', '%20').replace('_', '%20')
        path = base + cls_url + '.npy'
        try:
            urllib.request.urlretrieve(path, os.path.join(args.store_dir, c + '.npy'))
            print("[{0:>3}/{1:>3}] Save '{2:>20}' class, path: {3}".format(i + 1, args.max_classes, cls_url, path))
        except Exception as e:
            print("[{0:>3}/{1:>3}] {2:>20} download failed".format(i + 1, args.max_classes, cls_url))
            print(e)


def convert_all_data(args):
    print("<convert_all_data> --------------")
    all_files = glob.glob(os.path.join(args.store_dir, "*.npy"))

    sqr_img_size = args.image_size * args.image_size * args.image_channel

    #initialize variables
    x = np.empty([0, args.max_data_per_class, sqr_img_size])
    class_names = []

    for idx, file in enumerate(all_files):
        class_name, ext = os.path.splitext(os.path.basename(file))
        class_names.append(class_name)

        try:
            data = np.load(file)
        except Exception as e:
            print(f"load failed {class_name}")
            print(e)
        else:
            if data.shape[0] < args.max_data_per_class:
                print(f"class_names data is not enough, size: {data.shape}")
            else:
                data = data[0: args.max_data_per_class, :]

                data = data.reshape((1, args.max_data_per_class, sqr_img_size))

                x = np.concatenate((x, data), axis=0)
                print(f"[{idx + 1}/{len(all_files)}]add date {class_name}")

    data = None

    x = x.reshape((-1, args.max_data_per_class, args.image_size, args.image_size, args.image_channel))
    x = x.astype(np.float32)

    print(f"[convert_all_data] data size: {x.shape}")
    np.save(args.dataset_file, x)
    print(f"[convert_all_data] success to save, path: {args.dataset_file}.npy")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--store_dir", type=str, default="quickdraw/data/")
    parser.add_argument("--dataset_file", type=str, default="dagan/datasets/quick_dataset",
                        help="")
    parser.add_argument("--classes_file", type=str, default="quickdraw/categories.txt",
                        help="")
    parser.add_argument("--max_classes", type=int, default=345,
                        help="maximum number of classes")
    parser.add_argument("--max_data_per_class", type=int, default=1000,
                        help="maximum number of data per class")
    parser.add_argument("--download_specific", type=str, nargs='?',
                        help="")
    parser.add_argument("--image_size", type=int, default=28)
    parser.add_argument("--image_channel", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2021)

    args = parser.parse_args()

    print('<Parsed arguments>')
    for k, v in vars(args).items():
        print('{}: {}'.format(k, v))
    print('')

    # Set seed
    set_seed(args.seed)

    # if npy files is not exist, download files
    dir_empty = False
    if not os.path.exists(args.store_dir):
        os.makedirs(args.store_dir)
        dir_empty = True
    else:
        if not os.listdir(args.store_dir): # if directory is empty
            dir_empty = True
    if dir_empty:
        download(args)

    convert_all_data(args)


if __name__ == "__main__":
    main()

