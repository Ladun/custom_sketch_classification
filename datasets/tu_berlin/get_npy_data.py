import os
import tqdm
import numpy as np

from PIL import Image

import matplotlib.pyplot as plt


def main():
    # unzip tu-berlin sketch datasets file into 'tu_berlin/data/'
    base_dir = "tu_berlin/data/train"
    path_list = []
    with open(os.path.join(base_dir, "filelist.txt"), 'r') as f:
        path_list = f.readlines()

    dic = {}
    count = 0
    for path_name in tqdm.tqdm(path_list):
        path_name = path_name.replace('\n', '')
        file_dir, file_name = path_name.split('/')

        img = Image.open(os.path.join(base_dir, path_name))
        img = img.resize((128, 128))
        img = np.asarray(img)
        img = img[np.newaxis, :, :, np.newaxis]
        count += 1
        if file_dir in dic:
            dic[file_dir].append(img)
        else:
            dic[file_dir] = [img]

    total_file = []
    for k in dic:
        dic[k] = np.concatenate(dic[k], axis =0)
        print(f"{k}: {dic[k].shape}")
        total_file.append(dic[k][np.newaxis, :])
    total_file = np.concatenate(total_file, axis=0)
    total_file = total_file.astype(np.float32)
    print(f"Total category: {len(dic)}")

    np.save("tu_berlin/data/tu_berlin_dataset.npy", total_file)
    print("Finish save file to tu_berlin/data/tu_berlin_dataset.npy")


if __name__ == "__main__":
    main()