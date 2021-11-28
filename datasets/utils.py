
import argparse
import os

import numpy as np
import cv2



def convert_line_to_dotted_line(img, scale=5):
    scale = 5
    noise = np.random.randint(0, 256, (img.shape[0] // scale, img.shape[1] // scale), dtype=np.uint8)
    noise = cv2.resize(noise, dsize=(img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    noise = np.repeat(noise.reshape(noise.shape + (1,)), repeats=3, axis=-1)
    noise = (noise > 128)

    img[noise] = 255

    return img


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str)
    parser.add_argument("--output_path", type=str,default="")

    args = parser.parse_args()

    if not os.path.isdir(args.image_path):
        output_path = args.output_path
        if not output_path:
            n, e = os.path.splitext(args.image_path)
            output_path = n + "_out" + e

        print(f"Image path is {args.image_path}, Output path is {output_path}")

        img = cv2.imread(args.image_path)
        img = convert_line_to_dotted_line(img)
        cv2.imwrite(output_path, img)
    else:

        for (root, _, files) in os.walk(args.image_path):
            if len(files) > 0:
                print(f"In [{root}] ==========")
                for file in files:
                    if file.endswith(".png"):
                        n, e = os.path.splitext(file)
                        output_path = n + "_out" + e
                        if not n.endswith("_out"):
                            print(f"[{file}] => [{output_path}]")

                            img = cv2.imread(os.path.join(root, file))
                            img = convert_line_to_dotted_line(img)
                            cv2.imwrite(os.path.join(root, output_path), img)






