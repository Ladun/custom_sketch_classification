
import numpy as np
import cv2


def convert_line_to_dotted_line(img, scale=5):
    scale = 5
    noise = np.random.randint(0, 256, (img.shape[0] // scale, img.shape[1] // scale), dtype=np.uint8)
    noise = cv2.resize(noise, dsize=(img.shape[:2]), interpolation=cv2.INTER_NEAREST)
    noise = np.repeat(noise.reshape(noise.shape + (1,)), repeats=3, axis=-1)
    noise = (noise > 128)

    img[noise] = 255

    return img