
from flask import Flask, jsonify, request

import numpy as np
import torch
from torchvision import transforms
import cv2.cv2 as cv2

import matplotlib.pyplot as plt

from classifier.few_shot.inferencer import Inferencer

'''
run command:
    python -m api.flask_server
'''

infer_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize([224, 224]),
    transforms.Grayscale(num_output_channels=3)
])

inferencer = Inferencer(model_ckpt="classifier/few_shot/checkpoint/best_few_shot_50_dotted.ckpt",
                        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                        support_dir="datasets/data/few_shot_test/support",
                        transforms=infer_transforms,
                        n_shot=5,
                        seed=2021)
app = Flask(__name__)


@app.route('/')
def hello():
    return "Image Classification Sample"


def convert_to_image(data):
    test_size = 1111

    image_points = 1 - np.array(data['images'])
    image_points = image_points.reshape(image_points.shape[0] // 2, 2)
    image_points *= test_size
    image_points = image_points.astype(np.uint32)

    images = np.ones((test_size, test_size)) * 255
    for point in image_points:
        images = cv2.circle(images, (point[0], point[1]), 5, (0, 0, 0), cv2.FILLED)
    images[image_points[:, 1], image_points[:, 0]] = 0
    images = images.reshape((test_size, test_size, -1)).repeat(repeats=3, axis=-1)
    return images.astype(np.uint8)


@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    if data["is_point"]:
        query_images = convert_to_image(data)

        # cv2.imwrite("test1.png", query_images)
        # plt.imshow(query_images), plt.show()
    else:
        query_images = np.array(data['images'], dtype=np.uint8)
    query_images = infer_transforms(query_images)

    class_id, class_name, softmax = inferencer.inference(query_images)

    print(f"Result: {({'class_id': class_id, 'class_name': class_name,'softmax': np.round(softmax.tolist(), 4)})}")
    return jsonify({'class_id': class_id, 'class_name': class_name,'softmax': softmax.tolist()})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, threaded=False)
