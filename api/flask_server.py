
from flask import Flask, jsonify, request

import numpy as np
import torch
from torchvision import transforms

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

inferencer = Inferencer(model_ckpt="classifier/few_shot/checkpoint/best_few_shot.ckpt",
                        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                        support_dir="datasets/data/few_shot_test/support",
                        transforms=infer_transforms,
                        n_shot=5)
app = Flask(__name__)


@app.route('/')
def hello():
    return "Image Classification Sample"


@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    query_images = np.array(data['images'], dtype=np.uint8)
    query_images = infer_transforms(query_images)

    class_id, class_name, softmax = inferencer.inference(query_images)

    return jsonify({'class_id': class_id, 'class_name': class_name, "softmax": softmax.tolist()})


if __name__ == "__main__":

    app.run(threaded=False)