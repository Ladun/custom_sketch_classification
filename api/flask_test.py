import requests
import json
from PIL import Image
import numpy as np

import matplotlib.pyplot as plt

headers = {'Content-Type':'application/json'}

image = Image.open('../datasets/data/few_shot_test/query/gun1.png').convert("RGB")
pixels = np.array(image)
data = {"images": pixels.tolist(), "is_point": False}
# data = {"images": [1, 1, 2, 2], "is_point": True}
# data["images"] => (height, witdh, channel)

resp = requests.post("http://localhost:5000/predict",
                     data=json.dumps(data),
                     headers=headers)

print(resp.json())