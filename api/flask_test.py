import requests
import json
from PIL import Image
import numpy as np

headers = {'Content-Type':'application/json'}

image = Image.open('../datasets/data/few_shot_test/query/axe0.png').convert("RGB")
pixels = np.array(image)
data = {"images": pixels.tolist()}
# data["images"] => (height, witdh, channel)

resp = requests.post("http://localhost:5000/predict",
                     data=json.dumps(data),
                     headers=headers)

print(resp.json())