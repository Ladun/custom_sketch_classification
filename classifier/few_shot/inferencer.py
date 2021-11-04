import os
import numpy as np
import random
from PIL import Image

import torch
import torch.nn as nn

from classifier.few_shot.model import PrototypicalNet
from torchvision.models import resnet18
from torchvision import transforms


class Inferencer(object):

    def __init__(self,
                 model_ckpt,
                 device,
                 support_dir,
                 transforms,
                 n_shot):
        '''
        :param support_dir:
            path of inference datasets,
            need to follow the directory structure below
            support_dir
                - class1
                    - 'image001...'
                    - ...
                - class2
                    - 'image00n...'
                    - ...
                - class3
                    - 'image00m...'
                    - ...
        '''

        self.device = device

        # --------- Model ---------
        backbone = resnet18(pretrained=True)
        backbone.fc = nn.Identity()
        self.model = PrototypicalNet(backbone)

        # --------- Load Model ----------
        checkpoint = torch.load(model_ckpt, map_location=self.device)
        self.model.load_state_dict(checkpoint["classifier"])

        self.transforms = transforms
        self.n_shot = n_shot

        # support images dictionary
        self._support_images = {}
        self._support_keys = os.listdir(support_dir)
        for key in self._support_keys:
            path = os.path.join(support_dir, key)

            file_list = os.listdir(path)
            for file in file_list:
                image = Image.open(os.path.join(path, file)).convert("RGB")
                image = np.array(image)

                if key in self._support_images.keys():
                    self._support_images[key].append(image)
                else:
                    self._support_images[key] = [image]

        # real uses support images
        self._update_support()

    def _update_support(self):
        _support_images = [
            (self._support_images[label][:self.n_shot], label)
            for label in self._support_keys
        ]

        self.support_images = []
        self.support_labels = []
        self.true_class = {}
        idx = 0
        for image_list , label in _support_images:
            self.true_class[idx] = label
            for image in image_list:
                image = self.transforms(image)
                self.support_images.append(image.unsqueeze(0))
                self.support_labels.append(idx)
            idx += 1
        self.support_images = torch.cat(self.support_images, dim=0)
        self.support_labels = torch.tensor(self.support_labels)

    def inference(self, query_images):

        query_images = query_images.to(self.device)
        if len(query_images.size()) == 3:
            query_images = query_images.unsqueeze(0)

        scores = self.model.forward(self.support_images, self.support_labels, query_images)
        _, logits_index = torch.max(scores, 1)

        softmax = nn.Softmax(dim=-1)(scores)

        return logits_index.item(), self.true_class[logits_index.item()], softmax.detach().numpy()[0]


if __name__ == "__main__":
    infer_transforms = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Resize([224, 224]),
                    transforms.Grayscale(num_output_channels=3)
                ])
    query = Image.open("../../datasets/data/few_shot_test/query/axe0.png").convert("RGB")

    inferencer = Inferencer(model_ckpt="checkpoint/best_few_shot.ckpt",
                            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                            support_dir="../../datasets/data/few_shot_test/support",
                            transforms=infer_transforms,
                            n_shot=5)

    query = infer_transforms(np.array(query))

    class_id, class_name, softmax = inferencer.inference(query)

    print(f"{class_id}, {class_name}, {softmax}")
