
import torch.nn as nn
import torchvision.models


class FeatureExtractor(nn.Module):
    def __init__(self, feature_network):
        super(FeatureExtractor, self).__init__()

        assert feature_network in ['vgg', 'resnet-101']

        if feature_network == 'resnet-101':
            model = torchvision.models.resnet101(pretrained=True)
            conv1 = nn.Conv2d(1, 64, (7, 7), (2, 2), (3, 3), bias=False)
            layers = [conv1, model.bn1, model.relu, model.maxpool, model.layer1, model.layer2]
            self.model = nn.Sequential(*layers)

        elif feature_network == 'vgg':
            vgg = torchvision.models.vgg19_bn(pretrained=True)
            self.model = vgg.features[:37]

        # FeatureExtractor should not be trained
        for child in self.model.children():
            for param in child.parameters():
                param.requires_grad = False

    def forward(self, x):
        x = self.model(x)
        return x