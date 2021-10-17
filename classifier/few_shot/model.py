
import torch
import torch.nn as nn


class PrototypicalNet(nn.Module):
    def __init__(self, backbone):
        super().__init__()

        self.backbone = backbone

    def forward(self, support_images, support_labels, query_images):

        # Extract the features of support and query images
        z_support = self.backbone(support_images)
        z_query = self.backbone(query_images)

        print(z_support.shape)
        print(z_query.shape)

        # Infer the number of different classes from the labels of the support set
        n_way = len(torch.unique(support_labels))
        # Prototype i is the mean of all instances of features corresponding to labels == i
        z_proto = torch.cat(
            [
                z_support[torch.nonzero(support_labels == label)].mean(0)
                for label in range(n_way)
            ]
        )
        print(z_proto.shape)

        # Compute the euclidean distance from queries to prototypes
        dists = torch.cdist(z_query, z_proto)

        # And here is the super complicated operation to transform those distances into classification scores!
        scores = -dists

        return scores
