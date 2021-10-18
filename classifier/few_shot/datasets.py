import os
import random
from typing import List, Tuple
from PIL import Image

import torch
from torch.utils.data import Sampler, Dataset


class TaskSampler(Sampler):
    """
    Samples batches in the shape of few-shot classification tasks. At each iteration, it will sample
    n_way classes, and then sample support and query images from these classes.
    """

    def __init__(
        self, dataset: Dataset, n_way: int, n_shot: int, n_query: int, n_tasks: int
    ):
        """
        Args:
            dataset: dataset from which to sample classification tasks. Must have a field 'label': a
                list of length len(dataset) containing containing the labels of all images.
            n_way: number of classes in one task
            n_shot: number of support images for each class in one task
            n_query: number of query images for each class in one task
            n_tasks: number of tasks to sample
        """
        super().__init__(data_source=None)
        self.n_way = n_way
        self.n_shot = n_shot
        self.n_query = n_query
        self.n_tasks = n_tasks

        self.items_per_label = {}
        # labels means integer label to each images
        # labels[0] -> 0 index images integer label
        assert hasattr(
            dataset, "labels"
        ), "TaskSampler needs a dataset with a field 'label' containing the labels of all images."
        for item, label in enumerate(dataset.labels):
            if label in self.items_per_label.keys():
                self.items_per_label[label].append(item)
            else:
                self.items_per_label[label] = [item]

    def __len__(self):
        return self.n_tasks

    def __iter__(self):
        for _ in range(self.n_tasks):
            yield torch.cat(
                [
                    # pylint: disable=not-callable
                    torch.tensor(
                        random.sample(
                            self.items_per_label[label], self.n_shot + self.n_query
                        )
                    )
                    # pylint: enable=not-callable
                    for label in random.sample(self.items_per_label.keys(), self.n_way)
                ]
            )

    def episodic_collate_fn(
        self, input_data: List[Tuple[torch.Tensor, int]]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[int]]:
        """
        Collate function to be used as argument for the collate_fn parameter of episodic
            data loaders.
        Args:
            input_data: each element is a tuple containing:
                - an image as a torch Tensor
                - the label of this image
        Returns:
            tuple(Tensor, Tensor, Tensor, Tensor, list[int]): respectively:
                - support images,
                - their labels,
                - query images,
                - their labels,
                - the dataset class ids of the class sampled in the episode
        """

        true_class_ids = list({x[1] for x in input_data})

        all_images = torch.cat([x[0].unsqueeze(0) for x in input_data])
        all_images = all_images.reshape(
            (self.n_way, self.n_shot + self.n_query, *all_images.shape[1:])
        )
        # pylint: disable=not-callable
        all_labels = torch.tensor(
            [true_class_ids.index(x[1]) for x in input_data]
        ).reshape((self.n_way, self.n_shot + self.n_query))
        # pylint: enable=not-callable

        support_images = all_images[:, : self.n_shot].reshape(
            (-1, *all_images.shape[2:])
        )
        query_images = all_images[:, self.n_shot :].reshape((-1, *all_images.shape[2:]))
        support_labels = all_labels[:, : self.n_shot].flatten()
        query_labels = all_labels[:, self.n_shot :].flatten()

        return (
            support_images,
            support_labels,
            query_images,
            query_labels,
            true_class_ids,
        )


class InferenceDataset(Dataset):
    def __init__(self, infer_dataset_dir, transforms, n_way=5, n_shot=5):
        '''

        :param infer_dataset_dir:
            path of inference datasets,
            need to follow the directory structure below
            infer_dataset_dir
                - support
                    - class1
                        - 'image001...'
                        - ...
                    - class2
                        - 'image00n...'
                        - ...
                    - class3
                        - 'image00m...'
                        - ...
                - query
                    - images...
                    - ...

        '''

        self.n_way = n_way
        self.n_shot = n_shot
        self.transforms = transforms

        dataset_dir = infer_dataset_dir
        list_dir = os.listdir(dataset_dir)
        assert ('query' in list_dir and 'support' in list_dir),\
            "dataset dir need to have directory 'query' and 'support'"

        self.support_images = {}
        self.support_keys = os.listdir(os.path.join(dataset_dir, 'support'))
        for key in self.support_keys:
            path = os.path.join(dataset_dir, 'support', key)

            file_list = os.listdir(path)
            for file in file_list:
                image = Image.open(os.path.join(path, file))

                if key in self.support_images.keys():
                    self.support_images[key].append(image)
                else:
                    self.support_images[key] = [image]

        self.query_images = []
        for file in os.listdir(os.path.join(dataset_dir, 'query')):
            file_path = os.path.join(dataset_dir, 'query', file)
            image = Image.open(file_path)
            self.query_images.append(image)

    def __len__(self):
        return len(self.query_images)

    def __getitem__(self, idx):

        query = self.transforms(self.query_images[idx])
        _support_images = [
                    (random.sample(
                        self.support_images[label], self.n_shot
                    ), label)
                    for label in random.sample(self.support_images.keys(), self.n_way)
                ]

        support_images = []
        support_labels = []
        true_class = {}
        idx = 0
        for image_list, label in _support_images:
            true_class[idx] = label
            for image in image_list:
                image = self.transforms(image)
                support_images.append(image.unsqueeze(0))
                support_labels.append(idx)
            idx += 1

        support_images = torch.cat(support_images, dim=0)
        support_labels = torch.tensor(support_labels)

        return support_images, support_labels, query, true_class
