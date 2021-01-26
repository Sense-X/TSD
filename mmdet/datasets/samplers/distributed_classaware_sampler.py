import math
import pickle as pk

import numpy as np
import torch
from torch.utils.data import BatchSampler, Sampler


class DistributedClassAwareSampler(Sampler):
    def __init__(self, dataset, num_replicas=None, rank=None, class_sample_path=None):

        if num_replicas is None:
            num_replicas = get_world_size()
        if rank is None:
            rank = get_rank()

        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas

        with open(class_sample_path, "rb") as f:
            self.class_dic = pk.load(f)
        self.class_num = len(self.class_dic.keys())
        self.class_num_list = [
            len(self.class_dic[i + 1]) for i in range(self.class_num)
        ]
        self.class_unique_num = len([i for i in self.class_num_list if i != 0])
        self.indices = None

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch

        def gen_class_num_indices(class_num_list):
            class_indices = np.random.permutation(self.class_num)
            id_indices = [
                self.class_dic[class_indice + 1][
                    np.random.permutation(class_num_list[class_indice])[0]
                ]
                for class_indice in class_indices
                if class_num_list[class_indice] != 0
            ]
            return id_indices

        # deterministically shuffle based on epoch
        np.random.seed(self.epoch + 1)
        num_bins = int(math.floor(self.total_size * 1.0 / self.class_num))
        indices = []
        for i in range(num_bins):
            indices += gen_class_num_indices(self.class_num_list)

        # add extra samples to make it evenly divisible
        indices += indices[: (self.total_size - len(indices))]
        assert len(indices) == self.total_size
        # subsample
        offset = self.num_samples * self.rank
        indices = indices[offset : offset + self.num_samples]
        assert len(indices) == self.num_samples
        self.indices = indices
