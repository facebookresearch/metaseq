# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch

from metaseq.data import data_utils


class StreamingShuffleDataset(torch.utils.data.IterableDataset):
    """Shuffle a dataset across epochs.

    Note that :func:`set_epoch` must be called before the first iteration.

    Args:
        dataset (~torch.utils.data.Dataset): dataset to shuffle
        seed (int): iterate over the underlying dataset in random order using
            this random seed
    """

    def __init__(self, dataset: torch.utils.data.Dataset, seed: int):
        super().__init__()
        self.dataset = dataset
        self.seed = seed

        assert len(dataset) > 0

        self.indices = None

    def set_epoch(self, epoch):
        # shuffle the dataset according to the seed argument and epoch
        seed = int(hash((self.seed, epoch)) % 1e6)
        with data_utils.numpy_seed(seed):
            self.indices = np.random.permutation(len(self.dataset))

        if hasattr(self.dataset, "set_epoch"):
            self.dataset.set_epoch(epoch)

    def __iter__(self):
        assert (
            self.indices is not None
        ), "must call StreamingShuffleDataset.set_epoch before iteration"
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None and worker_info.num_workers > 1:
            chunks = np.array_split(self.indices, worker_info.num_workers)
            indices = chunks[worker_info.id]
        else:
            indices = self.indices

        for idx in indices:
            yield self.dataset[idx]
