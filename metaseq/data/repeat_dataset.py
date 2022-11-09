# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import bisect

import numpy as np
from torch.utils.data.dataloader import default_collate

from . import BaseWrapperDataset


class RepeatDataset(BaseWrapperDataset):
    def __init__(self, dataset, replication_factor: int):
        super(RepeatDataset, self).__init__(dataset)
        self.replication_factor = replication_factor

    def __len__(self):
        return len(self.dataset) * self.replication_factor

    def __getitem__(self, idx):
        return self.dataset[idx // self.replication_factor]

    def size(self, idx: int):
        return self.dataset.size(idx // self.replication_factor)

    def num_tokens(self, idx: int):
        return self.dataset.num_tokens(idx // self.replication_factor)

    @property
    def sizes(self):
        if isinstance(self.dataset.sizes, np.ndarray):
            return np.repeat(self.dataset.sizes, self.replication_factor)
        else:
            # Only support underlying dataset with single size array.
            assert isinstance(self.dataset.sizes, list)
            list_expanded = []
            for s in self.dataset.sizes:
                list_expanded.extend([s] * self.replication_factor)
            return list_expanded

    def collater(self, samples):
        return self.dataset.collater(samples)

    def ordered_indices(self):
        return self.dataset.ordered_indices()

    @property
    def supports_prefetch(self):
        return getattr(self.dataset, "supports_prefetch", False)

    def attr(self, attr: str, index: int):
        return self.dataset.attr(attr, index // self.replication_factor)

    def prefetch(self, indices):
        return False
