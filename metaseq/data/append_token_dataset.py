# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch

from . import BaseWrapperDataset


class AppendTokenDataset(BaseWrapperDataset):
    def __init__(self, dataset: BaseWrapperDataset, token: str = None):
        super().__init__(dataset)
        self.token = token
        if token is not None:
            self._sizes = np.array(dataset.sizes) + 1
        else:
            self._sizes = dataset.sizes

    def __getitem__(self, idx: int):
        item = self.dataset[idx]
        if self.token is not None:
            item = torch.cat([item, item.new([self.token])])
        return item

    @property
    def sizes(self):
        return self._sizes

    def num_tokens(self, index: int):
        n = self.dataset.num_tokens(index)
        if self.token is not None:
            n += 1
        return n

    def size(self, index: int):
        n = self.dataset.size(index)
        if self.token is not None:
            n += 1
        return n
