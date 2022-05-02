# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch

from metaseq.data import data_utils
from . import BaseWrapperDataset


class PadDataset(BaseWrapperDataset):
    def __init__(self, dataset, pad_idx, left_pad, pad_length=None):
        super().__init__(dataset)
        self.pad_idx = pad_idx
        self.left_pad = left_pad
        self.pad_length = pad_length

    def collater(self, samples):
        return data_utils.collate_tokens(
            samples, self.pad_idx, left_pad=self.left_pad, pad_to_length=self.pad_length
        )


class LeftPadDataset(PadDataset):
    def __init__(self, dataset, pad_idx, pad_length=None):
        super().__init__(dataset, pad_idx, left_pad=True, pad_length=pad_length)


class RightPadDataset(PadDataset):
    def __init__(self, dataset, pad_idx, pad_length=None):
        super().__init__(dataset, pad_idx, left_pad=False, pad_length=pad_length)


class MultiplePadDataset(BaseWrapperDataset):
    """
    This class pads the given dataset to ensure that the padded size is a
    multiple of the given `multiple`.

    For instance,
    MultiplePadDataset(
        tgt_dataset, pad_idx=self.source_dictionary.pad(), multiple=8
    )
    would pad the tgt_dataset in multiples of 8.
    """

    def __init__(self, dataset, pad_idx, multiple):
        super().__init__(dataset)
        self.pad_idx = pad_idx
        self.multiple = multiple

    def collater(self, samples):
        max_len = max([s.size(0) for s in samples])
        max_len_multiple = int(math.ceil(max_len / self.multiple)) * self.multiple

        return data_utils.collate_tokens(
            samples, self.pad_idx, left_pad=False, pad_to_length=max_len_multiple
        )

    def __getitem__(self, index):
        l = len(self.dataset[index])
        cur_block = []
        cur_block.append(self.dataset[index])
        cur_block_remain = int(math.ceil(l / self.multiple) * self.multiple)

        cur_block_remain -= self.dataset[index].numel()
        padding = cur_block[-1].new_full((cur_block_remain,), self.pad_idx)
        cur_block.append(padding)

        return torch.cat(cur_block)
