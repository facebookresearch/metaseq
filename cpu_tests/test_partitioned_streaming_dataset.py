# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch

from metaseq.data import PartitionedStreamingDataset


class TensorListIterableDataset(torch.utils.data.IterableDataset):
    def __init__(self, tensor_list):
        self.tensor_list = tensor_list

    def __iter__(self):
        for tensor in self.tensor_list:
            yield tensor


def get_simple_dataset():
    return TensorListIterableDataset(
        [
            torch.LongTensor([0, 1]),
            torch.LongTensor([2, 3]),
            torch.LongTensor([4, 5]),
            torch.LongTensor([6, 7]),
            torch.LongTensor([8, 9]),
        ]
    )


class TestPartitionedStreamingDataset(unittest.TestCase):
    def test_drop_last_True_shard_0(self):
        self._test_simple(drop_last=True, shard_id=0)

    def test_drop_last_True_shard_1(self):
        self._test_simple(drop_last=True, shard_id=1)

    def test_drop_last_False_shard_0(self):
        self._test_simple(drop_last=False, shard_id=0)

    def test_drop_last_False_shard_1(self):
        self._test_simple(drop_last=False, shard_id=1)

    def _test_simple(self, drop_last, shard_id):
        dataset = get_simple_dataset()
        partitioned_ds = PartitionedStreamingDataset(
            dataset,
            num_shards=2,
            shard_id=shard_id,
            drop_last=drop_last,
        )
        dataloader = iter(partitioned_ds)
        if shard_id == 0:
            assert next(dataloader).tolist() == [0, 1]
            assert next(dataloader).tolist() == [4, 5]
            if not drop_last:
                assert next(dataloader).tolist() == [8, 9]
        else:
            assert shard_id == 1
            assert next(dataloader).tolist() == [2, 3]
            assert next(dataloader).tolist() == [6, 7]
            if not drop_last:
                assert next(dataloader) is None
        with self.assertRaises(StopIteration):
            next(dataloader)
