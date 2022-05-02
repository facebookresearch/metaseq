# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch

from metaseq.data import StreamingShuffleDataset


class TensorListDataset(torch.utils.data.Dataset):
    def __init__(self, tensor_list):
        self.tensor_list = tensor_list

    def __getitem__(self, index):
        return self.tensor_list[index]

    def __len__(self):
        return len(self.tensor_list)


def get_simple_dataset():
    return TensorListDataset(
        [
            torch.LongTensor([0]),
            torch.LongTensor([1, 2, 3]),
            torch.LongTensor([4]),
            torch.LongTensor([5]),
            torch.LongTensor([6, 7, 8]),
            torch.LongTensor([9, 10]),
        ]
    )


class TestStreamingShuffleDataset(unittest.TestCase):
    def test_set_epoch(self):
        dataset = get_simple_dataset()
        shuffle_ds = StreamingShuffleDataset(dataset, seed=0)

        shuffle_ds.set_epoch(1)
        ref_epoch1 = list(shuffle_ds)
        shuffle_ds.set_epoch(2)
        ref_epoch2 = list(shuffle_ds)

        self.assertTrue(
            torch.cat(ref_epoch1).tolist() == torch.cat(ref_epoch1).tolist()
        )
        self.assertFalse(
            torch.cat(ref_epoch1).tolist() == torch.cat(ref_epoch2).tolist()
        )

        shuffle_ds.set_epoch(1)
        self._compare(ref_epoch1, shuffle_ds)
        shuffle_ds.set_epoch(2)
        self._compare(ref_epoch2, shuffle_ds)
        shuffle_ds.set_epoch(2)
        self._compare(ref_epoch2, shuffle_ds)
        shuffle_ds.set_epoch(1)
        self._compare(ref_epoch1, shuffle_ds)

    def _compare(self, reference, dataset):
        ref_itr = iter(reference)
        ds_itr = iter(dataset)
        for ref, ds in zip(ref_itr, ds_itr):
            self.assertEqual(ref.tolist(), ds.tolist())
        with self.assertRaises(StopIteration):
            next(ref_itr)
        with self.assertRaises(StopIteration):
            next(ds_itr)
