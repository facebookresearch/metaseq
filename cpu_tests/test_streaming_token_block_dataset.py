# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import numpy as np
import torch

from metaseq.data import StreamingTokenBlockDataset


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


class TestStreamingTokenBlockDataset(unittest.TestCase):
    def test_drop_last_True(self):
        self._test_simple(drop_last=True)

    def test_drop_last_False(self):
        self._test_simple(drop_last=False)

    def test_buffer_drop_last_True(self):
        self._test_buffer(drop_last=True)

    def test_buffer_drop_last_False(self):
        self._test_buffer(drop_last=False)

    def test_very_large_buffer_drop_last_True(self):
        self._test_very_large_buffer(drop_last=True)

    def test_very_large_buffer_drop_last_False(self):
        self._test_very_large_buffer(drop_last=False)

    def _test_simple(self, drop_last):
        dataset = get_simple_dataset()
        token_block_ds = StreamingTokenBlockDataset(
            dataset,
            block_size=2,
            drop_last=drop_last,
            padding_idx=-1,
        )
        dataloader = iter(token_block_ds)
        assert next(dataloader)["block"].tolist() == [0, 1]
        assert next(dataloader)["block"].tolist() == [2, 3]
        assert next(dataloader)["block"].tolist() == [4, 5]
        assert next(dataloader)["block"].tolist() == [6, 7]
        assert next(dataloader)["block"].tolist() == [8, 9]
        if not drop_last:
            assert next(dataloader)["block"].tolist() == [10, -1]
        with self.assertRaises(StopIteration):
            next(dataloader)

    def _test_buffer(self, drop_last, seed=42):
        # maintain shadow rng to ensure iteration order matches expectations
        shadow_rng = np.random.default_rng(2273 + seed)

        dataset = get_simple_dataset()
        token_block_ds = StreamingTokenBlockDataset(
            dataset,
            block_size=2,
            drop_last=drop_last,
            padding_idx=-1,
            shuffle_buffer_size=3,
            seed=seed,
        )
        dataloader = iter(token_block_ds)

        # we expect token_block_ds to buffer the first three blocks,
        # then return random blocks and replace them thereafter
        expected_buffer = [
            [0, 1],
            [2, 3],
            [4, 5],
        ]

        next_idx = shadow_rng.integers(3)
        assert next(dataloader)["block"].tolist() == expected_buffer[next_idx]
        expected_buffer[next_idx] = [6, 7]

        next_idx = shadow_rng.integers(3)
        assert next(dataloader)["block"].tolist() == expected_buffer[next_idx]
        expected_buffer[next_idx] = [8, 9]

        next_idx = shadow_rng.integers(3)
        assert next(dataloader)["block"].tolist() == expected_buffer[next_idx]
        if not drop_last:
            expected_buffer[next_idx] = [10, -1]
        else:
            expected_buffer.pop(next_idx)

        while expected_buffer:
            next_idx = shadow_rng.integers(len(expected_buffer))
            assert next(dataloader)["block"].tolist() == expected_buffer[next_idx]
            expected_buffer.pop(next_idx)

        with self.assertRaises(StopIteration):
            next(dataloader)

    def _test_very_large_buffer(self, drop_last, seed=42):
        # maintain shadow rng to ensure iteration order matches expectations
        shadow_rng = np.random.default_rng(2273 + seed)

        dataset = get_simple_dataset()
        token_block_ds = StreamingTokenBlockDataset(
            dataset,
            block_size=2,
            drop_last=drop_last,
            padding_idx=-1,
            shuffle_buffer_size=100,  # bigger than full dataset
            seed=seed,
        )
        dataloader = iter(token_block_ds)

        expected_buffer = [
            [0, 1],
            [2, 3],
            [4, 5],
            [6, 7],
            [8, 9],
        ]
        if not drop_last:
            expected_buffer.append([10, -1])

        while expected_buffer:
            next_idx = shadow_rng.integers(len(expected_buffer))
            assert next(dataloader)["block"].tolist() == expected_buffer[next_idx]
            expected_buffer.pop(next_idx)

        with self.assertRaises(StopIteration):
            next(dataloader)

    def _test_break_mode_eos_pad_8(self):
        dataset = TensorListDataset(
            [
                torch.LongTensor([0]),
                torch.LongTensor([1, 2, 3]),
                torch.LongTensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]),
            ]
        )
        token_block_ds = StreamingTokenBlockDataset(
            dataset,
            block_size=10,
            drop_last=False,
            padding_idx=-1,
            break_mode="eos_pad_8",
        )
        expected_buffer = [
            [0, -1, -1, -1, -1, -1, -1, -1, -1],
            [1, 2, 3, -1, -1, -1, -1, -1, -1],
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        ]

        dataloader = iter(token_block_ds)
        assert (
            next(dataloader)["block"].tolist() == expected_buffer[0]
        )  # padding to multiple of 8 + 1
        assert (
            next(dataloader)["block"].tolist() == expected_buffer[1]
        )  # padding to multiple of 8 + 1
        assert (
            next(dataloader)["block"].tolist() == expected_buffer[2]
        )  # padding to block size

        with self.assertRaises(StopIteration):
            next(dataloader)
