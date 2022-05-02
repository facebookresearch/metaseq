# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch

from metaseq.data import iterators


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


class TestStreamingIterators(unittest.TestCase):
    def test_streaming_counting_iterator(self):
        ref = list(range(10))
        itr = iterators.StreamingCountingIterator(ref)
        for i, ref_i in enumerate(ref):
            self.assertTrue(itr.has_next())
            self.assertEqual(itr.n, i)
            self.assertEqual(next(itr), ref_i)
        self.assertEqual(itr.n, len(ref))
        self.assertFalse(itr.has_next())
        with self.assertRaises(StopIteration):
            next(itr)

    def test_streaming_epoch_batch_iterator_drop_last_True(self):
        self._test_streaming_epoch_batch_iterator(drop_last=True)

    def test_streaming_epoch_batch_iterator_drop_last_False(self):
        self._test_streaming_epoch_batch_iterator(drop_last=False)

    def test_streaming_epoch_batch_iterator_state_dict(self):
        def hook_fn(epoch_batch_itr, itr):
            new_epoch_batch_itr = iterators.StreamingEpochBatchIterator(
                # recreate the dataset
                dataset=get_simple_dataset(),
                batch_size=epoch_batch_itr.batch_size,
                collate_fn=epoch_batch_itr.collate_fn,
                drop_last=epoch_batch_itr.drop_last,
            )
            new_epoch_batch_itr.load_state_dict(epoch_batch_itr.state_dict())
            return new_epoch_batch_itr, new_epoch_batch_itr.next_epoch_itr()

        self._test_streaming_epoch_batch_iterator(drop_last=True, hook_fn=hook_fn)
        self._test_streaming_epoch_batch_iterator(drop_last=False, hook_fn=hook_fn)

    def _test_streaming_epoch_batch_iterator(self, drop_last, hook_fn=None):
        dataset = get_simple_dataset()
        epoch_batch_itr = iterators.StreamingEpochBatchIterator(
            dataset,
            batch_size=2,
            collate_fn=torch.cat,
            drop_last=drop_last,
        )
        assert epoch_batch_itr.next_epoch_idx == 1
        itr = epoch_batch_itr.next_epoch_itr()
        assert epoch_batch_itr.iterations_in_epoch == 0
        assert not epoch_batch_itr.end_of_epoch()

        if hook_fn is not None:
            epoch_batch_itr, itr = hook_fn(epoch_batch_itr, itr)
        assert next(itr).tolist() == [0, 1, 2, 3]
        assert epoch_batch_itr.iterations_in_epoch == 1
        assert not epoch_batch_itr.end_of_epoch()

        if hook_fn is not None:
            epoch_batch_itr, itr = hook_fn(epoch_batch_itr, itr)
        assert next(itr).tolist() == [4, 5, 6, 7]
        assert epoch_batch_itr.iterations_in_epoch == 2

        if not drop_last:
            if hook_fn is not None:
                epoch_batch_itr, itr = hook_fn(epoch_batch_itr, itr)
            assert next(itr).tolist() == [8, 9]
            assert epoch_batch_itr.iterations_in_epoch == 3

        assert epoch_batch_itr.end_of_epoch()
        with self.assertRaises(StopIteration):
            next(itr)


if __name__ == "__main__":
    unittest.main()
