# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch

from metaseq.data import (
    iterators,
    StreamingShuffleDataset,
    StreamingTokenBlockDataset,
    PartitionedStreamingDataset,
)
from metaseq.data.document_to_sequence import DocumentToSequenceDataset, LockingArray

import random
import pickle


class TensorListDataset(torch.utils.data.Dataset):
    def __init__(self, tensor_list):
        self.tensor_list = tensor_list
        self.queried = 0

    def __len__(self):
        return len(self.tensor_list)

    def __getitem__(self, idx):
        self.queried += 1
        return self.tensor_list[idx]


def get_simple_dataset():
    dataset = TensorListDataset(
        [
            torch.LongTensor([0, 1]),
            torch.LongTensor([2, 3]),
            torch.LongTensor([4, 5]),
            torch.LongTensor([6, 7]),
            torch.LongTensor([8, 9]),
        ]
    )
    dataset = DocumentToSequenceDataset(
        dataset,
        block_size=None,
        permute_documents=False,
        break_mode="passthrough",
        padding_idx=1,
    )
    return dataset


class FakeTensorData(torch.utils.data.Dataset):
    def __init__(self):
        self.rng = random.Random(0)
        self.trng = torch.Generator()
        self.trng.manual_seed(0)
        self.items = [
            torch.randint(
                256, size=(self.rng.randrange(512, 2048),), generator=self.trng
            )
            for _ in range(len(self))
        ]
        self.queried = 0
        self.realized = [False for _ in self.items]

    def __len__(self):
        return 128

    def __getitem__(self, idx):
        self.queried += 1
        assert not self.realized[idx], "Document unexpectedly loaded twice"
        self.realized[idx] = True
        return self.items[idx]

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]


class TestStreamingIterators(unittest.TestCase):
    def test_streaming_counting_iterator(self):
        ref = list(range(10))
        itr = iterators.StreamingCountingIterator(ref, 0, 1, 1)
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
            queried = epoch_batch_itr.dataset.dataset.queried
            if epoch_batch_itr.iterations_in_epoch == 2:
                assert queried == 2, "Deferred Token cache didn't cache loading"
            new_epoch_batch_itr = iterators.StreamingEpochBatchIterator(
                # recreate the dataset
                dataset=get_simple_dataset(),
                batch_size=epoch_batch_itr.batch_size,
                collate_fn=epoch_batch_itr.collate_fn,
                drop_last=epoch_batch_itr.drop_last,
            )
            # pickle the state_dict to test picklability
            psd = pickle.dumps(epoch_batch_itr.state_dict())
            new_epoch_batch_itr.load_state_dict(pickle.loads(psd))
            return new_epoch_batch_itr, new_epoch_batch_itr.next_epoch_itr()

        self._test_streaming_epoch_batch_iterator(drop_last=True, hook_fn=hook_fn)
        self._test_streaming_epoch_batch_iterator(drop_last=False, hook_fn=hook_fn)

    def _test_streaming_epoch_batch_iterator(self, drop_last, hook_fn=None):
        dataset = get_simple_dataset()
        epoch_batch_itr = iterators.StreamingEpochBatchIterator(
            dataset,
            batch_size=2,
            collate_fn=lambda xs: torch.cat([x["block"] for x in xs]),
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

    def test_to_skip_iterator(self):
        def create_dataset(
            break_mode="none", drop_last=True, sequence_size=2049, num_shards=1
        ):
            dataset = FakeTensorData()
            token_dataset = DocumentToSequenceDataset(
                dataset,
                # We generate blocks with one extra token, so that we have a target
                # for the final input token. This results in slight data loss.
                block_size=sequence_size,
                break_mode=break_mode,
                # we drop the remainder block during training
                drop_last=drop_last,
                padding_idx=1,
                seed=42,
            )
            token_dataset.set_shuffle_buffer_size(4)
            token_dataset.set_epoch(0)
            partitioned_dataset = PartitionedStreamingDataset(
                token_dataset,
                num_shards=num_shards,
                shard_id=0,
                drop_last=True,
            )
            return partitioned_dataset, dataset, token_dataset

        def run_test(drop_last, break_mode):
            dataset, fake_dataset, token_dataset = create_dataset(
                drop_last=drop_last, break_mode=break_mode
            )
            num_iters = 0
            values = []
            for i, x in enumerate(dataset):
                assert isinstance(x["block"], torch.Tensor)
                assert (
                    x["block"].shape[0] == 2049
                    or (drop_last and x["block"].shape[0] <= 2049)
                    or break_mode == "eos_pad_8"
                )
                num_iters += 1
                values.append(x)

            a_fourth = num_iters // 4
            dataset2, fake_dataset2, token_dataset2 = create_dataset(
                drop_last=drop_last, break_mode=break_mode
            )

            token_dataset2.len_cache = token_dataset.len_cache
            token_dataset2.to_skip = a_fourth
            value1 = values[a_fourth]
            it = iter(dataset2)
            value2 = next(it)
            assert torch.allclose(value1["block"], value2["block"])
            # check that we didn't actually query all the dataset to do the fast forward
            assert fake_dataset2.queried <= len(value2["ids"]) + 1
            # load the rest of the dataset to check we are only
            # truly loading documents once even when we defer
            while True:
                try:
                    next(it)
                except StopIteration:
                    break

        run_test(drop_last=True, break_mode="complete")
        run_test(drop_last=False, break_mode="complete")

        run_test(drop_last=True, break_mode="eos_pad_8")
        run_test(drop_last=False, break_mode="eos_pad_8")

        run_test(drop_last=True, break_mode="none")
        run_test(drop_last=False, break_mode="none")

        # now do a test with actual dataloader object.
        # the tricky bit here is if we suspend on an iteration `n`` that is not a multiple of the
        # number of workers, we have to restore where the first requested batch is from
        # to a worker (n % num_workers) that isn't worker 0. However, DataLoader returns data from worker 0 first.
        # So we shift what each worker thinks its ID is by n so worker 0 will behave as worker (n % num_workers).
        dataset, fake_dataset, token_dataset = create_dataset(
            drop_last=True, break_mode="none"
        )
        token_dataset.set_num_workers(2)
        dataloader1 = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=1,
            num_workers=2,
            pin_memory=True,
            drop_last=True,
        )
        consumed = [0, 0]
        ITERS = 8
        for i, last in zip(range(ITERS), dataloader1):
            if i < ITERS - 1:  # don't include the last one
                consumed[i % 2] += 1

        len_cache = token_dataset.len_cache
        dataset, fake_dataset, token_dataset = create_dataset(
            drop_last=True, break_mode="none"
        )
        token_dataset.set_num_workers(2)
        token_dataset.len_cache = len_cache
        token_dataset.to_skip = consumed
        token_dataset.worker_offset = ITERS - 1
        dataloader2 = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=1,
            num_workers=2,
            pin_memory=True,
            drop_last=True,
        )
        first = next(iter(dataloader2))
        assert torch.allclose(last["block"], first["block"])

    def test_document_to_sequence(self):
        MAX_SEQ_LEN = 2048

        def get_traditional_iterator(dataset, break_mode, drop_last):
            shuffle_dataset = StreamingShuffleDataset(dataset, seed=42)
            shuffle_dataset.set_epoch(0)
            token_dataset = StreamingTokenBlockDataset(
                shuffle_dataset,
                # We generate blocks with one extra token, so that we have a target
                # for the final input token. This results in slight data loss.
                block_size=MAX_SEQ_LEN + 1,
                break_mode=break_mode,
                # we drop the remainder block during training
                drop_last=drop_last,
                padding_idx=1,
                # 1284 is a randomly-generated offset to decouple the seed used here
                # from the seed used above in StreamingShuffleDataset
                seed=1284 + 42,
            )
            token_dataset.set_shuffle_buffer_size(4)
            return token_dataset

        def get_document_to_sequence_iterator(dataset, break_mode, drop_last):
            document_to_sequence_dataset = DocumentToSequenceDataset(
                dataset,
                # We generate blocks with one extra token, so that we have a target
                # for the final input token. This results in slight data loss.
                block_size=MAX_SEQ_LEN + 1,
                break_mode=break_mode,
                # we drop the remainder block during training
                drop_last=drop_last,
                padding_idx=1,
                # 1284 is a randomly-generated offset to decouple the seed used here
                # from the seed used above in StreamingShuffleDataset
                seed=42,
            )
            document_to_sequence_dataset.set_epoch(0)
            document_to_sequence_dataset.set_shuffle_buffer_size(4)
            return document_to_sequence_dataset

        def compare(break_mode, drop_last):
            a = get_traditional_iterator(FakeTensorData(), break_mode, drop_last)
            b = get_document_to_sequence_iterator(
                FakeTensorData(), break_mode, drop_last
            )
            a_values = list(a)
            b_values = list(b)
            self.assertEqual(len(a_values), len(b_values))

            for av, bv in zip(a_values, b_values):
                self.assertTrue(torch.allclose(av["ids"], bv["ids"]))
                self.assertTrue(torch.allclose(av["block"], bv["block"]))

        compare("none", False)
        compare("eos_pad_8", False)
        compare("complete", False)

        compare("none", True)
        compare("eos_pad_8", True)
        compare("complete", True)

    def test_locking_array(self):
        l = LockingArray(20, 8)
        for i in range(20):
            l.data[i] = i
        l2 = pickle.loads(pickle.dumps(l))
        assert len(l2.data) == 20
        assert len(l2.worker_locks) == 8
        for i in range(20):
            assert l2.data[i] == i


if __name__ == "__main__":
    unittest.main()
