# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Optional

import numpy as np
import torch

from metaseq.data import data_utils
import time
from metaseq.data.atomic_array import AtomicArray


# gotta be a better way to get this...
def f():
    x = 4
    return lambda: x


Cell = type(f().__closure__[0])


class DocumentToSequenceDataset(torch.utils.data.IterableDataset):
    """Take a dataset containing documents and turn it into an iterable dataset
    returning sequences of block_size tokens.

    This dataset can only be iterated over once.

    Documents are optionally permuted (permute_documents).
    This iterator has a `len_cache` which is an AtomicArray mapping
    document id to the number of tokens in the document, which it populates.
    When populated the len_cache allows the iterator to quickly skip the first to_skip sequences.

    Args:
        dataset (~torch.utils.data.IterableDataset): dataset to chunk
        block_size (int): maximum block size
        break_mode (str, optional): Mode used for breaking tokens. Values can
            be one of:
            - 'none': break tokens into equally sized blocks (up to block_size)
        drop_last (bool, optional): drop the last item (default: False)
        padding_idx (int, optional): index to use for padding symbols
            (required if *drop_last* is ``False``)
        shuffle_buffer_size (int, optional): buffer this many items and shuffle
            using the provided *seed*; default value is 1, so no shuffling is
            performed. This can be adjusted dynamically after initialization,
            but only before iteration has begun.
        seed (int, optional): seed for shuffling
        permute_documents (bool, optional): randomly permute the order the documents are read (default: True)
        to_skip (int, optional): skip the first to_skip sequences (Default: 0)
    """

    def __init__(
        self,
        dataset: torch.utils.data.IterableDataset,
        block_size: int,
        break_mode: str = "none",
        drop_last: Optional[bool] = False,
        padding_idx: Optional[int] = None,
        shuffle_buffer_size: int = 1,
        seed: Optional[int] = None,
        len_cache=None,
        to_skip=0,
        permute_documents=True,
    ):
        super().__init__()
        self.dataset = dataset
        assert len(dataset) > 0
        self.worker_offset = 0

        self.document_shuffle_seed = seed
        self.shuffle_buffer_seed = (
            None if seed is None else seed + 2273 + 1284
        )  # at some point in the past these numbers were
        # added to keep the two seeds different

        self.indices = None
        self.len_cache = (
            AtomicArray(len(self.dataset)) if len_cache is None else len_cache
        )
        # self.len_cache = [ 0 for _ in range(len(self.dataset))] if len_cache is None else len_cache

        self.block_size = block_size
        self.break_mode = break_mode
        self.drop_last = drop_last
        self.padding_idx = padding_idx
        self.shuffle_buffer_size = shuffle_buffer_size
        self.to_skip = to_skip
        self.permute_documents = permute_documents

        if break_mode == "none":
            self.block_iterator = yield_token_blocks
        elif break_mode == "eos_pad_8":
            self.block_iterator = yield_single_sentences_pad_8
        elif break_mode == "complete":
            self.block_iterator = yield_doc_blocks
        elif break_mode == "passthrough":
            self.block_iterator = yield_passthrough
        else:
            raise ValueError(
                f'Invalid value for break_mode = {break_mode}. Available options are "none", "eos_pad_8" or "complete".'
            )

        if not drop_last and padding_idx is None:
            raise ValueError("padding_idx is required when drop_last is False")

        assert shuffle_buffer_size >= 1
        if shuffle_buffer_size > 1 and seed is None:
            raise ValueError("seed is required when shuffle_buffer_size > 1")

        # if break_mode != "none": raise NotImplementedError

        self._started_iteration = False

    def set_epoch(self, epoch):
        if hasattr(self.dataset, "set_epoch"):
            self.dataset.set_epoch(epoch)

        if self.permute_documents:
            # shuffle the dataset according to the seed argument and epoch
            seed = int(hash((self.document_shuffle_seed, epoch)) % 1e6)
            with data_utils.numpy_seed(seed):
                self.indices = np.random.permutation(len(self.dataset))
        else:
            self.indices = list(range(len(self.dataset)))

    def set_shuffle_buffer_size(self, new_shuffle_buffer_size):
        assert not self._started_iteration
        self.shuffle_buffer_size = new_shuffle_buffer_size

    def __iter__(self):
        skip_time = 0
        t0 = time.time()
        assert not self._started_iteration
        self.started_iteration = True

        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None and worker_info.num_workers > 1:
            chunks = np.array_split(self.indices, worker_info.num_workers)
            worker_id = (worker_info.id + self.worker_offset) % worker_info.num_workers
            indices = chunks[worker_id]
        else:
            worker_id = 0
            indices = self.indices

        to_skip = (
            self.to_skip if isinstance(self.to_skip, int) else self.to_skip[worker_id]
        )

        def documents():
            for idx in indices:
                ln = self.len_cache[idx]
                if ln == 0:
                    r = self.dataset[idx]
                    ln = r.shape[0]
                    self.len_cache[idx] = ln
                    yield (ln, Cell(r))
                else:
                    create = (lambda i: lambda: self.dataset[i])(idx)
                    yield (ln, Cell(create))

        block_itr = self.block_iterator(
            documents(),
            self.block_size,
            self.drop_last,
            self.padding_idx,
        )

        baserng = np.random.default_rng(self.shuffle_buffer_seed)
        rngstate = None
        rngcount = 0

        def create_rng():
            nonlocal rngstate, rngcount
            while True:
                rngstate = baserng._bit_generator.__getstate__()
                # overhead to calling 'integers' is high,
                # so request random numbers
                for x in baserng.integers(self.shuffle_buffer_size, size=1024):
                    rngcount += 1
                    yield x

        rng = None if self.shuffle_buffer_seed is None else iter(create_rng())

        buffer = []

        def get_next_item_and_replace_in_buffer(replacement_item):
            # return a random item from the buffer and replace with a new item
            if rng is not None:
                if len(buffer) == self.shuffle_buffer_size:
                    idx = next(rng)
                else:
                    if len(buffer) == self.shuffle_buffer_size - 1:
                        baserng._bit_generator.__setstate__(rngstate)
                        baserng.integers(self.shuffle_buffer_size, size=rngcount % 1024)
                    idx = baserng.integers(len(buffer))
            else:
                idx = 0
            item = buffer[idx]
            if replacement_item is not None:
                buffer[idx] = replacement_item
            else:
                buffer.pop(idx)
            return item

        def sequences():
            for block in block_itr:
                if len(buffer) < self.shuffle_buffer_size:
                    # initially fill the buffer to the requested size
                    buffer.append(block)
                else:
                    # return random block from the buffer and replace with new block
                    yield get_next_item_and_replace_in_buffer(block)

            # clear buffer of any remaining items
            while buffer:
                yield get_next_item_and_replace_in_buffer(None)

        for i, elem in enumerate(sequences()):
            if i >= to_skip:
                elem["ids"] = torch.LongTensor(elem["ids"])
                subsequences = []
                for c, start, ln in elem["block"]:
                    # A padding tensor (<padding_value>, 0, length)
                    if isinstance(c, int):
                        subsequences.append(subsequences[-1].new_full((ln,), c))
                    else:
                        # A deferred creation of a tensor (cell(<lambda>), start_idx, length)
                        # we use the cell to make sure we only ever create the tensor once
                        if not isinstance(c.cell_contents, torch.Tensor):
                            c.cell_contents = c.cell_contents()
                        # A tensor slice (cell(<Tensor>), start_idx, length)
                        subsequences.append(c.cell_contents[start : start + ln])
                elem["block"] = torch.cat(subsequences)
                elem["skip_time"] = skip_time
                yield elem
            elif i + 1 == to_skip:
                t1 = time.time()
                skip_time = t1 - t0


def yield_single_sentences_pad_8(iterable, block_size, drop_last, padding_idx):
    """Mimics sample-break-mode eos i.e. 1 example per sequence without any packing.
    When multiple examples are packed into a single sequence, example tokens would attend
    to tokens in neighbouring examples, which may be undesirable. This mode can
    avoid that. Since there is no packing, this mode is considerably slower.
    We round up the example length to a multiple of 8, pad to this length and
    return the example as is, without packing, truncating to block_size in cases of
    very long examples.
    """
    for idx, (tokens, tensor_generator) in enumerate(iterable):
        cur_block = []
        cur_block_ids = []
        if tokens > block_size:
            # truncate right side
            # TODO: Enable left side truncation
            tokens = block_size

        cur_block.append((tensor_generator, 0, tokens))

        # We round up to a multiple of 8 + 1, because later on
        # one element is removed for src/target tensor creation
        # which brings it back to a multiple of 8. block_size is
        # already passed with + 1 included.
        cur_block_remain = min(int(math.ceil(tokens / 8)) * 8 + 1, block_size)
        cur_block_remain -= tokens
        padding = (padding_idx, 0, cur_block_remain)
        cur_block.append(padding)
        cur_block_ids.append(idx)
        yield {
            "ids": cur_block_ids,
            "block": cur_block,
        }


def yield_doc_blocks(iterable, block_size, drop_last, padding_idx):
    """Mimics sample-break-mode complete"""
    cur_block = []
    cur_block_ids = []
    cur_block_remain = block_size
    for idx, (tokens, token_generator) in enumerate(iterable):
        if tokens > block_size:
            # truncate right side
            tokens = block_size

        if tokens > cur_block_remain:
            padding = (padding_idx, 0, cur_block_remain)
            cur_block.append(padding)
            yield {
                "ids": cur_block_ids,
                "block": cur_block,
            }

            cur_block = []
            cur_block_ids = []
            cur_block_remain = block_size

        cur_block.append((token_generator, 0, tokens))
        cur_block_ids.append(idx)
        cur_block_remain -= tokens
        assert cur_block_remain >= 0
    if not drop_last and len(cur_block) > 0:
        if cur_block_remain > 0:
            padding = (padding_idx, 0, cur_block_remain)
            cur_block.append(padding)
        yield {
            "ids": cur_block_ids,
            "block": cur_block,
        }


def yield_passthrough(iterable, block_size, drop_last, padding_idx):
    for idx, (tokens, token_generator) in enumerate(iterable):
        yield {
            "ids": [idx],
            "block": [(token_generator, 0, tokens)],
        }


def yield_token_blocks(iterable, block_size, drop_last, padding_idx):
    """Sample break mode = None. (Pre-Training default)."""
    cur_block = []
    cur_block_ids = []
    cur_block_remain = block_size
    for idx, (tokens, tensor_generator) in enumerate(iterable):
        cur_block_ids.append(idx)
        item_offset = 0
        while tokens:
            num_to_take = min(tokens, cur_block_remain)
            cur_block.append((tensor_generator, item_offset, num_to_take))
            item_offset += num_to_take
            cur_block_remain -= num_to_take
            tokens -= num_to_take

            if cur_block_remain == 0:
                yield {
                    "ids": cur_block_ids,
                    "block": cur_block,
                }
                cur_block = []
                cur_block_ids = []
                cur_block_remain = block_size

    if not drop_last and len(cur_block):
        if cur_block_remain:
            cur_block.append((padding_idx, 0, cur_block_remain))
        yield {
            "ids": cur_block_ids,
            "block": cur_block,
        }
