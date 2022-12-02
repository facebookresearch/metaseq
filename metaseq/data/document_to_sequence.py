# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import math
import logging
from typing import Optional

import numpy as np
import torch

from metaseq.data import data_utils
from metaseq.distributed import utils as distributed_utils
import time

from typing import Union, List, Iterable, Tuple, TypedDict, Literal

from multiprocessing import Array, Lock
from contextlib import contextmanager
from ctypes import c_int, sizeof, memmove, addressof

logger = logging.getLogger(__name__)


class LockingArray:
    def __init__(self, size, num_workers=1):
        # each worker is only allowed to write to its set of elements in the array
        # the worker locks should be uncontested in normal runs resulting in low overhead.

        # the training process can read the array by locking all the worker locks at once
        # when it needs to get the array for checkpointing

        self.data = Array("i", size, lock=False)
        self.set_num_workers(num_workers)

    def set_num_workers(self, num_workers):
        self.worker_locks = [Lock() for _ in range(num_workers)]

    @contextmanager
    def _lock_all(self):
        locked = []
        try:
            for l in self.worker_locks:
                l.acquire()
                locked.append(l)
            yield
        finally:
            for l in reversed(locked):
                l.release()

    def __getstate__(self):
        with self._lock_all():
            all_bytes = bytes(self.data)
        return (all_bytes, len(self.worker_locks))

    def __setstate__(self, state):
        all_bytes, num_workers = state
        ln = len(all_bytes) // sizeof(c_int)
        self.__init__(ln, num_workers)
        memmove(addressof(self.data), all_bytes, len(all_bytes))


# Documents can be in one of three states:
# (1) A torch.Tensor holding the tokens in the document
# (2) The string "padding" which is a document filled with self.padding_idx
#     used to pad out sequences in some break_modes
# (3) An integer index into the underlying dataset: self.dataset[index]
#     This form is used to avoid loading documents that will just be skipped
#     as specified by to_skip
Document = Union[
    int,  # an unloaded document self.dataset[value]
    Literal["padding"],  # conceptually a tensor full of self.padding_idx
    torch.Tensor,  # loaded 1-D tensor full of the tokens from the document
]


# A part of a sequence derived from the slice of a single document
# The first element is a single-element list containing a document.
# It is done this way so that if we need to actually tokenize the document
# represented by an index, we can replace it with a Tensor in all
# the Sequence fragments that reference the document.
SequenceFragment = Tuple[List[Document], int, int]  # document, offset, length


class Sequence(TypedDict):
    block: List[
        SequenceFragment
    ]  # These are torch.cat'd together to get the whole sequence
    ids: List[int]


def blocked_random(seed, normal_size):
    """
    Create a function that returns random numbers based on seed.
    Block calls to numpy's integers function because it has high overhead.
    """
    baserng = np.random.default_rng(seed)
    state = None
    buf = None
    n = 0
    use_batch = True

    def integers(high):
        nonlocal state, buf, n, use_batch
        if use_batch:
            # in the common case we ask for high=shuffle_buffer_size
            # we can batch calls to numpy to generate these values
            if high == normal_size:
                if n % 1024 == 0:
                    state = baserng._bit_generator.__getstate__()
                    buf = baserng.integers(normal_size, size=1024)
                r = buf[n % 1024]
                n += 1
                return r

            # when the buffer drains at the end, we start asking
            # for a smaller range of random numbers than we have batched.
            # To match our previous behavior, reset the
            # state to what it would have been before batching
            # and generate the final numbers without batching
            if state is not None:
                baserng._bit_generator.__setstate__(state)
                baserng.integers(normal_size, size=n % 1024)
            use_batch = False
        return baserng.integers(high)

    return integers


class DocumentToSequenceDataset(torch.utils.data.IterableDataset):
    """Take a dataset containing documents and turn it into an iterable dataset
    returning sequences of block_size tokens.

    This dataset can only be iterated over once.

    Documents are optionally permuted (permute_documents=True).
    This iterator has a `len_cache` which is an AtomicArray mapping
    document id to the number of tokens in the document, which it populates.
    When populated the len_cache allows the iterator to quickly skip the first to_skip sequences.

    Args:
        dataset (~torch.utils.data.IterableDataset): dataset to chunk
        block_size (int): maximum block size
        break_mode (str, optional): Mode used for breaking tokens. Values can
            be one of:
            - 'none': break tokens into equally sized blocks (up to block_size)
        drop_last (bool, optional): drop the last item (default: True)
        padding_idx (int, optional): index to use for padding symbols
            (required if *drop_last* is ``False``)
        shuffle_buffer_size (int, optional): buffer this many items and shuffle
            using the provided *seed*; default value is 1, so no shuffling is
            performed. This can be adjusted dynamically after initialization,
            but only before iteration has begun.
        seed (int, optional): seed for shuffling
        permute_documents (bool, optional): randomly permute the order the documents are read (default: True)
        source_target (bool, optional): the input dataset returns a tuple of tokens lists (source, target) (default: False)
        to_skip (int, optional): skip the first to_skip sequences before iteration begins (Default: 0)
    """

    def __init__(
        self,
        dataset: torch.utils.data.IterableDataset,
        block_size: int,
        break_mode: str = "none",
        drop_last: Optional[bool] = True,
        padding_idx: Optional[int] = None,
        shuffle_buffer_size: int = 1,
        seed: Optional[int] = None,
        len_cache=None,
        to_skip=0,
        permute_documents=True,
        source_target=False,
    ):
        super().__init__()
        self.dataset = dataset
        assert len(dataset) > 0

        # PyTorch dataloaders round-robin through worker processes to request data,
        # but always start with worker 0.
        # If we stop iteration on round n when n is not divisible by the
        # num_workers, then when we start again, we need to start with worker n % num_workers.
        # We adjust which worker is which by adding an offset to each worker, turning worker 0
        # of the new dataloaders into worker n % num_workers of the previous dataloader.
        self.worker_offset = 0

        self.document_shuffle_seed = seed
        self.shuffle_buffer_seed = (
            None if seed is None else seed + 2273 + 1284
        )  # at some point in the past these numbers were
        # added to keep the two seeds different

        self.indices = None

        # A map from document id -> number of tokens in the document.
        # This is an atomic array because it shared among the dataloader workers and the training
        # process.
        self.len_cache = (
            LockingArray(len(self.dataset)) if len_cache is None else len_cache
        )

        self.block_size = block_size
        self.break_mode = break_mode
        self.drop_last = drop_last
        self.padding_idx = padding_idx
        self.shuffle_buffer_size = shuffle_buffer_size
        self.to_skip = to_skip
        self.permute_documents = permute_documents
        self.source_target = source_target

        if break_mode == "none":
            if self.source_target:
                self.block_iterator = yield_doc_blocks
            else:
                self.block_iterator = yield_token_blocks
        elif break_mode == "eos_pad_8":
            self.block_iterator = yield_single_sentences_pad_8
        elif break_mode == "complete":
            self.block_iterator = yield_doc_blocks
        elif break_mode == "passthrough":
            self.block_iterator = yield_passthrough
        else:
            raise ValueError(
                f"Invalid value for break_mode = {break_mode}."
                'Available options are "none", "eos_pad_8", "complete", or "passthrough".'
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

    def set_num_workers(self, num_workers):
        self.len_cache.set_num_workers(max(1, num_workers))

    def __iter__(self):
        skip_time = 0
        t0 = time.time()
        assert not self._started_iteration
        self.started_iteration = True

        # When loading with multiple dataloader processes, split the
        # indices up among workers evenly.
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None and worker_info.num_workers > 1:
            chunks = np.array_split(self.indices, worker_info.num_workers)
            worker_id = (worker_info.id + self.worker_offset) % worker_info.num_workers
            indices = chunks[worker_id]
        else:
            worker_id = 0
            indices = self.indices

        # The number of sequences to skip before starting, used to fast-forward the
        # data loader after restarting from a checkpoint. This may be
        # different per worker if training stop on an iteration not divisible by the worker size
        to_skip = (
            self.to_skip if isinstance(self.to_skip, int) else self.to_skip[worker_id]
        )

        def documents() -> Iterable[Tuple[int, Document]]:
            """
            Generator that produces whole documents in a shuffled order (permute_documents=True).
            It returns a tuple of (number_of_tokens, List[Document]).
            When the number of tokens is already in the `len_cache`, we defer actually loading
            the document because we might end up skipping it entirely.
            """
            for idx in indices:
                ln = self.len_cache.data[idx]
                if ln == 0:
                    # Cache miss: we don't know the number of tokens
                    # so we have to load and tokenize the document.
                    r = self.dataset[idx]
                    if self.source_target:
                        ln = r[0].shape[0]
                    else:
                        ln = r.shape[0]
                    self.len_cache.data[idx] = ln
                    yield (ln, [r])
                else:
                    # Cache hit: we know the number of tokens, so we can
                    # skip loading the document for now.

                    # We create a single-element list here, so that we can replace the single element
                    # with the real Tensor value the first time _any_ SentenceFragment needs the
                    # real data from this document.
                    yield (ln, [int(idx)])

        block_itr = self.block_iterator(documents(), self.block_size, self.drop_last)

        # block_itr returns sequences from documents in order
        # we need to shuffle up this order to avoid having batches full of sequences from
        # one document. This is done with a "shuffle-buffer" that randomizes access

        buffer = []

        random = (
            blocked_random(self.shuffle_buffer_seed, self.shuffle_buffer_size)
            if self.shuffle_buffer_seed is not None
            else lambda x: 0
        )

        def get_next_item_and_replace_in_buffer(replacement_item: Sequence) -> Sequence:
            # return a random item from the buffer and replace with a new item
            idx = random(len(buffer))
            item = buffer[idx]
            if replacement_item is not None:
                buffer[idx] = replacement_item
            else:
                buffer.pop(idx)
            return item

        def sequences() -> Iterable[Sequence]:
            """
            Generator that returns sequences after shuffling the order of the
            sequences through the shuffle buffer.
            """
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

        # Finally, we iterate through our shuffled sequences, skipping the first to_skip
        # and performing any tokenization we delayed during the skipping process.
        seq_it = iter(sequences())

        try:
            if to_skip > 0 and worker_id == 0:
                logger.info(f"Skipping {to_skip} sequences")
            with self.len_cache.worker_locks[worker_id]:
                for i in range(to_skip):
                    next(seq_it)
            t1 = time.time()
            skip_time = t1 - t0
            if worker_id == 0 and distributed_utils.get_global_rank() == 0:
                local_rank = (
                    os.environ["LOCAL_RANK"] if "LOCAL_RANK" in os.environ else 0
                )
                logger.info(
                    f"Begin filling streaming dataset buffer for each worker on rank {local_rank}"
                )
            while True:
                with self.len_cache.worker_locks[worker_id]:
                    elem = next(seq_it)
                # we know we are not skipping this sequence, so
                # we perform any document loading that we deferred in the skipping process.
                elem["ids"] = torch.LongTensor(elem["ids"])
                subsequences = []
                # assemble the sequence form the SequenceFragment
                for doc, start, ln in elem["block"]:
                    # doc[0] can be (1) "padding", (2) a tensor of tokens,
                    # or (3) and index into self.dataset that hasn't been loaded yet.

                    # A padding tensor (<padding_value>, 0, length)
                    if doc[0] == "padding":
                        example = subsequences[-1]
                        if self.source_target:
                            example = example[0]
                        padding_tensor = example.new_full((ln,), self.padding_idx)
                        if self.source_target:
                            padding_tensor = (padding_tensor, padding_tensor)
                        subsequences.append(padding_tensor)
                    else:
                        # This single-element list is shared among all SequenceFragments that use
                        # the same document. We update the list to ensure we only
                        # ever tokenize the document once.
                        if isinstance(doc[0], int):
                            # an index into dataset that hasn't been loaded yet
                            # load it now (and for all other SequenceFragments where it hasn't been loaded yet)
                            doc[0] = self.dataset[doc[0]]
                        if self.source_target:
                            subsequences.append(
                                tuple(elem[start : start + ln] for elem in doc[0])
                            )
                        else:
                            subsequences.append(doc[0][start : start + ln])
                if self.source_target:
                    del elem["block"]
                    elem["src_block"] = torch.cat(tuple(s for s, t in subsequences))
                    elem["tgt_block"] = torch.cat(tuple(t for s, t in subsequences))
                else:
                    elem["block"] = torch.cat(subsequences)
                elem["skip_time"] = skip_time
                yield elem
        except StopIteration:
            return


def yield_single_sentences_pad_8(iterable, block_size, drop_last) -> Iterable[Sequence]:
    """Mimics sample-break-mode eos i.e. 1 example per sequence without any packing.
    When multiple examples are packed into a single sequence, example tokens would attend
    to tokens in neighbouring examples, which may be undesirable. This mode can
    avoid that. Since there is no packing, this mode is considerably slower.
    We round up the example length to a multiple of 8, pad to this length and
    return the example as is, without packing, truncating to block_size in cases of
    very long examples.
    """
    for idx, (tokens, document) in enumerate(iterable):
        cur_block = []
        cur_block_ids = []
        if tokens > block_size:
            # truncate right side
            # TODO: Enable left side truncation
            tokens = block_size

        cur_block.append((document, 0, tokens))

        # We round up to a multiple of 8 + 1, because later on
        # one element is removed for src/target tensor creation
        # which brings it back to a multiple of 8. block_size is
        # already passed with + 1 included.
        cur_block_remain = min(int(math.ceil(tokens / 8)) * 8 + 1, block_size)
        cur_block_remain -= tokens
        padding = (["padding"], 0, cur_block_remain)
        cur_block.append(padding)
        cur_block_ids.append(idx)
        yield {
            "ids": cur_block_ids,
            "block": cur_block,
        }


def yield_doc_blocks(iterable, block_size, drop_last) -> Iterable[Sequence]:
    """Mimics sample-break-mode complete"""
    cur_block = []
    cur_block_ids = []
    cur_block_remain = block_size
    for idx, (tokens, document) in enumerate(iterable):
        if tokens > block_size:
            # truncate right side
            tokens = block_size

        if tokens > cur_block_remain:
            padding = (["padding"], 0, cur_block_remain)
            cur_block.append(padding)
            yield {
                "ids": cur_block_ids,
                "block": cur_block,
            }

            cur_block = []
            cur_block_ids = []
            cur_block_remain = block_size

        cur_block.append((document, 0, tokens))
        cur_block_ids.append(idx)
        cur_block_remain -= tokens
        assert cur_block_remain >= 0
    if not drop_last and len(cur_block) > 0:
        if cur_block_remain > 0:
            padding = (["padding"], 0, cur_block_remain)
            cur_block.append(padding)
        yield {
            "ids": cur_block_ids,
            "block": cur_block,
        }


def yield_passthrough(iterable, block_size, drop_last) -> Iterable[Sequence]:
    for idx, (tokens, document) in enumerate(iterable):
        yield {
            "ids": [idx],
            "block": [(document, 0, tokens)],
        }


def yield_token_blocks(iterable, block_size, drop_last) -> Iterable[Sequence]:
    """Sample break mode = None. (Pre-Training default)."""
    cur_block = []
    cur_block_ids = []
    cur_block_remain = block_size
    for idx, (tokens, document) in enumerate(iterable):
        cur_block_ids.append(idx)
        item_offset = 0
        while tokens:
            num_to_take = min(tokens, cur_block_remain)
            cur_block.append((document, item_offset, num_to_take))
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
            cur_block.append((["padding"], 0, cur_block_remain))
        yield {
            "ids": cur_block_ids,
            "block": cur_block,
        }
