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

from typing import Union, List, Iterable, Tuple, TypedDict, Literal

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
SequenceFragment = Tuple[
    List[Document], int, int # document, offset, length
]


class Sequence(TypedDict):
    block: List[SequenceFragment] # These are torch.cat'd together to get the whole sequence
    ids: List[int]


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
        drop_last (bool, optional): drop the last item (default: False)
        padding_idx (int, optional): index to use for padding symbols
            (required if *drop_last* is ``False``)
        shuffle_buffer_size (int, optional): buffer this many items and shuffle
            using the provided *seed*; default value is 1, so no shuffling is
            performed. This can be adjusted dynamically after initialization,
            but only before iteration has begun.
        seed (int, optional): seed for shuffling
        permute_documents (bool, optional): randomly permute the order the documents are read (default: True)
        to_skip (int, optional): skip the first to_skip sequences before iteration begins (Default: 0)
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
            AtomicArray(len(self.dataset)) if len_cache is None else len_cache
        )

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
                ln = self.len_cache[idx]
                if ln == 0:
                    # Cache miss: we don't know the number of tokens
                    # so we have to load and tokenize the document.
                    r = self.dataset[idx]
                    ln = r.shape[0]
                    self.len_cache[idx] = ln
                    yield (ln, [r])
                else:
                    # Cache hit: we know the number of tokens, so we can
                    # skip loading the document for now.

                    # We create a single-element list here, so that we can replace the single element
                    # with the real Tensor value the first time _any_ SentenceFragment needs the
                    # real data from this document.
                    yield (ln, [idx])

        block_itr = self.block_iterator(documents(), self.block_size, self.drop_last)

        # block_itr returns sequences from documents in order
        # we need to shuffle up this order to avoid having batches full of sequences from
        # one document. This is done with a "shuffle-buffer" that randomizes access
        baserng = np.random.default_rng(self.shuffle_buffer_seed)
        rngstate = None
        rngcount = 0

        def create_rng():
            nonlocal rngstate, rngcount
            while True:
                rngstate = baserng._bit_generator.__getstate__()
                # overhead to calling 'integers' is high,
                # so request random numbers in batches of 1024
                for x in baserng.integers(self.shuffle_buffer_size, size=1024):
                    rngcount += 1
                    yield x

        rng = None if self.shuffle_buffer_seed is None else iter(create_rng())

        buffer = []

        def get_next_item_and_replace_in_buffer(replacement_item: Sequence) -> Sequence:
            # return a random item from the buffer and replace with a new item
            if rng is not None:
                if len(buffer) == self.shuffle_buffer_size:
                    idx = next(rng)
                else:
                    if len(buffer) == self.shuffle_buffer_size - 1:
                        # when the buffer drains at the end we are asking
                        # for a smaller range of random numbers than we have batched.
                        # To match our previous behavior, reset the
                        # state of base rng to what it would have been
                        # in previous version of the code to generate
                        # these smaller numbers (only happens at the very end)
                        if rngstate is not None:
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
        for i, elem in enumerate(sequences()):
            if i >= to_skip:
                # we know we are not skipping this sequence, so
                # we perform any document loading that we deferred in the skipping process.
                elem["ids"] = torch.LongTensor(elem["ids"])
                subsequences = []
                # assemble the sequence form the SequenceFragment objects
                for doc, start, ln in elem["block"]:
                    # doc[0] can be (1) "padding", (2) a tensor of tokens,
                    # or (3) and index into self.dataset that hasn't been loaded yet.

                    # A padding tensor (<padding_value>, 0, length)
                    if doc[0] == "padding":
                        subsequences.append(
                            subsequences[-1].new_full((ln,), self.padding_idx)
                        )
                    else:
                        # This single-element list is shared among all SequenceFragments that use
                        # the same document. We update the list to ensure we only
                        # ever tokenize the document once.
                        if not isinstance(doc[0], torch.Tensor):
                            # an index into dataset that hasn't been loaded yet
                            # load it now (and for all other SequenceFragments where it hasn't been loaded yet)
                            doc[0] = self.dataset[doc[0]]
                        subsequences.append(doc[0][start : start + ln])
                elem["block"] = torch.cat(subsequences)
                elem["skip_time"] = skip_time
                yield elem
            elif i + 1 == to_skip:
                # for timing purposes record how long skipping took
                t1 = time.time()
                skip_time = t1 - t0


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
