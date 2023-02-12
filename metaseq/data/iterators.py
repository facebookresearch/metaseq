# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import logging
import math
import operator
import os
import queue
import time
from threading import Thread
from typing import Callable, Optional
import numpy as np
import torch
from metaseq.distributed import utils as distributed_utils

from metaseq.data import data_utils
from metaseq.data.document_to_sequence import DocumentToSequenceDataset
from ctypes import c_int, sizeof, memmove, addressof

logger = logging.getLogger(__name__)

# Object used by _background_consumer to signal the source is exhausted
# to the main thread.
_sentinel = object()


class CountingIterator(object):
    """Wrapper around an iterable that maintains the iteration count.

    Args:
        iterable (iterable): iterable to wrap
        start (int): starting iteration count. Note that this doesn't
            actually advance the iterator.
        total (int): override the iterator length returned by
            ``__len__``. This can be used to truncate *iterator*.

    Attributes:
        n (int): number of elements consumed from this iterator
    """

    def __init__(self, iterable, start=None, total=None):
        self.iterable = iterable
        self.itr = iter(self)

        if start is None:
            self.n = getattr(iterable, "n", 0)
        else:
            self.n = start

        if total is None:
            self.total = self.n + len(iterable)
        else:
            self.total = total

    def __len__(self):
        return self.total

    def __iter__(self):
        for x in self.iterable:
            if self.n >= self.total:
                raise RuntimeError(
                    "Mismatch between actual and expected iterable length. "
                    "This may be caused by resuming training from a checkpoint using "
                    "a different number of GPUs, in which case you can try the "
                    "--reset-dataloader option. Alternatively you may have a train or "
                    "validation set that is smaller than the number of GPUs. If none "
                    "of these apply, please report this to the metaseq developers."
                )
            self.n += 1
            yield x

    def __next__(self):
        return next(self.itr)

    def has_next(self):
        """Whether the iterator has been exhausted."""
        return self.n < len(self)

    def skip(self, num_to_skip):
        """Fast-forward the iterator by skipping *num_to_skip* elements."""
        next(itertools.islice(self.itr, num_to_skip, num_to_skip), None)
        return self

    def take(self, n):
        """
        Truncates the iterator to n elements at most.
        """
        self.total = min(self.total, n)

        # Propagate this change to the underlying iterator
        # Only take after what we have already consumed (i.e. after restarting
        # from checkpoint mid epoch, we have to subtract self.n which is the
        # starting point)
        #
        # This to maintain the invariant self.total = self.n + len(iterable),
        # before calling __next__ or __iter__
        propagated_take = max(n - self.n, 0)
        if hasattr(self.iterable, "take"):
            self.iterable.take(propagated_take)
        else:
            self.iterable = itertools.islice(self.iterable, propagated_take)


class StreamingCountingIterator(object):
    """Wrapper around an iterable that maintains the iteration count.

    Args:
        iterable (iterable): iterable to wrap

    Attributes:
        n (int): number of elements consumed from this iterator
    """

    def __init__(self, iterable, num_workers, batch_size, num_shards):
        try:
            import more_itertools
        except ImportError:
            raise ImportError(
                "more_itertools is required for streaming iterators; "
                "please install with: pip install more_itertools"
            )
        self._peekable_itr = more_itertools.peekable(iterable)

        self.num_workers = 1 if num_workers == 0 else num_workers
        self.batch_size = batch_size
        self.num_shards = num_shards

        self.n = 0
        self.next_worker = 0
        self.sequences_consumed = [0 for _ in range(self.num_workers)]
        self.worker_offset = 0

    def __iter__(self):
        return self

    def __next__(self):
        worker_id, r = next(self._peekable_itr)
        worker_id = (worker_id + self.worker_offset) % self.num_workers
        self.sequences_consumed[worker_id] += self.batch_size * self.num_shards
        self.next_worker = (worker_id + 1) % self.num_workers
        self.n += 1
        return r

    def __len__(self):
        return 0

    def has_next(self):
        return bool(self._peekable_itr)  # whether peekable has items


class EpochBatchIterating(object):
    def __len__(self) -> int:
        raise NotImplementedError

    @property
    def next_epoch_idx(self):
        raise NotImplementedError

    def next_epoch_itr(
        self, shuffle=True, fix_batches_to_gpus=False, set_dataset_epoch=True
    ):
        """Return a new iterator over the dataset.

        Args:
            shuffle (bool, optional): shuffle batches before returning the
                iterator (default: True).
            fix_batches_to_gpus (bool, optional): ensure that batches are always
                allocated to the same shards across epochs. Requires
                that :attr:`dataset` supports prefetching (default: False).
            set_dataset_epoch (bool, optional): update the wrapped Dataset with
                the new epoch number (default: True).
        """
        raise NotImplementedError

    def end_of_epoch(self) -> bool:
        """Returns whether the most recent epoch iterator has been exhausted"""
        raise NotImplementedError

    @property
    def iterations_in_epoch(self) -> int:
        """The number of consumed batches in the current epoch."""
        raise NotImplementedError

    def state_dict(self):
        """Returns a dictionary containing a whole state of the iterator."""
        raise NotImplementedError

    def load_state_dict(self, state_dict):
        """Copies the state of the iterator from the given *state_dict*."""
        raise NotImplementedError

    @property
    def first_batch(self):
        return "DUMMY"


class _CollateWithWorkerID:
    def __init__(self, collate_fn):
        self.collate_fn = collate_fn

    def __call__(self, items):
        r = self.collate_fn(items)
        worker_info = torch.utils.data.get_worker_info()
        return (worker_info.id if worker_info else 0, r)


class StreamingEpochBatchIterator(EpochBatchIterating):
    """A steaming-style iterator over a :class:`torch.utils.data.IterableDataset`.

    Args:
        dataset (~torch.utils.data.Dataset): dataset from which to load the data
        batch_size (int): number of items in each batch
        collate_fn (callable): merges a list of samples to form a mini-batch
        drop_last (bool): whether to skip the last batch, in cases where it
            would be incomplete (i.e., have fewer than *batch_size* items)
        num_workers (int, optional): how many subprocesses to use for data
            loading. 0 means the data will be loaded in the main process
            (default: 0).
        epoch (int, optional): the epoch to start the iterator from
            (default: 1).
    """

    def __init__(
        self,
        dataset: torch.utils.data.IterableDataset,
        batch_size: int,
        collate_fn: Callable,
        drop_last: bool,
        num_workers: int = 0,
        epoch: int = 1,
        num_shards: int = 1,
    ):
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.collate_fn = collate_fn
        self.drop_last = drop_last
        self.num_workers = num_workers
        self.epoch = max(epoch, 1)  # we use 1-based indexing for epochs
        self.num_shards = num_shards
        assert isinstance(dataset, torch.utils.data.IterableDataset)

        self._itr: Optional[StreamingCountingIterator] = None
        self.worker_offset = 0

    @property
    def next_epoch_idx(self):
        """Return the epoch index after *next_epoch_itr* is called."""
        if self._itr is not None and self.end_of_epoch():
            return self.epoch + 1
        else:
            return self.epoch

    def next_epoch_itr(self, **kwargs):
        """
        Return a new iterator over the dataset.

        In case :func:`load_state_dict` has been called recently, this will
        return the loaded iterator.
        """
        self.epoch = self.next_epoch_idx
        if hasattr(self.dataset, "set_epoch"):
            self.dataset.set_epoch(self.epoch)
        if self._itr is None or self.end_of_epoch():
            self._itr = self._get_iterator_for_epoch(self.epoch)
        return self._itr

    def end_of_epoch(self) -> bool:
        """Returns whether the most recent epoch iterator has been exhausted"""
        return not self._itr.has_next()

    @property
    def iterations_in_epoch(self) -> int:
        """The number of consumed batches in the current epoch."""
        return self._itr.n

    def state_dict(self):
        """Returns a dictionary containing a whole state of the iterator."""
        if self.end_of_epoch():
            # small optimization: we advance the epoch before saving, so that
            # when loading later we don't end up fast-forwarding the iterator
            epoch = self.epoch + 1
            sequences_consumed = [0 for _ in range(self.num_workers)]
            n = 0
            next_worker = 0
        else:
            epoch = self.epoch
            sequences_consumed = self._itr.sequences_consumed
            n = self._itr.n
            next_worker = self._itr.next_worker

        dataset = self.dataset
        while not isinstance(dataset, DocumentToSequenceDataset):
            dataset = dataset.dataset
        logger.debug(
            f"Saving state_dict so we can skip workers quickly: {len(dataset.len_cache.data)} "
            f"entries in tokenization_cache, {sequences_consumed} sequences consumed per worker, iteration {n}"
        )
        return {
            "epoch": epoch,
            "sequences_consumed": sequences_consumed,
            "tokenization_cache": dataset.len_cache
            if distributed_utils.get_global_rank() == 0
            else None,
            "n": n,
            "next_worker": next_worker,
        }

    def load_state_dict(self, state_dict):
        """Copies the state of the iterator from the given *state_dict*."""
        self.epoch = state_dict["epoch"]
        if hasattr(self.dataset, "set_epoch"):
            self.dataset.set_epoch(self.epoch)

        # must be set before _get_iterator_for_epoch otherwise the datasets in the workers
        # will not be copied with the right state
        if (
            "sequences_consumed" in state_dict
            and max(state_dict["sequences_consumed"]) > 0
        ):
            sequences_consumed = state_dict["sequences_consumed"]
            n = state_dict["n"]
            next_worker = state_dict["next_worker"]

            logger.info(f"Skipping {sequences_consumed} sequences in each worker...")
            num_workers = 1 if self.num_workers == 0 else self.num_workers
            assert (
                len(sequences_consumed) == num_workers
            ), "changing the number of workers in the middle of a shard changes the order the data will be loaded in"
            dataset = self.dataset
            while not isinstance(dataset, DocumentToSequenceDataset):
                dataset = dataset.dataset
            dataset.to_skip = sequences_consumed
            dataset.worker_offset = next_worker
            global_group = distributed_utils.get_global_group()
            if global_group is None:
                dataset.len_cache = state_dict["tokenization_cache"]
            else:
                if distributed_utils.get_global_rank() == 0:
                    dataset.len_cache = state_dict["tokenization_cache"]
                    b, _ = dataset.len_cache.__getstate__()
                    len_tensor = torch.frombuffer(bytearray(b), dtype=torch.int8).cuda()
                    distributed_utils.broadcast(len_tensor, 0, global_group)
                else:
                    n_bytes = sizeof(c_int) * len(dataset.dataset)
                    len_tensor = torch.empty(n_bytes, dtype=torch.int8, device="cuda")
                    distributed_utils.broadcast(len_tensor, 0, global_group)
                    len_tensor = len_tensor.cpu()
                    memmove(
                        addressof(dataset.len_cache.data),
                        len_tensor.data_ptr(),
                        n_bytes,
                    )

            self._itr = self._get_iterator_for_epoch(self.epoch)
            self._itr.n = n

            if True:
                # Epilogue bug fixup
                # Warning: this fix is not correct for the last ~1% of an epoch, but it only needs to be
                # applied once earlier in the epoch to fix any data loaders with incorrect data.
                num_workers = self._itr.num_workers
                batch_size = self._itr.batch_size * self._itr.num_shards
                if sum(sequences_consumed) != n * batch_size:
                    logger.warning(
                        f"{distributed_utils.get_global_rank()}: Sequences appear corrupted: "
                        f"{n}*{batch_size} != sum({sequences_consumed})"
                    )
                    each, left = divmod(n, num_workers)
                    sequences_consumed = [
                        batch_size * (each + (1 if i < left else 0))
                        for i in range(num_workers)
                    ]
                assert sum(sequences_consumed) == n * batch_size

            self._itr.sequences_consumed = sequences_consumed
            self._itr.next_worker = next_worker
            self._itr.worker_offset = next_worker
        else:
            self._itr = self._get_iterator_for_epoch(self.epoch)
            # checkpoint from before sequences_consumed was added, slow fast forward...
            if (
                "iterations_in_epoch" in state_dict
                and state_dict["iterations_in_epoch"] > 0
            ):
                # fast-forward epoch iterator
                itr_pos = state_dict["iterations_in_epoch"]
                logger.info(
                    f"Fast-forwarding dataloader by {itr_pos} batches using slower logic because "
                    "checkpoint does not have a tokenization_cache..."
                )
                t0 = time.time()
                next(itertools.islice(self._itr, itr_pos, itr_pos), None)
                t1 = time.time()
                logger.info(f"done fast-forwarding dataloader in {t1 - t0:.1f} seconds")

    def _get_iterator_for_epoch(self, epoch, offset=0):
        if self.num_workers > 0:
            os.environ["PYTHONWARNINGS"] = "ignore:semaphore_tracker:UserWarning"

        itr = torch.utils.data.DataLoader(
            dataset=self.dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=_CollateWithWorkerID(self.collate_fn),
            pin_memory=True,
            drop_last=self.drop_last,
            worker_init_fn=getattr(self.dataset, "worker_init_fn", None),
        )

        itr = StreamingCountingIterator(
            itr, self.num_workers, self.batch_size, self.num_shards
        )

        return itr


class EpochBatchIterator(EpochBatchIterating):
    """A multi-epoch iterator over a :class:`torch.utils.data.Dataset`.

    Compared to :class:`torch.utils.data.DataLoader`, this iterator:

    - can be reused across multiple epochs with the :func:`next_epoch_itr`
      method (optionally shuffled between epochs)
    - can be serialized/deserialized with the :func:`state_dict` and
      :func:`load_state_dict` methods
    - supports sharding with the *num_shards* and *shard_id* arguments

    Args:
        dataset (~torch.utils.data.Dataset): dataset from which to load the data
        collate_fn (callable): merges a list of samples to form a mini-batch
        batch_sampler (~torch.utils.data.Sampler or a callable): an iterator over batches of
            indices, or a callable to create such an iterator (~torch.utils.data.Sampler).
            A callable batch_sampler will be called for each epoch to enable per epoch dynamic
            batch iterators defined by this callable batch_sampler.
        seed (int, optional): seed for random number generator for
            reproducibility (default: 1).
        num_shards (int, optional): shard the data iterator into N
            shards (default: 1).
        shard_id (int, optional): which shard of the data iterator to
            return (default: 0).
        num_workers (int, optional): how many subprocesses to use for data
            loading. 0 means the data will be loaded in the main process
            (default: 0).
        epoch (int, optional): the epoch to start the iterator from
            (default: 1).
        buffer_size (int, optional): the number of batches to keep ready in the
            queue. Helps speeding up dataloading. When buffer_size is zero, the
            default torch.utils.data.DataLoader preloading is used.
        timeout (int, optional): if positive, the timeout value for collecting a batch
            from workers. Should always be non-negative (default: ``0``).
        disable_shuffling (bool, optional): force disable shuffling
            (default: ``False``).
        skip_remainder_batch (bool, optional): if set, discard the last batch in an epoch
            for the sake of training stability, as the last batch is usually smaller than
                local_batch_size * distributed_word_size (default: ``True``).
    """

    def __init__(
        self,
        dataset,
        collate_fn,
        batch_sampler,
        seed=1,
        num_shards=1,
        shard_id=0,
        num_workers=0,
        epoch=1,
        buffer_size=0,
        timeout=0,
        disable_shuffling=False,
        skip_remainder_batch=True,
    ):
        assert isinstance(dataset, torch.utils.data.Dataset)
        self.dataset = dataset
        self.collate_fn = collate_fn
        self.batch_sampler = batch_sampler
        self._frozen_batches = (
            tuple(batch_sampler) if not callable(batch_sampler) else None
        )
        self.seed = seed
        self.num_shards = num_shards
        self.shard_id = shard_id
        self.num_workers = num_workers
        # This upper limit here is to prevent people from abusing this feature
        # in a shared computing environment.
        self.buffer_size = min(buffer_size, 20)
        self.timeout = timeout
        self.disable_shuffling = disable_shuffling
        self.skip_remainder_batch = skip_remainder_batch

        self.epoch = max(epoch, 1)  # we use 1-based indexing for epochs
        self.shuffle = not disable_shuffling
        self._cur_epoch_itr = None
        self._next_epoch_itr = None
        self._supports_prefetch = getattr(dataset, "supports_prefetch", False)

    @property
    def frozen_batches(self):
        if self._frozen_batches is None:
            self._frozen_batches = tuple(self.batch_sampler(self.dataset, self.epoch))
        return self._frozen_batches

    @property
    def first_batch(self):
        if len(self.frozen_batches) == 0:
            raise Exception(
                "The dataset is empty. This could indicate "
                "that all elements in the dataset have been skipped. "
                "Try increasing the max number of allowed tokens or using "
                "a larger dataset."
            )

        if getattr(self.dataset, "supports_fetch_outside_dataloader", True):
            return self.collate_fn([self.dataset[i] for i in self.frozen_batches[0]])
        else:
            return "DUMMY"

    def __len__(self):
        return int(math.ceil(len(self.frozen_batches) / float(self.num_shards)))

    @property
    def n(self):
        return self.iterations_in_epoch

    @property
    def next_epoch_idx(self):
        """Return the epoch index after *next_epoch_itr* is called."""
        if self._next_epoch_itr is not None:
            return self.epoch
        elif self._cur_epoch_itr is not None and self.end_of_epoch():
            return self.epoch + 1
        else:
            return self.epoch

    def next_epoch_itr(
        self, shuffle=True, fix_batches_to_gpus=False, set_dataset_epoch=True
    ):
        """Return a new iterator over the dataset.

        Args:
            shuffle (bool, optional): shuffle batches before returning the
                iterator (default: True).
            fix_batches_to_gpus (bool, optional): ensure that batches are always
                allocated to the same shards across epochs. Requires
                that :attr:`dataset` supports prefetching (default: False).
            set_dataset_epoch (bool, optional): update the wrapped Dataset with
                the new epoch number (default: True).
        """
        if self.disable_shuffling:
            shuffle = False
        self.epoch = self.next_epoch_idx
        if set_dataset_epoch and hasattr(self.dataset, "set_epoch"):
            self.dataset.set_epoch(self.epoch)
        if self._next_epoch_itr is not None:
            self._cur_epoch_itr = self._next_epoch_itr
            self._next_epoch_itr = None
        else:
            if callable(self.batch_sampler):
                # reset _frozen_batches to refresh the next epoch
                self._frozen_batches = None
            self._cur_epoch_itr = self._get_iterator_for_epoch(
                self.epoch,
                shuffle,
                fix_batches_to_gpus=fix_batches_to_gpus,
            )
        self.shuffle = shuffle
        return self._cur_epoch_itr

    def end_of_epoch(self) -> bool:
        """Returns whether the most recent epoch iterator has been exhausted"""
        return not self._cur_epoch_itr.has_next()

    @property
    def iterations_in_epoch(self):
        """The number of consumed batches in the current epoch."""
        if self._cur_epoch_itr is not None:
            return self._cur_epoch_itr.n
        elif self._next_epoch_itr is not None:
            return self._next_epoch_itr.n
        return 0

    def state_dict(self):
        """Returns a dictionary containing a whole state of the iterator."""
        if self.end_of_epoch():
            epoch = self.epoch + 1
            iter_in_epoch = 0
        else:
            epoch = self.epoch
            iter_in_epoch = self.iterations_in_epoch
        return {
            "version": 2,
            "epoch": epoch,
            "iterations_in_epoch": iter_in_epoch,
            "shuffle": self.shuffle,
        }

    def load_state_dict(self, state_dict):
        """Copies the state of the iterator from the given *state_dict*."""
        self.epoch = state_dict["epoch"]
        itr_pos = state_dict.get("iterations_in_epoch", 0)
        version = state_dict.get("version", 1)
        if itr_pos > 0:
            # fast-forward epoch iterator
            self._next_epoch_itr = self._get_iterator_for_epoch(
                self.epoch,
                shuffle=state_dict.get("shuffle", True),
                offset=itr_pos,
            )
            if self._next_epoch_itr is None:
                if version == 1:
                    # legacy behavior: we finished the epoch, increment epoch counter
                    self.epoch += 1
                else:
                    raise RuntimeError(
                        "Cannot resume training due to dataloader mismatch, please "
                        "report this to the metaseq developers. You can relaunch "
                        "training with `--reset-dataloader` and it should work."
                    )
        else:
            self._next_epoch_itr = None

    def _get_iterator_for_epoch(
        self, epoch, shuffle, fix_batches_to_gpus=False, offset=0
    ):
        def shuffle_batches(batches, seed):
            with data_utils.numpy_seed(seed):
                np.random.shuffle(batches)
            return batches

        if self._supports_prefetch:
            batches = self.frozen_batches

            if shuffle and not fix_batches_to_gpus:
                batches = shuffle_batches(list(batches), self.seed + epoch)

            batches = list(
                ShardedIterator(batches, self.num_shards, self.shard_id, fill_value=[])
            )
            self.dataset.prefetch([i for s in batches for i in s])

            if shuffle and fix_batches_to_gpus:
                batches = shuffle_batches(batches, self.seed + epoch + self.shard_id)
        else:
            if shuffle:
                batches = shuffle_batches(list(self.frozen_batches), self.seed + epoch)
            else:
                batches = self.frozen_batches
            batches = list(
                ShardedIterator(batches, self.num_shards, self.shard_id, fill_value=[])
            )

        if offset > 0 and offset >= len(batches):
            return None

        if self.num_workers > 0:
            os.environ["PYTHONWARNINGS"] = "ignore:semaphore_tracker:UserWarning"

        # Create data loader
        itr = torch.utils.data.DataLoader(
            self.dataset,
            collate_fn=self.collate_fn,
            batch_sampler=batches[offset:],
            num_workers=self.num_workers,
            timeout=self.timeout,
        )

        # Wrap with a BufferedIterator if needed
        if self.buffer_size > 0:
            itr = BufferedIterator(self.buffer_size, itr)

        # Wrap with CountingIterator
        itr = CountingIterator(itr, start=offset)

        if self.skip_remainder_batch:
            # TODO: Below is a lazy implementation which discard the final batch regardless
            # of whether it is a full batch or not.
            total_num_itrs = len(batches) - 1
            itr.take(total_num_itrs)
            logger.info(f"skip final residual batch, total_num_itrs = {total_num_itrs}")

        return itr


class GroupedIterator(CountingIterator):
    """Wrapper around an iterable that returns groups (chunks) of items.

    Args:
        iterable (iterable): iterable to wrap
        chunk_size (int): size of each chunk
        skip_remainder_batch (bool, optional): if set, discard the last grouped batch in
          each training epoch, as the last grouped batch is usually smaller than
                local_batch_size * distributed_word_size * chunk_size (default: ``True``).
    Attributes:
        n (int): number of elements consumed from this iterator
    """

    def __init__(self, iterable, chunk_size, skip_remainder_batch=True):
        if skip_remainder_batch:
            total_num_itrs = int(math.floor(len(iterable) / float(chunk_size)))
            logger.info(
                f"skip final residual batch, grouped total_num_itrs = {total_num_itrs}"
            )
        else:
            total_num_itrs = int(math.ceil(len(iterable) / float(chunk_size)))
            logger.info(f"grouped total_num_itrs = {total_num_itrs}")

        itr = _chunk_iterator(iterable, chunk_size, skip_remainder_batch)
        super().__init__(
            itr,
            start=int(math.ceil(getattr(iterable, "n", 0) / float(chunk_size))),
            total=total_num_itrs,
        )
        self.chunk_size = chunk_size

        if skip_remainder_batch:
            self.take(total_num_itrs)
            # TODO: [Hack] Here the grouped iterator modifies the base iterator size so that
            # training can move into the next epoch once the grouped iterator is exhausted.
            # Double-check this implementation in case unexpected behavior occurs.
            iterable.take(total_num_itrs * chunk_size)


def _chunk_iterator(itr, chunk_size, skip_remainder_batch=True):
    chunk = []
    for x in itr:
        chunk.append(x)
        if len(chunk) == chunk_size:
            yield chunk
            chunk = []
    if not skip_remainder_batch and len(chunk) > 0:
        yield chunk


class ShardedIterator(CountingIterator):
    """A sharded wrapper around an iterable, padded to length.

    Args:
        iterable (iterable): iterable to wrap
        num_shards (int): number of shards to split the iterable into
        shard_id (int): which shard to iterator over
        fill_value (Any, optional): padding value when the iterable doesn't
            evenly divide *num_shards* (default: None).

    Attributes:
        n (int): number of elements consumed from this iterator
    """

    def __init__(self, iterable, num_shards, shard_id, fill_value=None):
        if shard_id < 0 or shard_id >= num_shards:
            raise ValueError("shard_id must be between 0 and num_shards")
        sharded_len = int(math.ceil(len(iterable) / float(num_shards)))
        itr = map(
            operator.itemgetter(1),
            itertools.zip_longest(
                range(sharded_len),
                itertools.islice(iterable, shard_id, len(iterable), num_shards),
                fillvalue=fill_value,
            ),
        )
        super().__init__(
            itr,
            start=int(math.ceil(getattr(iterable, "n", 0) / float(num_shards))),
            total=sharded_len,
        )


class BackgroundConsumer(Thread):
    def __init__(self, queue, source, max_len):
        Thread.__init__(self)

        self._queue = queue
        self._source = source
        self._max_len = max_len
        self.count = 0

    def run(self):
        try:
            for item in self._source:
                self._queue.put(item)

                # Stop if we reached the maximum length
                self.count += 1
                if self._max_len is not None and self.count >= self._max_len:
                    break

            # Signal the consumer we are done.
            self._queue.put(_sentinel)
        except Exception as e:
            self._queue.put(e)


class BufferedIterator(object):
    def __init__(self, size, iterable):
        self._queue = queue.Queue(size)
        self._iterable = iterable
        self._consumer = None

        self.start_time = time.time()
        self.warning_time = None

        self.total = len(iterable)

    def _create_consumer(self):
        self._consumer = BackgroundConsumer(
            self._queue,
            self._iterable,
            self.total,
        )
        self._consumer.daemon = True
        self._consumer.start()

    def __iter__(self):
        return self

    def __len__(self):
        return self.total

    def take(self, n):
        self.total = min(self.total, n)

        # Propagate this change to the underlying iterator
        if hasattr(self._iterable, "take"):
            self._iterable.take(n)

    def __next__(self):
        # Create consumer if not created yet
        if self._consumer is None:
            self._create_consumer()

        # Notify the user if there is a data loading bottleneck
        if self._queue.qsize() < min(2, max(1, self._queue.maxsize // 2)):
            if time.time() - self.start_time > 5 * 60:
                if (
                    self.warning_time is None
                    or time.time() - self.warning_time > 15 * 60
                ):
                    logger.debug(
                        "Data loading buffer is empty or nearly empty. This may "
                        "indicate a data loading bottleneck, and increasing the "
                        "number of workers (--num-workers) may help."
                    )
                    self.warning_time = time.time()

        # Get next example
        item = self._queue.get(True)
        if isinstance(item, Exception):
            raise item
        if item is _sentinel:
            raise StopIteration()
        return item
