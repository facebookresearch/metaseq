# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from ctypes import addressof, memset, memmove
from multiprocessing import Array
import time

from metaseq.data.atomic import atomic_read, atomic_write, atomic_read_all


class AtomicArray:
    """
    A multi-process array array where the read and write operations are atomic.
    """

    def __init__(self, size):
        self.data = Array("i", size, lock=False)
        memset(addressof(self.data), 0, 4 * size)

    def __getitem__(self, idx):
        return atomic_read(addressof(self.data), idx)

    def __setitem__(self, idx, value):
        atomic_write(addressof(self.data), idx, value)

    def __len__(self):
        return len(self.data)

    def as_array(self):
        r = Array("i", len(self.data), lock=False)
        atomic_read_all(addressof(r), addressof(self.data), len(self.data))
        return r

    def __getstate__(self):
        return (len(self.data), bytes(self.as_array()))

    def __setstate__(self, state):
        l, b = state
        self.__init__(l)
        memmove(addressof(self.data), b, len(b))

    def from_tensor(self, t):
        memmove(addressof(self.data), t.data_ptr(), len(self.data) * 4)


def tree_flatten_instance(r, obj):
    if isinstance(obj, dict):
        ctors = [tree_flatten_instance(r, v) for v in obj.values()]
        keys = list(obj.keys())
        return lambda n: {k: v(n) for k, v in zip(keys, ctors)}
    elif isinstance(obj, (list, tuple)):
        t = type(obj)
        ctors = [tree_flatten_instance(r, v) for v in obj]
        return lambda n: t(ctor(n) for ctor in ctors)
    else:
        r.append(obj)
        return next


def tree_flatten(tree):
    r = []
    ctor = tree_flatten_instance(r, tree)
    return r, lambda ns: ctor(iter(ns))


def tree_map(fn, tree):
    vs, unflatten = tree_flatten(tree)
    return unflatten(fn(v) for v in vs)


class DeferredTensor:
    def __init__(self, size_or_value, ctor=None):
        if isinstance(size_or_value, int):
            self._size = size_or_value
            self.ctor = ctor
        else:
            self._value = size_or_value
            assert isinstance(self._value, torch.Tensor)
            assert len(self._value.shape) == 1, "can only defer 1-D tensors"
            self._size = self._value.shape[0]

    def realize(self):
        if hasattr(self, "ctor"):
            # print("REALIZING...")
            self._value = self.ctor()
            del self.ctor
            assert len(self._value.shape) == 1 and self._value.shape[0] == self._size
        return self._value

    def numel(self):
        return self._size

    @property
    def shape(self):
        return (self._size,)

    def __getitem__(self, s):
        if isinstance(s, slice) and len(self.shape) == 1:
            return SliceDeferredTensor(self, s)
        raise NotImplementedError("non-slice getitem")

    def new_full(self, size_tup, value):
        assert len(size_tup) == 1, "Unimplemented: multi-dim new_full"
        return DeferredTensor(
            size_tup[0], lambda: self.realize().new_full(size_tup, value)
        )

    @classmethod
    def __torch_function__(cls, fn, types, args, kwargs={}):
        if (
            fn is torch.cat
            and len(args) == 1
            and len(kwargs) == 0
            and all(len(x.shape) == 1 for x in args[0])
        ):
            new_size = sum(x.shape[0] for x in args[0])
            return DeferredTensor(
                new_size, lambda: torch.cat(tuple(x.realize() for x in args[0]))
            )
        raise NotImplementedError(f"Unimplemented: {args}, {kwargs}")


# optimization of slice of slice, because otherwise the tokenization code
# creates a really deep deferred tensor
# stack and then reaches max recursion
class SliceDeferredTensor(DeferredTensor):
    def __init__(self, to_slice, s):
        indices = s.indices(to_slice._size)
        new_size = len(range(*indices))
        super().__init__(new_size, lambda: to_slice.realize()[s])
        self.to_slice = to_slice
        self.indices = indices

    def __getitem__(self, s):
        orig_start, _, orig_step = self.indices
        start, end, step = s.indices(self._size)
        assert orig_step > 0 and step > 0
        return SliceDeferredTensor(
            self.to_slice, slice(orig_start + start, orig_start + end, orig_step * step)
        )


class _DeferredBase:
    def set_epoch(self, epoch):
        if hasattr(self.dataset, "set_epoch"):
            self.dataset.set_epoch(epoch)


class DeferredDataset(torch.utils.data.Dataset, _DeferredBase):
    """Generate deferred objects that might not be loaded by later stages in the data loader"""

    def __init__(self, dataset: torch.utils.data.Dataset, len_cache=None):
        super().__init__()
        self.dataset = dataset
        self.len_cache = (
            AtomicArray(len(self.dataset)) if len_cache is None else len_cache
        )
        self.enabled = True

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __getitem__(self, idx):
        if not self.enabled:
            return self.dataset[idx]
        assert idx >= 0 and idx < len(self.dataset)
        ln = self.len_cache[idx]
        if ln == 0:
            r = DeferredTensor(self.dataset[idx])
            self.len_cache[idx] = r._size
            return r
        else:
            return DeferredTensor(ln, lambda: self.dataset[idx])


class SkipDeferredDataset(torch.utils.data.IterableDataset, _DeferredBase):
    def __init__(self, dataset, to_skip: int):
        self.dataset = dataset
        self.to_skip = to_skip
        self.worker_offset = 0

    def __iter__(self):
        skip_time = 0
        t0 = time.time()
        if isinstance(self.to_skip, int):
            to_skip = self.to_skip
        else:
            info = torch.utils.data.get_worker_info()
            worker_id = (
                0
                if info is None
                else ((info.id + self.worker_offset) % info.num_workers)
            )
            to_skip = self.to_skip[worker_id]
        for i, elem in enumerate(self.dataset):
            if i >= to_skip:

                def to_tensor(x):
                    return x.realize() if isinstance(x, DeferredTensor) else x

                r = tree_map(to_tensor, elem)
                # inject timing information into output dictionary to
                # benchmark skip process
                if isinstance(r, dict):
                    r["skip_time"] = skip_time
                yield r
            elif i + 1 == to_skip:
                t1 = time.time()
                skip_time = t1 - t0
