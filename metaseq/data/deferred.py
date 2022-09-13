
import os
import torch
from torch.utils.cpp_extension import load
from ctypes import addressof, memset
from multiprocessing import Array, Process
from collections import namedtuple
from typing import NamedTuple


deferred_c_src = f'{os.path.dirname(os.path.abspath(__file__))}/deferred.cpp'
deferred_c = load('deferred', deferred_c_src)
atomic_read = deferred_c.atomic_read
atomic_write = deferred_c.atomic_write
atomic_read_all = deferred_c.atomic_read_all

from tqdm import tqdm

class AtomicArray:
    """
    A multi-process array array where the read and write operations are atomic.
    """
    def __init__(self, size):
        self.data = Array('i', size, lock=False)
        memset(addressof(self.data), 0, 4*size)

    def __getitem__(self, idx):
        return atomic_read(addressof(self.data), idx)

    def __setitem__(self, idx, value):
        atomic_write(addressof(self.data), idx, value)

    def __len__(self):
        return len(self.data)

    def as_array(self):
        r = Array('i', len(self.data), lock=False)
        atomic_read_all(addressof(r), addressof(self.data), len(self.data))
        return r


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

class Deferred:
    pass

class DeferredTensor(Deferred):
    def __init__(self, size_or_value, ctor=None):
        if not isinstance(size_or_value, torch.Tensor):
            self._size = size_or_value
            self.ctor = ctor
        else:
            self._value = size_or_value
            assert isinstance(self._value, torch.Tensor)
            self._size = tuple(self._value.shape)
        assert isinstance(self._size, tuple)
    def realize(self):
        if hasattr(self, 'ctor'):
            # print("REALIZING...")
            self._value = self.ctor()
            del self.ctor
            assert tuple(self._value.shape) == self._size
        return self._value

    def numel(self):
        p = 1
        for x in self.shape:
            p *= x
        return p

    @property
    def shape(self):
        return self._size

    def __getitem__(self, s):
        if isinstance(s, slice) and len(self.shape) == 1:
            return SliceDeferredTensor(self, s)
        raise NotImplementedError('non-slice getitem')

    def __torch_function__(self, fn, types, args, kwargs={}):
        if fn is torch.cat and len(args) == 1 and len(kwargs) == 0 and all(len(x.shape) == 1 for x in args[0]):
            new_size = (sum(x.shape[0] for x in args[0]),)
            return DeferredTensor(new_size, lambda: torch.cat(tuple(x.realize() for x in args[0])))
        raise NotImplementedError(f'Unimplemented: {args}, {kwargs}')

# optimization of slice of slice, because otherwise the tokenization code creates a really deep deferred tensor
# stack and then reaches max recursion
class SliceDeferredTensor(DeferredTensor):
    def __init__(self, to_slice, s):
        indices = s.indices(to_slice.shape[0])
        new_size = len(range(*indices))
        super().__init__((new_size,), lambda: to_slice.realize()[s])
        self.to_slice = to_slice
        self.indices = indices

    def __getitem__(self, s):
        orig_start, _, orig_step = self.indices
        start, end, step = s.indices(self._size[0])
        assert orig_step > 0 and step > 0
        return SliceDeferredTensor(self.to_slice, slice(orig_start + start, orig_start + end, orig_step * step))

class DeferredDataset(torch.utils.data.Dataset):
    """Generate deferred objects that might not be loaded by later stages in the data loader

    """

    def __init__(self, dataset: torch.utils.data.Dataset, len_cache=None, deferred_type=DeferredTensor):
        super().__init__()
        self.dataset = dataset
        self.deferred_type = deferred_type
        self.len_cache = AtomicArray(len(self.dataset)) if len_cache is None else len_cache
        self.enabled = True

    def __len__(self):
        return len(self.dataset)

    def _progress_bar(self):
        # just to viz how far into the the dataset we get while running
        worker_info = torch.utils.data.get_worker_info()
        if worker_info and worker_info.id == 0:
            # put a hacky progress bar here
            if not hasattr(self, 't'):
                self.t = tqdm(total=len(self.dataset))
            self.t.update(1)


    def __getitem__(self, idx):
        self._progress_bar()
        if not self.enabled:
            return self.dataset[idx]
        assert idx >= 0 and idx < len(self.dataset)
        l = self.len_cache[idx]
        if l == 0:
            # print("MISS ", idx)
            r = self.deferred_type(self.dataset[idx])
            assert len(r._size) == 1, "can only cache size of 1-dim tensors..."
            self.len_cache[idx] = r._size[0]
            return r
        # print("HIT ", idx)
        return self.deferred_type((l,), lambda: self.dataset[idx])

class SkipDeferredDataset(torch.utils.data.IterableDataset):
    def __init__(self, dataset, to_skip: int):
        self.dataset = dataset
        self.to_skip = to_skip

        self.batch_size  = 0 # hack to know how many None to return to get timing information

        self.enabled = True

    def __iter__(self):
        for i, elem in enumerate(self.dataset):
            if i >= self.to_skip:
                yield tree_map(lambda x: x.realize() if isinstance(x, Deferred) else x, elem)
            else:
                # the real version will not do this, but we need to return a sentinel to the main
                # process to stop timing of the skip process for benchmarking
                # if this is the last one we will skip, return enough None to fill an entire batch,
                if i + 1 == self.to_skip:
                    for _ in range(self.batch_size):
                        yield None # dummy return to time the skip process

