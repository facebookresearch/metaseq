# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

try:
    from collections.abc import Iterable
except ImportError:
    from collections import Iterable
import contextlib
import itertools
import logging
import os
import re
import warnings
from typing import Optional, Tuple

import numpy as np
import torch

from metaseq import utils
from metaseq.file_io import PathManager

logger = logging.getLogger(__name__)


def infer_language_pair(path):
    """Infer language pair from filename: <split>.<lang1>-<lang2>.(...).idx"""
    src, dst = None, None
    for filename in PathManager.ls(path):
        parts = filename.split(".")
        if len(parts) >= 3 and len(parts[1].split("-")) == 2:
            return parts[1].split("-")
    return src, dst


def collate_tokens(
    values,
    pad_idx,
    eos_idx=None,
    left_pad=False,
    move_eos_to_beginning=False,
    pad_to_length=None,
    pad_to_multiple=1,
    pad_to_bsz=None,
):
    """Convert a list of 1d tensors into a padded 2d tensor."""
    size = max(v.size(0) for v in values)
    size = size if pad_to_length is None else max(size, pad_to_length)
    if pad_to_multiple != 1 and size % pad_to_multiple != 0:
        size = int(((size - 0.1) // pad_to_multiple + 1) * pad_to_multiple)

    batch_size = len(values) if pad_to_bsz is None else max(len(values), pad_to_bsz)
    res = values[0].new(batch_size, size).fill_(pad_idx)

    def copy_tensor(src, dst):
        assert dst.numel() == src.numel()
        if move_eos_to_beginning:
            if eos_idx is None:
                # if no eos_idx is specified, then use the last token in src
                dst[0] = src[-1]
            else:
                dst[0] = eos_idx
            dst[1:] = src[:-1]
        else:
            dst.copy_(src)

    for i, v in enumerate(values):
        copy_tensor(v, res[i][size - len(v) :] if left_pad else res[i][: len(v)])
    return res


def load_indexed_dataset(
    path, dictionary=None, dataset_impl=None, combine=False, default="cached"
):
    """A helper function for loading indexed datasets.

    Args:
        path (str): path to indexed dataset (e.g., 'data-bin/train')
        dictionary (~metaseq.data.Dictionary): data dictionary
        dataset_impl (str, optional): which dataset implementation to use. If
            not provided, it will be inferred automatically. For legacy indexed
            data we use the 'cached' implementation by default.
        combine (bool, optional): automatically load and combine multiple
            datasets. For example, if *path* is 'data-bin/train', then we will
            combine 'data-bin/train', 'data-bin/train1', ... and return a
            single ConcatDataset instance.
    """
    import metaseq.data.indexed_dataset as indexed_dataset
    from metaseq.data.concat_dataset import ConcatDataset

    datasets = []
    # for k in itertools.count():
    # change for testing
    for k in range(0,2):
        path_k = path + (str(k) if k > 0 else "")
        path_k = indexed_dataset.get_indexed_dataset_to_local(path_k)

        dataset_impl_k = dataset_impl
        if dataset_impl_k is None:
            dataset_impl_k = indexed_dataset.infer_dataset_impl(path_k)
        dataset = indexed_dataset.make_dataset(
            path_k,
            impl=dataset_impl_k or default,
            fix_lua_indexing=True,
            dictionary=dictionary,
        )
        if dataset is None:
            break
        logger.info("loaded {:,} examples from: {}".format(len(dataset), path_k))
        datasets.append(dataset)
        if not combine:
            break
    if len(datasets) == 0:
        return None
    elif len(datasets) == 1:
        return datasets[0]
    else:
        return ConcatDataset(datasets)


@contextlib.contextmanager
def numpy_seed(seed, *addl_seeds):
    """Context manager which seeds the NumPy PRNG with the specified seed and
    restores the state afterward"""
    if seed is None:
        yield
        return
    if len(addl_seeds) > 0:
        seed = int(hash((seed, *addl_seeds)) % 1e6)
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def collect_filtered(function, iterable, filtered):
    """
    Similar to :func:`filter` but collects filtered elements in ``filtered``.

    Args:
        function (callable): function that returns ``False`` for elements that
            should be filtered
        iterable (iterable): iterable to filter
        filtered (list): list to store filtered elements
    """
    for el in iterable:
        if function(el):
            yield el
        else:
            filtered.append(el)


def _filter_by_size_dynamic(indices, size_fn, max_positions, raise_exception=False):
    def compare_leq(a, b):
        return a <= b if not isinstance(a, tuple) else max(a) <= b

    def check_size(idx):
        if isinstance(max_positions, float) or isinstance(max_positions, int):
            return size_fn(idx) <= max_positions
        elif isinstance(max_positions, dict):
            idx_size = size_fn(idx)
            assert isinstance(idx_size, dict)
            intersect_keys = set(max_positions.keys()) & set(idx_size.keys())
            return all(
                all(
                    a is None or b is None or a <= b
                    for a, b in zip(idx_size[key], max_positions[key])
                )
                for key in intersect_keys
            )
        else:
            # For MultiCorpusSampledDataset, will generalize it later
            if not isinstance(size_fn(idx), Iterable):
                return all(size_fn(idx) <= b for b in max_positions)
            return all(
                a is None or b is None or a <= b
                for a, b in zip(size_fn(idx), max_positions)
            )

    ignored = []
    itr = collect_filtered(check_size, indices, ignored)
    indices = np.fromiter(itr, dtype=np.int64, count=-1)
    return indices, ignored


def filter_by_size(indices, dataset, max_positions, raise_exception=False):
    """
    [deprecated] Filter indices based on their size.
    Use `BaseDataset::filter_indices_by_size` instead.

    Args:
        indices (List[int]): ordered list of dataset indices
        dataset (BaseDataset): metaseq dataset instance
        max_positions (tuple): filter elements larger than this size.
            Comparisons are done component-wise.
        raise_exception (bool, optional): if ``True``, raise an exception if
            any elements are filtered (default: False).
    """
    warnings.warn(
        "data_utils.filter_by_size is deprecated. "
        "Use `BaseDataset::filter_indices_by_size` instead.",
        stacklevel=2,
    )
    if isinstance(max_positions, float) or isinstance(max_positions, int):
        if hasattr(dataset, "sizes") and isinstance(dataset.sizes, np.ndarray):
            ignored = indices[dataset.sizes[indices] > max_positions].tolist()
            indices = indices[dataset.sizes[indices] <= max_positions]
        elif (
            hasattr(dataset, "sizes")
            and isinstance(dataset.sizes, list)
            and len(dataset.sizes) == 1
        ):
            ignored = indices[dataset.sizes[0][indices] > max_positions].tolist()
            indices = indices[dataset.sizes[0][indices] <= max_positions]
        else:
            indices, ignored = _filter_by_size_dynamic(
                indices, dataset.size, max_positions
            )
    else:
        indices, ignored = _filter_by_size_dynamic(indices, dataset.size, max_positions)

    if len(ignored) > 0 and raise_exception:
        raise Exception(
            (
                "Size of sample #{} is invalid (={}) since max_positions={}, "
                "skip this example with --skip-invalid-size-inputs-valid-test"
            ).format(ignored[0], dataset.size(ignored[0]), max_positions)
        )
    if len(ignored) > 0:
        logger.warning(
            (
                "{} samples have invalid sizes and will be skipped, "
                "max_positions={}, first few sample ids={}"
            ).format(len(ignored), max_positions, ignored[:10])
        )
    return indices


def filter_paired_dataset_indices_by_size(src_sizes, tgt_sizes, indices, max_sizes):
    """Filter a list of sample indices. Remove those that are longer
        than specified in max_sizes.

    Args:
        indices (np.array): original array of sample indices
        max_sizes (int or list[int] or tuple[int]): max sample size,
            can be defined separately for src and tgt (then list or tuple)

    Returns:
        np.array: filtered sample array
        list: list of removed indices
    """
    if max_sizes is None:
        return indices, []
    if type(max_sizes) in (int, float):
        max_src_size, max_tgt_size = max_sizes, max_sizes
    else:
        max_src_size, max_tgt_size = max_sizes
    if tgt_sizes is None:
        ignored = indices[src_sizes[indices] > max_src_size]
    else:
        ignored = indices[
            (src_sizes[indices] > max_src_size) | (tgt_sizes[indices] > max_tgt_size)
        ]
    if len(ignored) > 0:
        if tgt_sizes is None:
            indices = indices[src_sizes[indices] <= max_src_size]
        else:
            indices = indices[
                (src_sizes[indices] <= max_src_size)
                & (tgt_sizes[indices] <= max_tgt_size)
            ]
    return indices, ignored.tolist()


def batch_by_size(
    indices,
    num_tokens_fn,
    num_tokens_vec=None,
    max_tokens=None,
    max_sentences=None,
    required_batch_size_multiple=1,
    fixed_shapes=None,
):
    """
    Yield mini-batches of indices bucketed by size. Batches may contain
    sequences of different lengths.

    Args:
        indices (List[int]): ordered list of dataset indices
        num_tokens_fn (callable): function that returns the number of tokens at
            a given index
        num_tokens_vec (List[int], optional): precomputed vector of the number
            of tokens for each index in indices (to enable faster batch generation)
        max_tokens (int, optional): max number of tokens in each batch
            (default: None).
        max_sentences (int, optional): max number of sentences in each
            batch (default: None).
        required_batch_size_multiple (int, optional): require batch size to
            be less than N or a multiple of N (default: 1).
        fixed_shapes (List[Tuple[int, int]], optional): if given, batches will
            only be created with the given shapes. *max_sentences* and
            *required_batch_size_multiple* will be ignored (default: None).
    """
    try:
        from metaseq.data.data_utils_fast import (
            batch_by_size_fn,
            batch_by_size_vec,
            batch_fixed_shapes_fast,
        )
    except ImportError:
        raise ImportError(
            "Please build Cython components with: `pip install --editable .` "
            "or `python setup.py build_ext --inplace`"
        )
    except ValueError:
        raise ValueError(
            "Please build (or rebuild) Cython components with: `pip install "
            " --editable .` or `python setup.py build_ext --inplace`."
        )

    # added int() to avoid TypeError: an integer is required
    max_tokens = int(max_tokens) if max_tokens is not None else -1
    max_sentences = max_sentences if max_sentences is not None else -1
    bsz_mult = required_batch_size_multiple

    if not isinstance(indices, np.ndarray):
        indices = np.fromiter(indices, dtype=np.int64, count=-1)

    if num_tokens_vec is not None and not isinstance(num_tokens_vec, np.ndarray):
        num_tokens_vec = np.fromiter(num_tokens_vec, dtype=np.int64, count=-1)

    if fixed_shapes is None:
        if num_tokens_vec is None:
            return batch_by_size_fn(
                indices,
                num_tokens_fn,
                max_tokens,
                max_sentences,
                bsz_mult,
            )
        else:
            return batch_by_size_vec(
                indices,
                num_tokens_vec,
                max_tokens,
                max_sentences,
                bsz_mult,
            )
    else:
        fixed_shapes = np.array(fixed_shapes, dtype=np.int64)
        sort_order = np.lexsort(
            [
                fixed_shapes[:, 1].argsort(),  # length
                fixed_shapes[:, 0].argsort(),  # bsz
            ]
        )
        fixed_shapes_sorted = fixed_shapes[sort_order]
        return batch_fixed_shapes_fast(indices, num_tokens_fn, fixed_shapes_sorted)


def post_process(sentence: str, symbol: str):
    if symbol == "sentencepiece":
        sentence = sentence.replace(" ", "").replace("\u2581", " ").strip()
    elif symbol == "wordpiece":
        sentence = sentence.replace(" ", "").replace("_", " ").strip()
    elif symbol == "letter":
        sentence = sentence.replace(" ", "").replace("|", " ").strip()
    elif symbol == "_EOW":
        sentence = sentence.replace(" ", "").replace("_EOW", " ").strip()
    elif symbol == "none":
        pass
    elif symbol is not None:
        raise NotImplementedError(f"Unknown post_process option: {symbol}")
    return sentence


def compute_mask_indices(
    shape: Tuple[int, int],
    padding_mask: Optional[torch.Tensor],
    mask_prob: float,
    mask_length: int,
    mask_type: str = "static",
    mask_other: float = 0.0,
    min_masks: int = 0,
    no_overlap: bool = False,
    min_space: int = 0,
) -> np.ndarray:
    """
    Computes random mask spans for a given shape

    Args:
        shape: the the shape for which to compute masks.
            should be of size 2 where first element is batch size and 2nd is timesteps
        padding_mask: optional padding mask of the same size as shape, which will prevent masking padded elements
        mask_prob: probability for each token to be chosen as start of the span to be masked. this will be multiplied by
            number of timesteps divided by length of mask span to mask approximately this percentage of all elements.
            however due to overlaps, the actual number will be smaller (unless no_overlap is True)
        mask_type: how to compute mask lengths
            static = fixed size
            uniform = sample from uniform distribution [mask_other, mask_length*2]
            normal = sample from normal distribution with mean mask_length and stdev mask_other. mask is min 1 element
            poisson = sample from possion distribution with lambda = mask length
        min_masks: minimum number of masked spans
        no_overlap: if false, will switch to an alternative recursive algorithm that prevents spans from overlapping
        min_space: only used if no_overlap is True, this is how many elements to keep unmasked between spans
    """

    bsz, all_sz = shape
    mask = np.full((bsz, all_sz), False)

    all_num_mask = int(
        # add a random number for probabilistic rounding
        mask_prob * all_sz / float(mask_length)
        + np.random.rand()
    )

    all_num_mask = max(min_masks, all_num_mask)

    mask_idcs = []
    for i in range(bsz):
        if padding_mask is not None:
            sz = all_sz - padding_mask[i].long().sum().item()
            num_mask = int(
                # add a random number for probabilistic rounding
                mask_prob * sz / float(mask_length)
                + np.random.rand()
            )
            num_mask = max(min_masks, num_mask)
        else:
            sz = all_sz
            num_mask = all_num_mask

        if mask_type == "static":
            lengths = np.full(num_mask, mask_length)
        elif mask_type == "uniform":
            lengths = np.random.randint(mask_other, mask_length * 2 + 1, size=num_mask)
        elif mask_type == "normal":
            lengths = np.random.normal(mask_length, mask_other, size=num_mask)
            lengths = [max(1, int(round(x))) for x in lengths]
        elif mask_type == "poisson":
            lengths = np.random.poisson(mask_length, size=num_mask)
            lengths = [int(round(x)) for x in lengths]
        else:
            raise Exception("unknown mask selection " + mask_type)

        if sum(lengths) == 0:
            lengths[0] = min(mask_length, sz - 1)

        if no_overlap:
            mask_idc = []

            def arrange(s, e, length, keep_length):
                span_start = np.random.randint(s, e - length)
                mask_idc.extend(span_start + i for i in range(length))

                new_parts = []
                if span_start - s - min_space >= keep_length:
                    new_parts.append((s, span_start - min_space + 1))
                if e - span_start - keep_length - min_space > keep_length:
                    new_parts.append((span_start + length + min_space, e))
                return new_parts

            parts = [(0, sz)]
            min_length = min(lengths)
            for length in sorted(lengths, reverse=True):
                lens = np.fromiter(
                    (e - s if e - s >= length + min_space else 0 for s, e in parts),
                    np.int,
                )
                l_sum = np.sum(lens)
                if l_sum == 0:
                    break
                probs = lens / np.sum(lens)
                c = np.random.choice(len(parts), p=probs)
                s, e = parts.pop(c)
                parts.extend(arrange(s, e, length, min_length))
            mask_idc = np.asarray(mask_idc)
        else:
            min_len = min(lengths)
            if sz - min_len <= num_mask:
                min_len = sz - num_mask - 1

            mask_idc = np.random.choice(sz - min_len, num_mask, replace=False)

            mask_idc = np.asarray(
                [
                    mask_idc[j] + offset
                    for j in range(len(mask_idc))
                    for offset in range(lengths[j])
                ]
            )

        mask_idcs.append(np.unique(mask_idc[mask_idc < sz]))

    min_len = min([len(m) for m in mask_idcs])
    for i, mask_idc in enumerate(mask_idcs):
        if len(mask_idc) > min_len:
            mask_idc = np.random.choice(mask_idc, min_len, replace=False)
        mask[i, mask_idc] = True

    return mask


def get_mem_usage():
    try:
        import psutil

        mb = 1024 * 1024
        return f"used={psutil.virtual_memory().used / mb}Mb; avail={psutil.virtual_memory().available / mb}Mb"
    except ImportError:
        return "N/A"


def lengths_to_padding_mask(lens: torch.LongTensor) -> torch.BoolTensor:
    bsz, max_lens = lens.size(0), torch.max(lens).item()
    mask = torch.arange(max_lens).to(lens.device).view(1, max_lens)
    mask = mask.expand(bsz, -1) >= lens.view(bsz, 1).expand(-1, max_lens)
    return mask


def lengths_to_mask(lens: torch.LongTensor) -> torch.BoolTensor:
    return ~lengths_to_padding_mask(lens)


def _find_extra_valid_paths(dataset_path: str) -> set:
    paths = utils.split_paths(dataset_path)
    all_valid_paths = set()
    for sub_dir in paths:
        contents = PathManager.ls(sub_dir)
        valid_paths = [c for c in contents if re.match("valid*[0-9].*", c) is not None]
        all_valid_paths |= {os.path.basename(p) for p in valid_paths}
    # Remove .bin, .idx etc
    roots = {os.path.splitext(p)[0] for p in all_valid_paths}
    return roots


def raise_if_valid_subsets_unintentionally_ignored(train_cfg) -> None:
    """Raises if there are paths matching 'valid*[0-9].*' which are not combined or ignored."""
    if (
        train_cfg.dataset.ignore_unused_valid_subsets
        or train_cfg.dataset.combine_valid_subsets
        or train_cfg.dataset.disable_validation
        or getattr(train_cfg.task, "data", None) is None
    ):
        return
    other_paths = _find_extra_valid_paths(train_cfg.task.data)
    specified_subsets = train_cfg.dataset.valid_subset.split(",")
    ignored_paths = [p for p in other_paths if p not in specified_subsets]
    if ignored_paths:
        advice = "Set --combine-val to combine them or --ignore-unused-valid-subsets to ignore them."
        msg = f"Valid paths {ignored_paths} will be ignored. {advice}"
        raise ValueError(msg)
