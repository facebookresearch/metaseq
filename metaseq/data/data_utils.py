# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

try:
    from collections.abc import Iterable
except ImportError:
    from collections import Iterable

from collections.abc import Iterator as AbcIterator

import contextlib
import itertools
import logging
import os
import re
from typing import List

import numpy as np
import torch

from metaseq import utils
from metaseq.file_io import PathManager

logger = logging.getLogger(__name__)


def collate_tokens(
    values,
    pad_idx,
    eos_idx=None,
    left_pad=False,
    move_eos_to_beginning=False,
    pad_to_length=None,
    pad_to_multiple=1,
    pad_to_bsz=None,
    mask_prediction_value=None,
):
    """ Gets a list with <batch_size> tensors and transform them
        into a single padded tensor.
        E.g. 1: receiving tokens as inputs
            input with batch size of 2:
            values = [[13650, 290, 2692], [13650, 1245, 287]]
            pad_to_length = 4
            output:
            tensor([[13650, 290, 2692, 1], [13650, 1245, 287, 1]])
        E.g. 2: receiving logprobs as inputs
            input with batch size of 2, top k = 3:
            values = [
                [[[13650, -0.04], [14234, -1.24], [567, -2.21]],
                [[290, -0.001],  [1244, -2.35],  [534, -3.64]],
                [[2692, -0.2],   [134,  -2.34],  [342, -5.89]]],
                [[[13650, -0.04], [2342, -1.24], [213, -2.21]],
                [[1245, -0.001], [7561, -2.35], [3211, -3.64]],
                [[287, -0.2], [5456, -2.34], [3125, -5.89]]],
            ]
            pad_to_length = 4
            mask_prediction_value = 0
            output:
            tensor(
                [[[[13650, -0.04], [14234, -1.24], [567, -2.21]],
                [[290, -0.001],  [1244, -2.35],  [534, -3.64]],
                [[2692, -0.2],   [134,  -2.34],  [342, -5.89]]
                [[-1, 0],    [-1,  0],   [-1, 0]]],
                [[[13650, -0.04], [2342, -1.24], [213, -2.21]],
                [[1245, 0], [7561, -2.35], [3211, -3.64]],
                [[287, -0.2], [5456, -2.34], [3125, -5.89]],
                [[-1, 0], [-1,  0],  [-1, 0]]]]
            )
    """
    size = max(v.size(0) for v in values)
    size = size if pad_to_length is None else max(size, pad_to_length)
    if pad_to_multiple != 1 and size % pad_to_multiple != 0:
        size = int(((size - 0.1) // pad_to_multiple + 1) * pad_to_multiple)

    batch_size = len(values) if pad_to_bsz is None else max(len(values), pad_to_bsz)
    dimensions = [batch_size, size]

    num_dimensions = max(v.dim() for v in values)
    assert num_dimensions in (1, 3)
    if num_dimensions > 1:
        # The first dimension is the generation size. We already have it. Now we need to
        # get the size of other dimensions.

        max_top_k = max(v.size(1) for v in values)
        # data stands for the token_id and the correspondent logit/logprob value
        max_data = max(v.size(2) for v in values)

        dimensions.extend([max_top_k, max_data])

    res = values[0].new(*dimensions).fill_(pad_idx)

    if num_dimensions > 1:
        if mask_prediction_value is None:
            raise ValueError('mask_prediction_value must not be None when using multiple dimensions as target')
        # Masked tokens will be removed from loss computation after the forward pass.
        res.index_fill_(dim=-1, index=torch.LongTensor([-1]), value=mask_prediction_value)

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
    for k in itertools.count():
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


def _find_extra_valid_paths(dataset_path: str) -> set:
    paths = utils.split_paths(dataset_path)
    all_valid_paths = set()
    for sub_dir in paths:
        if "://" in sub_dir:
            continue
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


def get_number_of_lines_in_file(file_path: str):
    """
    Get the number of lines in a file.

    :param str file_path: Path of the file we want to count the lines of.
    :return int: the number of lines in the file.
    """
    with open(file_path, "r") as fh:
        return sum(1 for _ in fh)


def multiple_file_line_generator(file_paths: List[str]):
    """
    Takes a list of file_paths and returns a iterator that will yield one line
    at a time from all files one after the other.

    The files paths are first ordered lexicographically, and the generator will
    consume each file in order, yielding one line at a time.

    :param List[str] file_paths: Paths of the files that want to be iterated over
    """

    class MultiFileIterator(AbcIterator):

        def __init__(self, file_paths) -> None:
            assert len(file_paths) > 0, "Can't iterate over an empty list of files!"

            self.file_paths = sorted(file_paths)

            self.total_lines = sum(get_number_of_lines_in_file(file_path) for file_path in file_paths)

            self.current_line_num = 0
            self.current_file_path = self.file_paths.pop(0)

            self.f_handle = open(self.current_file_path, "r", encoding="utf-8")

        def __len__(self):
            return self.total_lines

        def __iter__(self):
            return self

        def __next__(self) -> str:
            return self._get_next_line()

        def _get_next_line(self):
            try:
                line = next(self.f_handle)
                self.current_line_num += 1
                return line
            except StopIteration:
                if len(self.file_paths) > 0:
                    # close current file and open next one
                    self.f_handle.close()

                    self.current_file_path = self.file_paths.pop(0)
                    self.current_line_num = 0
                    self.f_handle = open(self.current_file_path, "r", encoding="utf-8")

                    return self._get_next_line()
                else:
                    # stop iteration if there are no more files
                    raise StopIteration

    return MultiFileIterator(file_paths)


def truncate_target(target_tokens, tokens_per_sample):
    if len(target_tokens) <= tokens_per_sample:
        return target_tokens

    logger.warning(f"Target sequence length {len(target_tokens)} is longer than max sequence length {tokens_per_sample}")
    logger.warning("Truncating target sequence to 70% of max sequence length")

    # Right truncation for Target tokens
    target_tokens = target_tokens[:int(tokens_per_sample * 0.7)]

    return target_tokens


def truncate_source(
    source_tokens,
    max_context_length,
    truncation_type: str,
    len_prompt_eos_tokens: int = 0,
    masked_source_tokens: list = None
):
    if len(source_tokens) <= max_context_length:
        return source_tokens, masked_source_tokens

    if truncation_type == "left":
        source_tokens = source_tokens[-max_context_length:]
        if masked_source_tokens:
            masked_source_tokens = masked_source_tokens[-max_context_length:]
    elif truncation_type == "right":
        source_tokens = source_tokens[:max_context_length]
        if masked_source_tokens:
            masked_source_tokens = masked_source_tokens[:max_context_length]
    else:
        logger.warning(
            f"Source prompt has more tokens than max context length of {max_context_length + len_prompt_eos_tokens}, but truncation is set to {truncation_type}"
        )

    return source_tokens, masked_source_tokens
