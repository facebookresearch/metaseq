# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
import logging
import mmap
import os
import sys
import threading
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch

import metaseq.distributed.utils as distributed_utils

logger = logging.getLogger(__name__)


class JsonlDataset(torch.utils.data.Dataset):
    """
    For loading JSONL data and encoding on-the-fly with a given tokenizer.

    JSONL format is expected to roughly follow that of The Pile.
    One-line-per-document of the form:
    ```
    {
        "text": "text goes here, with newlines",
        "meta": {"pile_set_name": "name of corpus", "other": "metadata"}
    }
    ```

    Note that only the "text" key is used.
    """

    def __init__(
        self,
        path: str,
        tokenizer: Optional[Callable] = None,
        recache=False,
        epoch=1,
        data_subshard_count=1,
    ):
        self.path = path
        self.tokenizer = tokenizer

        self.threadlocal = threading.local()
        # TODO(susan): Fix this fairseq reference. _build_index fails otherwise.
        self.cache = Path(f"{path}.fairseq.idx.npy")
        # only build the cache in on the primary worker to prevent overloading nfs
        if distributed_utils.get_global_rank() != 0:
            distributed_utils.global_barrier()
        if self.cache.exists() and not recache:
            logger.info(f"Loading up cache: {self.cache}")
            self.offsets = np.load(self.cache, allow_pickle=True)
        elif distributed_utils.get_global_rank() == 0:
            self.offsets = self._build_index(path)
            np.save(self.cache, self.offsets, allow_pickle=False)
        if distributed_utils.get_global_rank() == 0:
            distributed_utils.global_barrier()

        self.epoch = epoch
        self.data_subshard_count = data_subshard_count

    def _get_mmap(self):
        if not hasattr(self.threadlocal, "handles"):
            f = open(self.path, "rb")
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            self.threadlocal.handles = [f, mm]
            if (
                self.path.endswith(".gz")
                or self.path.endswith(".bz")
                or self.path.endswith(".bz2")
            ):
                raise NotImplementedError(
                    "Compressed files are not supported because .seek() would require "
                    "rereading the entire file, making performance too slow."
                )
        return self.threadlocal.handles[-1]

    def __getitem__(self, idx):
        # Convert 0 based idx to subshard based idx
        # For instance, for a data_subshard_count of 3 and epoch number of 1,
        # subshard_idx goes like 0, 3, 6, 9 ...
        # For more details, see https://github.com/facebookresearch/metaseq/issues/166
        subshard_idx = self._get_subshard_id() + idx * self.data_subshard_count
        if subshard_idx < 0 or subshard_idx >= len(self.offsets):
            raise IndexError
        f = self._get_mmap()
        f.seek(self.offsets[subshard_idx])
        item = f.readline().decode("utf-8")
        item = json.loads(item)
        if self.tokenizer is not None:
            item = self.tokenizer(item)
        return item

    def __len__(self):
        # Virtual length of the dataset depends on the epoch number if the number of documents
        # is not perfectly divisible by the data_subshard_count
        if len(self.offsets) % self.data_subshard_count == 0:
            return len(self.offsets) // self.data_subshard_count
        else:
            # We are left with len(self.offsets) % self.data_subshard_count extra documents at the end
            extra_document_count = len(self.offsets) % self.data_subshard_count

            # Depending on the subshard id, these extra documents would be included or not
            if self._get_subshard_id() + 1 <= extra_document_count:
                return (len(self.offsets) // self.data_subshard_count) + 1
            else:
                return len(self.offsets) // self.data_subshard_count

    def _get_subshard_id(self):
        # Returns the subshard_id, which goes from 0 to self.data_subshard_count - 1 (0 indexed)
        # and then wraps around if the epoch id goes beyond the data_subshard_count
        return (self.epoch - 1) % self.data_subshard_count

    def _build_index(self, path: str):
        """Build index of start positions of each line."""
        logger.info(f"Building index for file: {path}")
        f = self._get_mmap()
        f.seek(0)
        offsets = []
        cur = 0
        line_num = 0
        while True:
            line = f.readline()
            if line != b"":
                try:
                    json.loads(line)
                except json.decoder.JSONDecodeError:
                    raise json.decoder.JSONDecodeError(
                        doc=path,
                        pos=line_num,
                        msg=f"Error while loading JSONL file {path} at line {line_num + 1}",
                    )
            if line == b"":
                break
            offsets.append(cur)
            cur += len(line)
            line_num += 1
        return offsets

    def __setstate__(self, state):
        self.__dict__ = state
        self.threadlocal = threading.local()

    def __getstate__(self):
        d = {}
        for i, v in self.__dict__.items():
            if i != "threadlocal":
                d[i] = v
        return d

    def __del__(self):
        if hasattr(self.threadlocal, "handles"):
            # cleanup files we opened on initialization
            while self.threadlocal.handles:
                self.threadlocal.handles.pop().close()

    @staticmethod
    def exists(path):
        return os.path.exists(path)


if __name__ == "__main__":
    """Usage:
    python metaseq/data/jsonl_dataset.py "flan_streaming/valid/00/*.jsonl"
    """
    parser = argparse.ArgumentParser(
        description="Precompute index file from JSONL files"
    )
    parser.add_argument(
        "pattern", help="glob to jsonl files, e.g. flan_streaming/valid/00/*.jsonl"
    )
    args = parser.parse_args()

    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=os.environ.get("LOGLEVEL", "INFO").upper(),
        stream=sys.stdout,
    )
    from glob import glob

    from tqdm import tqdm

    for f in tqdm(list(glob(args.pattern))):
        JsonlDataset(f, recache=True)
