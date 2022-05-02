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

    def __init__(self, path: str, tokenizer: Optional[Callable] = None, recache=False):
        self.path = path
        self.tokenizer = tokenizer

        self.threadlocal = threading.local()
        # TODO(susan): Fix this fairseq reference. _build_index fails otherwise.
        self.cache = Path(f"{path}.fairseq.idx.npy")
        if self.cache.exists() and not recache:
            self.offsets = np.load(self.cache)
        else:
            self.offsets = self._build_index(path)
            np.save(self.cache, self.offsets)
        # print(f'n offsets: {len(self.offsets)}')

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
        if idx < 0 or idx >= len(self):
            raise IndexError
        f = self._get_mmap()
        f.seek(self.offsets[idx])
        item = f.readline().decode("utf-8")
        item = json.loads(item)
        if self.tokenizer is not None:
            item = self.tokenizer(item)
        return item

    def __len__(self):
        return len(self.offsets)

    def _build_index(self, path: str):
        """Build index of start positions of each line."""
        logger.info(f"Building index for file: {path}")
        f = self._get_mmap()
        f.seek(0)
        offsets = []
        cur = 0
        while True:
            line = f.readline()
            if line == b"":
                break
            offsets.append(cur)
            cur += len(line)
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
