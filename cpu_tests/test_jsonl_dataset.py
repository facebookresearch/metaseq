# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
import random
import string
import tempfile
import unittest
from unittest.mock import MagicMock

from metaseq.data import JsonlDataset


def write_one_jsonl_(jsonl_path, num_lines=5, text_len_min=5, text_len_max=50):
    data = []
    with open(jsonl_path, "w") as h:
        for _ in range(num_lines):
            text_len = random.choice(range(text_len_min, text_len_max))
            data.append(
                {"text": "".join(random.choices(string.ascii_letters, k=text_len))}
            )
            print(json.dumps(data[-1]), file=h)
    return data


class TestJsonlDataset(unittest.TestCase):
    def test_one_line(self):
        self._test_jsonl_dataset(num_lines=1)

    def test_multiple_lines(self):
        self._test_jsonl_dataset(num_lines=5)

    def test_bad_cache(self):
        with tempfile.NamedTemporaryFile() as jsonl_file:
            write_one_jsonl_(jsonl_file.name, num_lines=3)
            dataset = JsonlDataset(jsonl_file.name)
            assert len(dataset) == 3

            write_one_jsonl_(jsonl_file.name, num_lines=5)
            dataset = JsonlDataset(jsonl_file.name)
            assert len(dataset) == 3  # it's still 3 because of the cache

            os.remove(dataset.cache)
            dataset = JsonlDataset(jsonl_file.name)
            assert len(dataset) == 5  # it's now 5 because the cache is recreated

    def test_tokenizer(self, num_lines=5):
        def tokenizer_fn(jsonl):
            return list(jsonl["text"])

        tokenizer = MagicMock(wraps=tokenizer_fn)
        with tempfile.NamedTemporaryFile() as jsonl_file:
            orig_data = write_one_jsonl_(jsonl_file.name, num_lines=num_lines)
            assert len(orig_data) == num_lines
            dataset = JsonlDataset(jsonl_file.name, tokenizer=tokenizer)
            assert tokenizer.call_count == 0

            foo = dataset[1]
            assert foo == list(orig_data[1]["text"])
            assert tokenizer.call_count == 1

            foo = dataset[1]
            assert tokenizer.call_count == 2

            foo = dataset[4]
            assert foo == list(orig_data[4]["text"])
            assert tokenizer.call_count == 3

    def _test_jsonl_dataset(self, num_lines, tokenizer=None):
        with tempfile.NamedTemporaryFile() as jsonl_file:
            orig_data = write_one_jsonl_(jsonl_file.name, num_lines=num_lines)
            assert len(orig_data) == num_lines
            dataset = JsonlDataset(jsonl_file.name, tokenizer=None)
            assert len(dataset) == len(orig_data)
            for orig_json, read_json in zip(orig_data, dataset):
                assert orig_json.keys() == read_json.keys()
                for k in orig_json.keys():
                    assert orig_json[k] == read_json[k]
