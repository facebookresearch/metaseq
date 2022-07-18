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


def write_one_jsonl_(
    jsonl_path, num_lines=5, text_len_min=5, text_len_max=50, truncate=False
):
    data = []
    with open(jsonl_path, "w") as h:
        for _ in range(num_lines):
            text_len = random.choice(range(text_len_min, text_len_max))
            data.append(
                {"text": "".join(random.choices(string.ascii_letters, k=text_len))}
            )
            print(json.dumps(data[-1]), file=h)
            if truncate and _ == 0:
                line = "".join(random.choices(string.ascii_letters, k=text_len))
                data.append(line)
                print(line, file=h)
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

    def test_non_empty_jsonl(self):
        with tempfile.NamedTemporaryFile() as jsonl_file:
            orig_data = write_one_jsonl_(jsonl_file.name, num_lines=0)
            assert len(orig_data) == 0
            self.assertRaises(ValueError, JsonlDataset, jsonl_file.name)

    def test_formatting_json(self):
        with tempfile.NamedTemporaryFile() as jsonl_file:
            orig_data = write_one_jsonl_(jsonl_file.name, num_lines=5, truncate=True)
            assert (
                len(orig_data) == 6
            )  # it's 6 because we add an extra line of badly formatted json
            self.assertRaises(
                json.decoder.JSONDecodeError, JsonlDataset, jsonl_file.name
            )

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

    def _iterate_over_dataset(self, dataset: JsonlDataset):
        iterated_documents = []
        for idx in range(len(dataset)):
            iterated_documents.append(dataset[idx]["text"])

        return iterated_documents

    def test_dataset_with_subshards(self):
        with tempfile.NamedTemporaryFile() as jsonl_file:
            documents_as_dict = write_one_jsonl_(jsonl_file.name, num_lines=11)
            documents = [elem["text"] for elem in documents_as_dict]

            dataset = JsonlDataset(jsonl_file.name, epoch=1, data_subshard_count=3)
            # The 4 documents would be 0, 3, 6 and 9 (0 based indexing)
            self.assertEqual(
                set([documents[idx] for idx in [0, 3, 6, 9]]),
                set(self._iterate_over_dataset(dataset)),
            )

            dataset = JsonlDataset(jsonl_file.name, epoch=3, data_subshard_count=3)
            # The 3 documents would be 2, 5 and 8 (0 based indexing)
            self.assertEqual(
                set([documents[idx] for idx in [2, 5, 8]]),
                set(self._iterate_over_dataset(dataset)),
            )

            dataset = JsonlDataset(jsonl_file.name, epoch=4, data_subshard_count=3)
            # If epoch > data_subshard_count , we wrap around. So epoch=4 behaves like epoch=1
            self.assertEqual(
                set([documents[idx] for idx in [0, 3, 6, 9]]),
                set(self._iterate_over_dataset(dataset)),
            )

        # Confirm that iterating on the dataset works as expected
        with tempfile.NamedTemporaryFile() as jsonl_file:
            documents_as_dict = write_one_jsonl_(jsonl_file.name, num_lines=11)
            documents = [elem["text"] for elem in documents_as_dict]

            # Assuming a data_subshard_count of 3, in 3 epochs we should have iterated
            # over the whole dataset
            iterated_documents = []
            for epoch in range(1, 4):
                dataset = JsonlDataset(
                    jsonl_file.name, epoch=epoch, data_subshard_count=3
                )
                iterated_documents.extend(self._iterate_over_dataset(dataset))

            # Ensure that all 11 documents have been iterated through
            self.assertEqual(set(iterated_documents), set(documents))

            # Now, let's try iterating for a total of 9 epochs, and assert that the entire data
            # was iterated over thrice
            iterated_documents = []
            for epoch in range(1, 10):
                dataset = JsonlDataset(
                    jsonl_file.name, epoch=epoch, data_subshard_count=3
                )
                iterated_documents.extend(self._iterate_over_dataset(dataset))

            assert len(iterated_documents) == 33  # 11*3

            # We iterated over the same data thrice, so deduplicated documents should still match
            self.assertEqual(set(documents), set(iterated_documents))


if __name__ == "__main__":
    unittest.main()
