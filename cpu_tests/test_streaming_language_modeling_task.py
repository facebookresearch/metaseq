# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import tempfile
import unittest

import torch
from metaseq import options

from metaseq.tasks.streaming_language_modeling import StreamingLanguageModelingTask
from cpu_tests.test_utils import (
    write_one_jsonl,
    write_dummy_bpe,
)
from metaseq.dataclass.utils import convert_namespace_to_omegaconf

try:
    import tokenizers  # noqa

    has_hf_tokenizers = True
except ImportError:
    has_hf_tokenizers = False


class TestDatasetLoading(unittest.TestCase):
    @unittest.skipIf(not has_hf_tokenizers, "skip test if tokenizers is missing")
    def test_load_dataset(self):
        with tempfile.TemporaryDirectory() as data_dir:
            print(data_dir)
            train_dir = os.path.join(data_dir, "train")
            shard_00 = os.path.join(train_dir, "00")
            shard_01 = os.path.join(train_dir, "01")

            vocab_file, merges_file = write_dummy_bpe(data_dir)

            # Create shard folders, and jsonl files
            os.makedirs(shard_00)
            os.makedirs(shard_01)

            shard_00_json_1 = write_one_jsonl(
                os.path.join(shard_00, "json_1.jsonl"), num_lines=11
            )
            shard_00_json_2 = write_one_jsonl(
                os.path.join(shard_00, "json_2.jsonl"), num_lines=12
            )
            shard_01_json_1 = write_one_jsonl(
                os.path.join(shard_01, "json_1.jsonl"), num_lines=13
            )
            shard_01_json_2 = write_one_jsonl(
                os.path.join(shard_01, "json_2.jsonl"), num_lines=14
            )

            train_parser = options.get_training_parser()
            train_args = options.parse_args_and_arch(
                train_parser,
                [
                    "--arch",
                    "transformer_lm_gpt2_tiny",
                    "--task",
                    "streaming_language_modeling",
                    data_dir,
                    "--vocab-filename",
                    vocab_file,
                    "--merges-filename",
                    merges_file,
                    "--sample-break-mode",
                    "complete",
                ],
            )
            cfg = convert_namespace_to_omegaconf(train_args)
            # Data subshard count = 3
            cfg.task.data_subshard_count = 3

            self.task = StreamingLanguageModelingTask(cfg.task)

            jsonl_data = {
                "shard_00_json_1": [],
                "shard_00_json_2": [],
                "shard_01_json_1": [],
                "shard_01_json_2": [],
            }
            for elem in shard_00_json_1:
                jsonl_data["shard_00_json_1"].append(self.task._tokenize_one_json(elem))
            for elem in shard_00_json_2:
                jsonl_data["shard_00_json_2"].append(self.task._tokenize_one_json(elem))
            for elem in shard_01_json_1:
                jsonl_data["shard_01_json_1"].append(self.task._tokenize_one_json(elem))
            for elem in shard_01_json_2:
                jsonl_data["shard_01_json_2"].append(self.task._tokenize_one_json(elem))

            # Iterate over epochs 1 to 3
            # After these epochs, we should have iterated over shard 00, which consists of
            # jsonl_data["shard_00_json_1"] and jsonl_data["shard_00_json_2"]
            self.ensure_epoch_iteration_is_consistent(
                jsonl_data["shard_00_json_1"],
                jsonl_data["shard_00_json_2"],
                1,
                3,
                self.task.args.data_subshard_count,
            )

            # Iterate over epochs 4 to 6
            # After these epochs, we should have iterated over shard 01, which consists of
            # jsonl_data["shard_01_json_1"] and jsonl_data["shard_01_json_2"]
            self.ensure_epoch_iteration_is_consistent(
                jsonl_data["shard_01_json_1"],
                jsonl_data["shard_01_json_2"],
                4,
                6,
                self.task.args.data_subshard_count,
            )

            # Iterate over epochs 7 to 9. We should wrap around and again iterate over shard 00
            self.ensure_epoch_iteration_is_consistent(
                jsonl_data["shard_00_json_1"],
                jsonl_data["shard_00_json_2"],
                7,
                9,
                self.task.args.data_subshard_count,
            )

    def ensure_epoch_iteration_is_consistent(
        self, jsonl_data_1, jsonl_data_2, epoch_start, epoch_end, data_subshard_count
    ):
        iterated_data = []
        for epoch in range(epoch_start, epoch_end + 1):
            it_data = self.ensure_iterated_data_is_consistent(
                jsonl_data_1,
                jsonl_data_2,
                epoch,
                data_subshard_count,
            )

            iterated_data.extend(it_data)

        # Ensure that after epoch_start-epoch_end epochs, we have iterated over the whole shard
        # Assumption - The shard contains only two datasets - jsonl_data_1 and jsonl_data_2
        data_that_should_be_iterated = jsonl_data_1 + jsonl_data_2
        self.assertTrue(
            set(self._stringify_and_sort_tensor_list(iterated_data)),
            set(self._stringify_and_sort_tensor_list(data_that_should_be_iterated)),
        )

    def ensure_iterated_data_is_consistent(
        self, jsonl_data_1, jsonl_data_2, epoch, data_subshard_count
    ):
        """
        Helper function to iterate over a single epoch, and ensure that the iterated documents
        match our expectation from the shard/subshard standpoint.
        """
        self.task.load_dataset("train", epoch=epoch)
        iterated_data = [doc for doc in self.task.dataset("train").dataset.dataset]

        # For a given epoch, the start offset would be
        offset = (epoch - 1) % data_subshard_count

        data_that_should_be_iterated = []
        # Note that the iteration skips over data_subshard_count documents, and starts
        # from an offset dictated by the value of epoch.
        for ind in range(offset, len(jsonl_data_1), data_subshard_count):
            data_that_should_be_iterated.append(jsonl_data_1[ind])
        for ind in range(offset, len(jsonl_data_2), data_subshard_count):
            data_that_should_be_iterated.append(jsonl_data_2[ind])

        self.assertTrue(
            all(
                [
                    torch.equal(elem1, elem2)
                    for elem1, elem2 in zip(data_that_should_be_iterated, iterated_data)
                ]
            )
        )

        return iterated_data

    def _stringify_and_sort_tensor_list(self, tensor_list):
        return sorted([str(tensor) for tensor in tensor_list])


if __name__ == "__main__":
    unittest.main()
