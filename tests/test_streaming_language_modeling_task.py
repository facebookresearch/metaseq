# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
import tempfile
import unittest

import torch

from tests.utils import train_language_model
from cpu_tests.test_utils import write_dummy_jsonl_data_dir, write_dummy_bpe

try:
    import tokenizers  # noqa

    has_hf_tokenizers = True
except ImportError:
    has_hf_tokenizers = False


class TestReproducibility(unittest.TestCase):
    @unittest.skipIf(not has_hf_tokenizers, "skip test if tokenizers is missing")
    def _test_reproducibility(
        self,
        name,
        extra_flags=None,
        delta=0.0001,
        resume_checkpoint="checkpoint1.pt",
        max_epoch=3,
    ):
        def get_last_log_stats_containing_string(log_records, search_string):
            for log_record in logs.records[::-1]:
                if isinstance(log_record.msg, str) and search_string in log_record.msg:
                    return json.loads(log_record.msg)

        if extra_flags is None:
            extra_flags = []

        with tempfile.TemporaryDirectory(name) as data_dir:
            write_dummy_jsonl_data_dir(data_dir)
            vocab, merges = write_dummy_bpe(data_dir)

            # train epochs 1 and 2 together
            with self.assertLogs() as logs:
                train_language_model(
                    data_dir=data_dir,
                    arch="transformer_lm_gpt2_tiny",
                    extra_flags=[
                        "--vocab-filename",
                        vocab,
                        "--merges-filename",
                        merges,
                        "--dropout",
                        "0.0",
                        "--log-format",
                        "json",
                        "--log-interval",
                        "1",
                        "--max-epoch",
                        str(max_epoch),
                        "--batch-size",
                        "2",
                    ]
                    + extra_flags,
                    task="streaming_language_modeling",
                    max_tokens=None,
                )
            train_log = get_last_log_stats_containing_string(logs.records, "train_loss")
            valid_log = get_last_log_stats_containing_string(logs.records, "valid_loss")

            # train epoch 2, resuming from previous checkpoint 1
            os.rename(
                os.path.join(data_dir, resume_checkpoint),
                os.path.join(data_dir, "checkpoint_last.pt"),
            )
            with self.assertLogs() as logs:
                train_language_model(
                    data_dir=data_dir,
                    arch="transformer_lm_gpt2_tiny",
                    extra_flags=[
                        "--vocab-filename",
                        vocab,
                        "--merges-filename",
                        merges,
                        "--dropout",
                        "0.0",
                        "--log-format",
                        "json",
                        "--log-interval",
                        "1",
                        "--max-epoch",
                        str(max_epoch),
                        "--batch-size",
                        "2",
                    ]
                    + extra_flags,
                    task="streaming_language_modeling",
                    max_tokens=None,
                )
            train_res_log = get_last_log_stats_containing_string(
                logs.records, "train_loss"
            )
            valid_res_log = get_last_log_stats_containing_string(
                logs.records, "valid_loss"
            )

            for k in ["train_loss", "train_ppl", "train_num_updates", "train_gnorm"]:
                self.assertAlmostEqual(
                    float(train_log[k]), float(train_res_log[k]), delta=delta
                )
            for k in [
                "valid_loss",
                "valid_ppl",
                "valid_num_updates",
                "valid_best_loss",
            ]:
                self.assertAlmostEqual(
                    float(valid_log[k]), float(valid_res_log[k]), delta=delta
                )

    def test_reproducibility(self):
        self._test_reproducibility("test_reproducibility")

    @unittest.skipIf(not torch.cuda.is_available(), "test requires a GPU")
    def test_reproducibility_fp16(self):
        self._test_reproducibility(
            "test_reproducibility_fp16",
            [
                "--fp16",
                "--fp16-init-scale",
                "4096",
            ],
            delta=0.011,
        )

    @unittest.skipIf(not torch.cuda.is_available(), "test requires a GPU")
    def test_reproducibility_memory_efficient_fp16(self):
        self._test_reproducibility(
            "test_reproducibility_memory_efficient_fp16",
            [
                "--memory-efficient-fp16",
                "--fp16-init-scale",
                "4096",
            ],
        )

    def test_mid_epoch_reproducibility(self):
        self._test_reproducibility(
            "test_mid_epoch_reproducibility",
            ["--save-interval-updates", "3"],
            resume_checkpoint="checkpoint_1_3.pt",
            max_epoch=1,
        )


if __name__ == "__main__":
    unittest.main()
