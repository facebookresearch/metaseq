# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import array
import random
import os
import tempfile
import unittest

from metaseq.cli.train import post_checkpoint_callback
from metaseq.dataclass.configs import MetaseqConfig


def create_local_test_file(path, length=4096):
    value = random.randint(0, 16)
    with open(path, "wb") as f:
        array.array("b", [value] * length).tofile(f)


class TestPostCheckpointCallback(unittest.TestCase):
    def test_nfs_copy(self):
        with (
            tempfile.TemporaryDirectory() as local_dir,
            tempfile.TemporaryDirectory() as nfs_dir,
        ):
            checkpoint_path = os.path.join(
                local_dir, "checkpoint_100-model_part-0-shard0.pt"
            )
            create_local_test_file(checkpoint_path)

            cfg = MetaseqConfig()
            cfg.checkpoint.cloud_upload_path = f"nfs:{nfs_dir}"
            # Prevent evals
            cfg.checkpoint.nfs_eval_frequency = 0

            post_checkpoint_callback(
                cfg=cfg,
                num_updates=10,
                training_finished=False,
                filename=checkpoint_path,
                files_to_symlink_to=None,
            )

            expected_path = os.path.join(
                nfs_dir, "checkpoint_100/checkpoint-model_part-0-shard0.pt"
            )
            self.assertTrue(
                os.path.exists(expected_path), f"File should exist: {expected_path}"
            )

    def test_nfs_copy_with_symlinks(self):
        with (
            tempfile.TemporaryDirectory() as local_dir,
            tempfile.TemporaryDirectory() as nfs_dir,
        ):
            checkpoint_path = os.path.join(local_dir, "checkpoint_10.pt")
            create_local_test_file(checkpoint_path)

            cfg = MetaseqConfig()
            cfg.checkpoint.cloud_upload_path = f"nfs:{nfs_dir}"
            # Prevent evals
            cfg.checkpoint.nfs_eval_frequency = 0

            post_checkpoint_callback(
                cfg=cfg,
                num_updates=10,
                training_finished=False,
                filename=checkpoint_path,
                files_to_symlink_to=[os.path.join(local_dir, "checkpoint_last.pt")],
            )

            self.assertTrue(
                os.path.exists(os.path.join(nfs_dir, "checkpoint_10/checkpoint.pt"))
            )
            self.assertTrue(
                os.path.islink(
                    os.path.join(nfs_dir, "checkpoint_10/checkpoint_last.pt")
                )
            )


if __name__ == "__main__":
    unittest.main()
