# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import array
import random
import os
import tempfile
import unittest
from unittest.mock import patch, MagicMock

from metaseq.cli.train import post_checkpoint_callback, _get_destination_path
from metaseq.dataclass.configs import MetaseqConfig


def create_local_test_file(path, length=4096):
    value = random.randint(0, 16)
    with open(path, "wb") as f:
        array.array("b", [value] * length).tofile(f)


class TestPostCheckpointCallback(unittest.TestCase):
    def test_destination_path(self):
        self.assertEqual(
            _get_destination_path("/path/ckpt.pt", "/other"),
            "/other/ckpt.pt",
        )
        self.assertEqual(
            _get_destination_path(
                "/path/ckpt.pt", "https://acc.blob.core.windows.net/other?q=1"
            ),
            "https://acc.blob.core.windows.net/other/ckpt.pt?q=1",
        )
        self.assertEqual(
            _get_destination_path(
                "https://acc.blob.core.windows.net/path/ckpt.pt?q=1", "/other"
            ),
            "/other/ckpt.pt",
        )

    def test_nfs_copy(self):
        with tempfile.TemporaryDirectory() as local_dir, tempfile.TemporaryDirectory() as nfs_dir:
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
        with tempfile.TemporaryDirectory() as local_dir, tempfile.TemporaryDirectory() as nfs_dir:
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
                files_to_symlink_to=[
                    os.path.join(local_dir, "checkpoint_last-model_part-0-shard0.pt")
                ],
            )

            self.assertTrue(
                os.path.exists(
                    os.path.join(
                        nfs_dir, "checkpoint_100/checkpoint-model_part-0-shard0.pt"
                    )
                )
            )
            self.assertTrue(
                os.path.islink(
                    os.path.join(
                        nfs_dir,
                        "checkpoint_last/checkpoint_last-model_part-0-shard0.pt",
                    )
                )
            )

    def assert_azcopy(self, mock, src, dst):
        def _match(c):
            _, args, _ = c
            cmd = args[0]
            return cmd[-2] == src and cmd[-1] == dst

        self.assertTrue(
            any([_match(c) for c in mock.mock_calls]),
            f"Expected azcopy {src} -> {dst}\n\n{mock.mock_calls}",
        )

    def test_azure_blob_with_symlinks(self):
        mock_azcopy = MagicMock(return_value=MagicMock(returncode=0))
        with patch("metaseq.cli.train._run_azcopy", mock_azcopy):
            with tempfile.TemporaryDirectory() as local_dir:
                checkpoint_path = os.path.join(local_dir, "checkpoint_10.pt")
                create_local_test_file(checkpoint_path)

                upload_path = "https://testaccount.blob.core.windows.net/dest?q=1"
                cfg = MetaseqConfig()
                cfg.checkpoint.cloud_upload_path = upload_path

                post_checkpoint_callback(
                    cfg=cfg,
                    num_updates=10,
                    training_finished=False,
                    filename=checkpoint_path,
                    files_to_symlink_to=[os.path.join(local_dir, "checkpoint_last.pt")],
                )

                upload_src = "https://testaccount.blob.core.windows.net/dest/checkpoint_10.pt?q=1"
                upload_dst = "https://testaccount.blob.core.windows.net/dest/checkpoint_last.pt?q=1"
                self.assert_azcopy(mock_azcopy, checkpoint_path, upload_src)
                self.assert_azcopy(mock_azcopy, upload_src, upload_dst)


if __name__ == "__main__":
    unittest.main()
