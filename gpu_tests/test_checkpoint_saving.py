# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import sys
import os
import subprocess
import json
import logging
import multiprocessing
from functools import partial, partialmethod
import unittest
from unittest.mock import patch, Mock, MagicMock
import torch
from metaseq.dataclass.configs import DistributedTrainingConfig
from metaseq.launcher.opt_baselines import cli_main as sweep_cli_main
from metaseq.cli.train import cli_main as train_cli_main
from metaseq.distributed.utils import distributed_main
from metaseq.launcher.opt_job_constants import Size, M
import metaseq.utils as metaseq_utils


@unittest.skipIf(not torch.cuda.is_available(), "test requires 4 GPUs, none found")
@unittest.skipIf(
    DistributedTrainingConfig.distributed_world_size != 4,
    "test requires 4 GPUs",
)
class TestCheckpointSavingAndUploading(unittest.TestCase):
    def test_checkpoint_saving_and_uploading(self):
        max_update_first_run = 20
        multiprocessing.set_start_method("spawn", force=True)
        with torch.multiprocessing.Manager() as manager:
            events = manager.list()
            p = multiprocessing.Process(
                target=run_training,
                args=(
                    events,
                    max_update_first_run,
                ),
            )
            p.start()
            p.join()
            events_first_run = list(events)

        # check that training ran correctly
        training_log_events = [
            json.loads(event["message"])
            for event in events_first_run
            if event["type"] == "log" and event["message"].startswith('{"epoch"')
        ]
        self.assertEqual(len(training_log_events), max_update_first_run)
        self.assertEqual(
            int(training_log_events[-1]["num_updates"]), max_update_first_run
        )
        self.assertAlmostEqual(float(training_log_events[-1]["loss"]), 14.574, 1)

        # check that the correct checkpoints were created and uploaded
        upload_events = [
            event for event in events_first_run if event["type"] == "upload"
        ]
        self.assertEqual(len(upload_events), 8)
        checkpoint_dir = "test-checkpoint"
        common_checkpoint_model_dir = upload_events[0]["checkpoint_model_dir"]
        file_names_saved_azure = sorted(
            [worker_cmd["checkpoint_file"] for worker_cmd in upload_events]
        )
        expected_file_names = sorted(
            [
                "checkpoint_18-model_part-0-shard0.pt",
                "checkpoint_18-model_part-0-shard1.pt",
                "checkpoint_18-model_part-1-shard0.pt",
                "checkpoint_18-model_part-1-shard1.pt",
                "checkpoint_last-model_part-0-shard0.pt",
                "checkpoint_last-model_part-0-shard1.pt",
                "checkpoint_last-model_part-1-shard0.pt",
                "checkpoint_last-model_part-1-shard1.pt",
            ]
        )
        self.assertEqual(file_names_saved_azure, expected_file_names)
        for worker_cmd in upload_events:
            self.assertEqual(
                worker_cmd["command"],
                [
                    "azcopy",
                    "copy",
                    "--cap-mbps",
                    "96.0",
                    "https://myaccount.blob.core.windows.net/test",
                ],
            )
            self.assertEqual(
                worker_cmd["checkpoint_model_dir"], common_checkpoint_model_dir
            )
            self.assertEqual(worker_cmd["checkpoint_dir"], checkpoint_dir)
            self.assertEqual(worker_cmd["file_saved_locally"], True)

        # start second run, mock download the checkpoints from azure and keep training
        max_update_second_run = 35
        with torch.multiprocessing.Manager() as manager:
            events = manager.list()
            p = multiprocessing.Process(
                target=run_training,
                args=(
                    events,
                    max_update_second_run,
                ),
            )
            p.start()
            p.join()
            events_second_run = list(events)

        # check that that checkpoints were downloaded
        download_events = [
            event for event in events_second_run if event["type"] == "download"
        ]
        file_names_downloaded = sorted(
            [download["checkpoint_file"] for download in download_events]
        )
        last_checkpoints = sorted(
            [
                "checkpoint_last-model_part-0-shard0.pt",
                "checkpoint_last-model_part-0-shard1.pt",
                "checkpoint_last-model_part-1-shard0.pt",
                "checkpoint_last-model_part-1-shard1.pt",
            ]
        )
        self.assertEqual(file_names_downloaded, last_checkpoints)
        for download in download_events:
            self.assertEqual(
                download["blob_url"], "https://myaccount.blob.core.windows.net/test"
            )
            self.assertEqual(
                download["checkpoint_model_dir"], common_checkpoint_model_dir
            )
            self.assertEqual(download["checkpoint_dir"], checkpoint_dir)
            self.assertTrue(download["checkpoint_file"].endswith(download["suffix"]))

        # check that second training ran correctly
        training_log_events = [
            json.loads(event["message"])
            for event in events_second_run
            if event["type"] == "log" and event["message"].startswith('{"epoch"')
        ]
        self.assertEqual(
            len(training_log_events), max_update_second_run - max_update_first_run
        )
        self.assertEqual(
            int(training_log_events[-1]["num_updates"]), max_update_second_run
        )
        self.assertAlmostEqual(float(training_log_events[-1]["loss"]), 12.666, 1)

        # cleanup
        cleanup_checkpoints = subprocess.Popen(
            "rm -r ./test-checkpoint".split(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
        )
        _, _ = cleanup_checkpoints.communicate()


def run_training(events, max_update):
    argv_injection = (
        "python3 metaseq/launcher/opt_baselines.py   "
        "--prefix train.8m    --model-size 8m    --checkpoints-dir ./test-checkpoint    "
        "--tensorboard-logdir ./test-checkpoint    --num-trials 1    --azure   "
        "--num-gpus 4 --num-nodes 1   --seed 1   "
        "--local --disable-validation    --max-epoch 5    --max-update 5 --benchmark    "
        "--full-azure-upload-path https://myaccount.blob.core.windows.net/test   "
    )
    with patch("sys.argv", argv_injection.split()[1:]), patch(
        "metaseq.launcher.slurm.local_run",
        partial(local_run_mock, max_update=max_update, events=events),
    ), patch.dict(
        "metaseq.launcher.opt_job_constants.MODEL_SIZES",
        {"8m": Size(4, 128, 2, 64, int(0.0625 * M), 1.0e-3, 2)},
    ):
        sweep_cli_main()


def local_run_mock(args, env, train_cmd, dry_run, max_update, events):
    train_cmd[train_cmd.index("--max-update") + 1] = str(max_update)
    train_cmd[train_cmd.index("--log-interval") + 1] = "1"
    train_cmd[train_cmd.index("--save-interval-updates") + 1] = "18"
    train_cmd[train_cmd.index("--num-workers") + 1] = "1"
    with patch.dict("os.environ", env, clear=True):
        with patch("sys.argv", train_cmd[1:]):
            with patch(
                "metaseq.distributed.utils.distributed_main",
                partial(distributed_main_mock, events=events),
            ):
                train_cli_main()


def distributed_main_mock(i, main, cfg, kwargs, events):
    # need to patch this seperately here, otherwise spawns won't be patched
    with patch("logging.Logger._log", partialmethod(log_to_events, events=events)):
        with patch(
            "metaseq.cli.train._run_azcopy",
            partial(subprocess_run_mock, events=events),
        ):
            # need to mock this because the async part break the mock
            with patch(
                "metaseq.cli.train.Trainer.save_checkpoint",
                partialmethod(save_checkpoint_mock),
            ):
                with patch("metaseq.cli.train.os.remove"):
                    mock_metaseq_internal = MagicMock()
                    mock_metaseq_internal.azure_utils.download_recent_ckpt = partial(
                        download_checkpoint_mock, events=events
                    )
                    sys.modules["metaseq_internal"] = mock_metaseq_internal
                    distributed_main(i, main, cfg, kwargs)


def download_checkpoint_mock(blob_url, checkpoint_path, suffix, events):
    # mocks the download of the checkpoint from azure
    _, checkpoint_dir, checkpoint_model_dir, checkpoint_file = checkpoint_path.split(
        "/"
    )
    events.append(
        {
            "type": "download",
            "blob_url": blob_url,
            "checkpoint_dir": checkpoint_dir,
            "checkpoint_model_dir": checkpoint_model_dir,
            "checkpoint_file": checkpoint_file,
            "suffix": suffix,
        }
    )

    return True


def subprocess_run_mock(cmd, stdout, stderr, events):
    # replaces subprocess.run azcopy command that uploads to azure
    _, checkpoint_dir, checkpoint_model_dir, checkpoint_file = cmd[4].split("/")
    events.append(
        {
            "type": "upload",
            "command": cmd[:4] + cmd[5:],
            "checkpoint_dir": checkpoint_dir,
            "checkpoint_model_dir": checkpoint_model_dir,
            "checkpoint_file": checkpoint_file,
            "file_saved_locally": os.path.exists(cmd[4]),
        }
    )

    res = Mock()
    res.returncode = 0
    return res


def save_checkpoint_mock(
    self, filename, extra_state, training_finished=False, async_callback_fn=None
):
    logger = logging.getLogger("metaseq.trainer")
    """Save all training state in a checkpoint file."""
    # call state_dict on all ranks in case it needs internal communication
    state_dicts = self.state_dict(filename, training_finished)
    for filename, state_dict in state_dicts.items():
        logger.info(f"Saving checkpoint to {filename}")
        state_dict = metaseq_utils.move_to_cpu(
            state_dict,
            # keep params in FP16 when training with --memory-efficient-fp16
            cast_to_fp32=not self.cfg.common.memory_efficient_fp16,
        )
        state_dict["extra_state"].update(extra_state)
        if self.should_save_checkpoint_on_current_rank:
            # remove async part which break the patch
            # if not hasattr(self, "async_checkpoint"):
            #     self.async_checkpoint = ThreadPoolExecutor(max_workers=1)

            def perform_save():
                try:
                    logger.info(f"Beginning asynchronous torch.save to {filename}")
                    torch.save(state_dict, filename)
                    if async_callback_fn is not None:
                        async_callback_fn(filename)
                    logger.info(f"Asynchronous torch.save to {filename} complete.")
                except Exception as e:
                    logger.exception(f"Asynchronous save failed: {e}")

            # remove async part which break the patch
            # self.async_checkpoint.submit(perform_save)
            perform_save()
        logger.info(f"Finished saving checkpoint to {filename}")


def log_to_events(self, info, message, args, events, **kwargs):
    print(self, message)
    if isinstance(message, str):
        events.append(
            {
                "type": "log",
                "message": message,
            }
        )


if __name__ == "__main__":
    unittest.main()
