# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import sys
import os
import subprocess
import json
import multiprocessing
from functools import partial, partialmethod
import unittest
from unittest.mock import patch, MagicMock
import torch
from metaseq.dataclass.configs import DistributedTrainingConfig
from metaseq.launcher.opt_baselines import cli_main as sweep_cli_main
from metaseq.cli.train import cli_main as train_cli_main
from metaseq.distributed.utils import distributed_main
from metaseq.launcher.opt_job_constants import Size, M
import logging


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

        # check that the correct checkpoints were created
        checkpoint_dir = "test-checkpoint-local"
        common_checkpoint_model_dir = sorted(os.listdir(checkpoint_dir))[0]
        assert common_checkpoint_model_dir.endswith(".ngpu4")

        file_names_saved_local = []
        for file in os.listdir(
            os.path.join(checkpoint_dir, common_checkpoint_model_dir)
        ):
            if file.endswith(".pt"):
                file_names_saved_local.append(file)
        file_names_saved_local.sort()

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
        self.assertEqual(file_names_saved_local, expected_file_names)

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

        # check that second training ran correctly
        training_log_events_second = [
            json.loads(event["message"])
            for event in events_second_run
            if event["type"] == "log" and event["message"].startswith('{"epoch"')
        ]
        self.assertEqual(
            len(training_log_events_second),
            max_update_second_run - max_update_first_run,
        )
        self.assertEqual(
            int(training_log_events_second[-1]["num_updates"]), max_update_second_run
        )
        self.assertAlmostEqual(float(training_log_events_second[-1]["loss"]), 12.666, 1)

        # cleanup
        cleanup_checkpoints = subprocess.Popen(
            "rm -r ./test-checkpoint-local".split(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
        )
        _, _ = cleanup_checkpoints.communicate()


def run_training(events, max_update):
    argv_injection = (
        "python3 metaseq/launcher/opt_baselines.py   "
        "--prefix train.8m    --model-size 8m    --checkpoints-dir ./test-checkpoint-local    "
        "--tensorboard-logdir ./test-checkpoint-local    --num-trials 1    --azure   "
        "--num-gpus 4 --num-nodes 1   --seed 1   "
        "--local --disable-validation    --max-epoch 5    --max-update 5 --benchmark    "
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
    with patch.object(
        logging.Logger,
        "_log",
        new=partialmethod(log_to_events, events=events),
    ):
        with patch("metaseq.cli.train.os.remove"):
            mock_metaseq_internal = MagicMock()
            sys.modules["metaseq_internal"] = mock_metaseq_internal
            distributed_main(i, main, cfg, kwargs)


def log_to_events(self, info, message, args, events, **kwargs):
    print(self, info, message)
    if isinstance(message, str):
        events.append(
            {
                "type": "log",
                "message": message,
            }
        )


if __name__ == "__main__":
    unittest.main()
