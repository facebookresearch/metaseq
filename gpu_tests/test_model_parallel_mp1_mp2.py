# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import subprocess
import json
import multiprocessing
from functools import partial, partialmethod
import unittest
from unittest.mock import patch
import torch
from metaseq.dataclass.configs import DistributedTrainingConfig
from metaseq.launcher.opt_baselines import cli_main as sweep_cli_main
from metaseq.cli.train import cli_main as train_cli_main
from metaseq.launcher.opt_job_constants import Size, M


@unittest.skipIf(not torch.cuda.is_available(), "test requires 4 GPUs, none found")
@unittest.skipIf(
    DistributedTrainingConfig.distributed_world_size != 4,
    "test requires 4 GPUs",
)
class TestModelParallelMP1(unittest.TestCase):
    """
    The test will verify that the model can be trained with
    model_parallel = 1
    The test checks hat the number of trianing steps performed is correct
    and that the required loss is achieved on the last iteration
    """

    def test_model_parallel_mp1(self):
        # parameters to train an mp1 model
        argv_injection = (
            "python3 metaseq/launcher/opt_baselines.py   "
            "--prefix train.8m    --model-size 8m_mp1    --checkpoints-dir ./test-checkpoint    "
            "--tensorboard-logdir ./test-checkpoint    --num-trials 1    --azure   "
            "--num-gpus 4 --num-nodes 1   --seed 1   "
            "--local --disable-validation    --max-epoch 5    --max-update 5 --benchmark    "
        )
        max_update_first_run = 20
        size_patch_dict = {"8m_mp1": Size(4, 128, 2, 64, int(0.03125 * M), 1.0e-3, 1)}

        training_log_events = self._test_model_parallel(
            max_update_first_run=max_update_first_run,
            argv_injection=argv_injection,
            size_patch_dict=size_patch_dict,
        )

        # check that training ran correctly
        # check that the number of updates was correct
        self.assertNotEqual(training_log_events, [])
        self.assertIsNotNone(training_log_events[-1]["num_updates"])
        self.assertEqual(
            int(training_log_events[-1]["num_updates"]), max_update_first_run
        )
        # check the achieved loss is correct
        loss_val = float(training_log_events[-1]["loss"])
        self.assertAlmostEqual(loss_val, 14.736, 1)  # 1 digit precision

    def test_model_parallel_mp2(self):
        # parameters to train an mp2 model
        argv_injection = (
            "python3 metaseq/launcher/opt_baselines.py   "
            "--prefix train.8m    --model-size 8m    --checkpoints-dir ./test-checkpoint    "
            "--tensorboard-logdir ./test-checkpoint    --num-trials 1    --azure   "
            "--num-gpus 4 --num-nodes 1   --seed 1   "
            "--local --disable-validation    --max-epoch 5    --max-update 5 --benchmark    "
        )
        max_update_first_run = 20
        size_patch_dict = {"8m": Size(4, 128, 2, 64, int(0.03125 * M), 1.0e-3, 2)}

        training_log_events = self._test_model_parallel(
            max_update_first_run=max_update_first_run,
            argv_injection=argv_injection,
            size_patch_dict=size_patch_dict,
        )

        # check that training ran correctly
        # check that the number of updates was correct
        self.assertNotEqual(training_log_events, [])
        self.assertIsNotNone(training_log_events[-1]["num_updates"])
        self.assertEqual(
            int(training_log_events[-1]["num_updates"]), max_update_first_run
        )
        # check the achieved loss is correct
        loss_val = float(training_log_events[-1]["loss"])
        self.assertAlmostEqual(loss_val, 14.744, 1)  # 1 digit precision

    def _test_model_parallel(
        self, max_update_first_run, argv_injection, size_patch_dict
    ):
        """
        Helper function to run the test
        """
        # start the process for the model run
        multiprocessing.set_start_method("spawn", force=True)
        with torch.multiprocessing.Manager() as manager:
            events = manager.list()
            p = multiprocessing.Process(
                target=run_training,
                args=(max_update_first_run, events, argv_injection, size_patch_dict),
            )
            p.start()
            p.join()
            events_first_run = list(events)

        # cleanup of the checkpoints files
        cleanup_checkpoints = subprocess.Popen(
            "rm -r ./test-checkpoint".split(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
        )
        _, _ = cleanup_checkpoints.communicate()

        # parse the log events from the log_to_events()
        training_log_events = [
            json.loads(event["message"])
            for event in events_first_run
            if event["type"] == "log" and event["message"].startswith('{"epoch"')
        ]

        return training_log_events


def run_training(max_update, events, argv_injection, size_patch_dict):
    # clean any unused cach to reduce CUDA OOM
    torch.cuda.empty_cache()
    # main arguments to run the training script
    # both patches are aneeded to run the job of the circleci GPUs
    with patch("sys.argv", argv_injection.split()[1:]), patch(
        "metaseq.launcher.slurm.local_run",
        partial(local_run_mock, max_update=max_update, events=events),
    ), patch.dict(
        "metaseq.launcher.opt_job_constants.MODEL_SIZES",
        # reduce the batch size for CUDA memory optimization
        size_patch_dict,
    ):
        sweep_cli_main()


def local_run_mock(args, env, train_cmd, dry_run, max_update, events):
    """
    The function introduces several pathces for the argumets of the
    model training. These patches are needed to pass gpu tests on
    circleci GPUs (empirical knowledge)
    """
    train_cmd[train_cmd.index("--max-update") + 1] = str(max_update)
    train_cmd[train_cmd.index("--num-workers") + 1] = "1"

    with patch("logging.Logger._log", partialmethod(log_to_events, events=events)):
        with patch.dict("os.environ", env, clear=True):
            with patch("sys.argv", train_cmd[1:]):
                train_cli_main()


def log_to_events(self, info, message, args, events, **kwargs):
    """
    The function is used to collect logging info from the subprocesses
    and store it in the 'events' variable, which is then passed over
    to the main process for asserting that the model ran correctly
    """
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
