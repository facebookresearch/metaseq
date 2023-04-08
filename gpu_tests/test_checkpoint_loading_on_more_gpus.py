# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import json
import logging
import multiprocessing
import subprocess
import unittest
from functools import partial, partialmethod
from unittest.mock import patch

import torch

from metaseq.cli.train import cli_main as train_cli_main
from metaseq.dataclass.configs import DistributedTrainingConfig
from metaseq.launcher.opt_baselines import cli_main as sweep_cli_main
from metaseq.launcher.opt_job_constants import Size, M

logger = logging.getLogger(__name__)


@unittest.skipIf(not torch.cuda.is_available(), "test requires 4 GPUs, none found")
@unittest.skipIf(
    DistributedTrainingConfig.distributed_world_size != 4,
    "test requires 4 GPUs",
)
class TestLoadOnMoreGPUs(unittest.TestCase):
    """
    The test will verify that the model can started with more GPUs
    from a checkpoint created with less GPUs. We test the first
    run with 2 GPUs and the second run is started with 4 GPUs
    from the previous checkpoint with 2GPUs.
    """

    def test_load_checkpoint(self):
        argv_injection = (
            "python3 metaseq/launcher/opt_baselines.py   "
            "--prefix train.8m    --model-size 8m    --checkpoints-dir ./test-checkpoint    "
            "--tensorboard-logdir ./test-checkpoint    --num-trials 1    --azure   "
            "--num-gpus 2 --num-nodes 1   --seed 1   "
            "--local --disable-validation    --max-epoch 5    --max-update 5 --benchmark    "
        )
        max_update_first_run = 20
        size_patch_dict = {"8m": Size(4, 128, 2, 64, int(0.03125 * M), 1.0e-3, 2)}

        training_log_events_first_run = self._helper(
            max_update=max_update_first_run,
            argv_injection=argv_injection,
            size_patch_dict=size_patch_dict,
        )

        # check that training ran correctly
        # check that the number of updates was correct
        self.assertNotEqual(training_log_events_first_run, [])
        self.assertIsNotNone(training_log_events_first_run[-1]["num_updates"])
        self.assertEqual(
            int(training_log_events_first_run[-1]["num_updates"]), max_update_first_run
        )
        # check the achieved loss is correct
        loss_val = float(training_log_events_first_run[-1]["loss"])
        self.assertAlmostEqual(loss_val, 14.744, 1)  # 1 digit precision

        # get the list of files from the chekpoint folder
        # from the 2 GPUs run: ./test-checkpoint/*.ngpu2
        first_run_checkpoints = subprocess.Popen(
            "ls -1 ./test-checkpoint/*.ngpu2",
            shell=True,  # this enables the * to be interpreted as a wildcard pattern
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
        )
        stdout_first, stderr_first = first_run_checkpoints.communicate()
        files_first = stdout_first.split("\n")
        # check that there are only 2 checkpoint files after running on 2 GPUs
        # ['checkpoint_18-model_part-0-shard0.pt', 'checkpoint_18-model_part-1-shard0.pt']
        checkpoints_first_num = len(
            [
                filename
                for filename in files_first
                if filename.startswith("checkpoint_18")
            ]
        )
        self.assertEqual(
            checkpoints_first_num,
            2,
            f"Expected 2 checkpoint files got {checkpoints_first_num}. List all files in the dir: {files_first}",
        )

        # start second run with 4 gpus from a previously created checkpoint
        # with 2 gpus from a "*.ngpu2/checkpoint_18.pt"
        argv_injection = (
            "python3 metaseq/launcher/opt_baselines.py   "
            "--prefix train.8m    --model-size 8m    --checkpoints-dir ./test-checkpoint    "
            "--tensorboard-logdir ./test-checkpoint    --num-trials 1    --azure   "
            "--num-gpus 4 --num-nodes 1   --seed 1   "
            "--local --disable-validation    --max-epoch 5    --max-update 5 --benchmark    "
            "--restore-file ./test-checkpoint/train.8m.dummy_lm.me_fp16.fsdp.zero2.relu."
            "transformer_lm_megatron.nlay4.emb128.lrnpos.0emb_scale.tps2048.adam.b2_0.95."
            "eps1e-08.cl1.0.lr0.001.endlr0.0001.wu50.dr0.1.atdr0.1.0emb_dr.wd0.1.ms16.uf1."
            "mu50.s1.me5.ngpu2/checkpoint_18.pt"
        )

        max_update_second_run = 40

        training_log_events_second_run = self._helper(
            max_update=max_update_second_run,
            argv_injection=argv_injection,
            size_patch_dict=size_patch_dict,
        )

        # check that training ran correctly
        self.assertNotEqual(training_log_events_second_run, [])

        # check the achieved loss is correct
        # check that loss is same as with 2 GPUs after reloading on 4 GPUs
        loss_val_start = float(training_log_events_second_run[0]["loss"])
        self.assertAlmostEqual(loss_val_start, 14.744, 1)  # 1 digit precision

        # check that loss improved during training on 4 GPUs
        loss_val_end = float(training_log_events_second_run[-1]["loss"])
        self.assertAlmostEqual(loss_val_end, 12.165, 1)  # 1 digit precision

        # get the list of files from the chekpoint folder
        # from the 4 GPUs run: ./test-checkpoint/*.ngpu4
        second_run_checkpoints = subprocess.Popen(
            "ls -1 ./test-checkpoint/*.ngpu4",
            shell=True,  # this enables the * to be interpreted as a wildcard pattern
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
        )
        stdout_second, _ = second_run_checkpoints.communicate()
        files_second = stdout_second.split("\n")

        # check that there are now 4 checkpoint files after running on 4 GPUs
        # and starting from the checkpoint folder with 2 checkpoints
        # ['checkpoint_36-model_part-0-shard0.pt', 'checkpoint_36-model_part-0-shard1.pt',
        # 'checkpoint_36-model_part-1-shard0.pt', 'checkpoint_36-model_part-1-shard1.pt'
        checkpoints_second_num = len(
            [
                filename
                for filename in files_second
                if filename.startswith("checkpoint_36")
            ]
        )
        self.assertEqual(
            checkpoints_second_num,
            4,
            f"Expected 4 checkpoint files got {checkpoints_second_num}. List all files in the dir: {files_second}",
        )

        # cleanup of the checkpoints files
        cleanup_checkpoints = subprocess.Popen(
            "rm -r ./test-checkpoint",
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
        )
        _, _ = cleanup_checkpoints.communicate()

    def _helper(self, max_update, argv_injection, size_patch_dict):
        """
        Helper function to run the test
        """
        # start the process for the model run
        multiprocessing.set_start_method("spawn", force=True)
        with torch.multiprocessing.Manager() as manager:
            events = manager.list()
            p = multiprocessing.Process(
                target=run_training,
                args=(max_update, events, argv_injection, size_patch_dict),
            )
            p.start()
            p.join()
            events_first_run = list(events)

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
    train_cmd[train_cmd.index("--save-interval-updates") + 1] = "18"

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
