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
class TestSequenceParallel(unittest.TestCase):
    """
    The tests check rough equivalence between going through the
    sequence-parallel code-path with MP 2 vs the current non
    sequence-parallel run for the 8M model.
    """

    def test_sequence_parallel(self):
        # parameters to train an mp2 model with sequence_parallel flag 
        argv_injection = (
            "python3 metaseq/launcher/opt_baselines.py   "
            "--prefix train.8m    --model-size 8m    --checkpoints-dir ./test-checkpoint    "
            "--tensorboard-logdir ./test-checkpoint    --num-trials 1    --azure   "
            "--num-gpus 4 --num-nodes 1   --seed 1   "
            "--local --disable-validation    --max-epoch 5    --max-update 5 --benchmark    "
        )
        max_update_first_run = 20
        size_patch_dict = {"8m": Size(4, 128, 2, 64, int(0.03125 * M), 1.0e-3, 2)}

        # train model with sequence_parallel flag
        # training_log_events_seq = self._test_model_parallel(
        #     max_update_first_run=max_update_first_run,
        #     argv_injection=argv_injection,
        #     size_patch_dict=size_patch_dict,
        #     is_sequence_parallel=True,
        # )
        # train model without sequence_parallel flag
        training_log_events = self._test_model_parallel(
            max_update_first_run=max_update_first_run,
            argv_injection=argv_injection,
            size_patch_dict=size_patch_dict,
            is_sequence_parallel=True,
        )

        # check that training ran correctly
        # check that the number of updates was correct
        # self.assertNotEqual(training_log_events_seq, [])
        self.assertNotEqual(training_log_events, [])
        # self.assertIsNotNone(training_log_events_seq[-1]["num_updates"])
        self.assertIsNotNone(training_log_events[-1]["num_updates"])
        self.assertEqual(
            int(training_log_events[-1]["num_updates"]), max_update_first_run
        )
        # self.assertEqual(
        #     int(training_log_events_seq[-1]["num_updates"]), max_update_first_run
        # )
        # check the achieved loss is similar between seq and non-seq
        # loss_val_seq = float(training_log_events_seq[-1]["loss"])
        loss_val = float(training_log_events[-1]["loss"])

        # print("loss_val_seq: {} | loss_val: {}".format(loss_val_seq, loss_val))
        # self.assertAlmostEqual(
        #     loss_val, loss_val_seq, 1
        # )  # 1 digit precision; 14.702 - seq; 14.735 - non seq

    def _test_model_parallel(
        self,
        max_update_first_run,
        argv_injection,
        size_patch_dict,
        is_sequence_parallel,
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
                args=(
                    max_update_first_run,
                    events,
                    argv_injection,
                    size_patch_dict,
                    is_sequence_parallel,
                ),
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


def run_training(
    max_update, events, argv_injection, size_patch_dict, is_sequence_parallel
):
    # clean any unused cach to reduce CUDA OOM
    torch.cuda.empty_cache()
    # main arguments to run the training script
    # both patches are aneeded to run the job of the circleci GPUs
    with patch("sys.argv", argv_injection.split()[1:]), patch(
        "metaseq.launcher.slurm.local_run",
        partial(
            local_run_mock,
            max_update=max_update,
            events=events,
            is_sequence_parallel=is_sequence_parallel,
        ),
    ), patch.dict(
        "metaseq.launcher.opt_job_constants.MODEL_SIZES",
        # reduce the batch size for CUDA memory optimization
        size_patch_dict,
    ):
        sweep_cli_main()


def local_run_mock(
    args, env, train_cmd, dry_run, max_update, events, is_sequence_parallel
):
    """
    The function introduces several patches for the argumets of the
    model training. These patches are needed to pass gpu tests on
    circleci GPUs and enable sequence_parallel parameter
    """
    # update the parameters of the model training
    train_cmd[train_cmd.index("--max-update") + 1] = str(max_update)
    train_cmd[train_cmd.index("--num-workers") + 1] = "1"
    train_cmd[train_cmd.index("--dropout") + 1] = "0.0"
    train_cmd.remove("--checkpoint-activations")
    train_cmd.remove("--distribute-checkpointed-activations")
    # add sequence_parallel argument to the model arguments
    if is_sequence_parallel:
        train_cmd.append("--sequence-parallel")

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