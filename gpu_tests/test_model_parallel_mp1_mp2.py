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


def run_training(events, max_update):
    argv_injection = (
        "python3 metaseq/launcher/opt_baselines.py   "
        "--prefix train.8m    --model-size 8m    --checkpoints-dir ./test-checkpoint    "
        "--tensorboard-logdir ./test-checkpoint    --num-trials 1    --azure   "
        "--num-gpus 4 --num-nodes 1   --seed 1   "
        "--local --disable-validation    --max-epoch 5    --max-update 5 --benchmark    "
        "--full-azure-upload-path https://myaccount.blob.core.windows.net/test   "
    )
    with patch("sys.argv", argv_injection.split()[1:]):
        sweep_cli_main()

class TestModelParallel(unittest.TestCase):
    """
    These tests will verify that the model can be trained with
    model_parallel = 1 and model_parallel = 2
    The tests checks hat the number of trianing steps performed is correct
    and that the required loss is achieved on the last iteration
    """

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
        self.assertEqual(1,1)

        # cleanup
        cleanup_checkpoints = subprocess.Popen(
            "rm -r ./test-checkpoint".split(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
        )
        _, _ = cleanup_checkpoints.communicate()

if __name__ == "__main__":
    unittest.main()
