# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import re
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

class TestModelParallel(unittest.TestCase):
    """
    These tests will verify that the model can be trained with
    model_parallel = 1 and model_parallel = 2
    The tests checks hat the number of trianing steps performed is correct
    and that the required loss is achieved on the last iteration
    """

    def test_model_parallel_mp2(self):
        # run a 8M model with 2 model parallel (mp2)
        mp1_results = subprocess.Popen(
            "python3 metaseq/launcher/opt_baselines.py \
            --prefix train.8m --model-size 8m_mp1 --checkpoints-dir ./test-checkpoint \
            --tensorboard-logdir ./test-checkpoint --num-trials 1 --azure \
            --num-gpus 4 --num-nodes 1 --seed 1 \
            --local --disable-validation --max-epoch 5 --max-update 5 --benchmark".split(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
        )
        mp1_stdout, mp1_stderr = mp1_results.communicate()
        print("mp1_stderr:", mp1_stderr)
        print("mp1_stdout:", mp1_stdout)
        # cleanup generated checkpoints files
        cleanup_checkpoints = subprocess.Popen(
            "rm -r ./test-checkpoint".split(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
        )
        _, _ = cleanup_checkpoints.communicate()

        # mp1: check that the training was successfull
        training_mp1_log_events = re.findall(r'{"epoch.*"}', mp1_stdout)
        last_epoch_mp1_loss = float(json.loads(training_mp1_log_events[-1])["loss"])

        # check that number of steps performed is correct
        self.assertEqual(len(training_mp1_log_events), 10)
        # check that the achieved loss is correct
        self.assertAlmostEqual(
            last_epoch_mp1_loss, 10.48, 1
        )  # one decimal point precision


if __name__ == "__main__":
    unittest.main()
