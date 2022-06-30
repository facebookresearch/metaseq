# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest
import torch
from metaseq.dataclass.configs import DistributedTrainingConfig
import subprocess


@unittest.skipIf(not torch.cuda.is_available(), "test requires a GPU")
@unittest.skipIf(
    DistributedTrainingConfig.distributed_world_size != 8,
    "test requires at least 8 GPU's",
)
class TestTraining(unittest.TestCase):
    def test_training(self):
        command = (
            "python3 metaseq/launcher/opt_baselines.py   "
            "--prefix xlmg.try.cm3 --model-size 8m    --checkpoints-dir ./test-checkpoint    "
            "--tensorboard-logdir ./test-checkpoint     --num-trials 1  --azure   --num-gpus 8 --num-nodes 1   --seed 1   "
            "--partition xlmg --circleci --local --disable-validation --max-epoch 5 --max-update 5"
        )
        p = subprocess.Popen(
            command.split(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
        )
        outs, errs = p.communicate()
        outs_list = outs.split()
        outs_list.reverse()
        ind = outs_list.index('"loss":')
        ans = outs_list[ind - 1][1:-2]
        ans = float(ans)
        self.assertAlmostEqual(ans, 15.601)  # assertion of loss after 10 iterations
        assert "done training" in outs  # assertion of training completion succesfully
        r = subprocess.Popen(
            "rm -r ./test-checkpoint".split(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
        )
        _, _ = r.communicate()


if __name__ == "__main__":
    unittest.main()
