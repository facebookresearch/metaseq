# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest
import torch
from metaseq.dataclass.configs import DistributedTrainingConfig
import subprocess
import urllib.request
import tarfile


@unittest.skipIf(not torch.cuda.is_available(), "test requires a GPU")
@unittest.skipIf(
    DistributedTrainingConfig.distributed_world_size != 8,
    "test requires at least 8 GPU's",
)
class TestTraining(unittest.TestCase):
    def test_training(self):
        link_to_data = (
            "https://dl.fbaipublicfiles.com/metaseq-train-integration-test.tar.gz"
        )
        urllib.request.urlretrieve(link_to_data, "files.tar.gz")
        file = tarfile.open("files.tar.gz")
        file.extractall("./gpu_tests")
        file.close()
        command = (
            "python3 metaseq/launcher/opt_baselines.py   "
            "--prefix train.8m --model-size 8m    --checkpoints-dir ./test-checkpoint    "
            "--tensorboard-logdir ./test-checkpoint     --num-trials 1  --azure   --num-gpus 8 --num-nodes 1   --seed 1   "
            "--circleci --local --disable-validation --max-epoch 5 --max-update 5"
        )
        p = subprocess.Popen(
            command.split(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
        )
        outs, errs = p.communicate()
        print(outs)
        print(errs)
        outs_list = outs.split()
        outs_list.reverse()
        ind = outs_list.index('"loss":')
        ans = outs_list[ind - 1][1:-2]
        ans = float(ans)
        self.assertAlmostEqual(ans, 15.601)  # assertion of loss after 10 iterations
        assert "done training" in outs  # assertion of training completion succesfully
        cleanup_checkpoints = subprocess.Popen(
            "rm -r ./test-checkpoint".split(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
        )
        _, _ = cleanup_checkpoints.communicate()

        cleanup_tarball = subprocess.Popen(
            "rm files.tar.gz".split(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
        )
        _, _ = cleanup_tarball.communicate()

        cleanup_files = subprocess.Popen(
            "rm -r ./gpu_tests/circleci".split(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
        )

        _, _ = cleanup_files.communicate()


if __name__ == "__main__":
    unittest.main()
