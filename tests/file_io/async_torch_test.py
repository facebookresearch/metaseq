#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
    Utility for testing async writes with `torch.save`.
    Usage:
        buck run @mode/opt //fair_infra/data/iopath/tests:async_torch_test
"""

import os
import tempfile
import time

import torch
import torch.nn as nn
import torch.optim as optim
from metaseq.file_io.common import PathManager


class Model(nn.Module):
    # pyre-fixme[3]: Return type must be annotated.
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)


class TestDriver:
    _pathmgr = PathManager()

    def test(self) -> None:
        model = Model()
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

        print("Model's state_dict:")
        for param_tensor in model.state_dict():
            print(f"{param_tensor}\t{model.state_dict()[param_tensor].size()}")
        print("Optimizer's state_dict:")
        for var_name in optimizer.state_dict():
            print(f"{var_name}\t{optimizer.state_dict()[var_name]}")

        with tempfile.TemporaryDirectory() as _tmpdir:
            try:
                URI = os.path.join(_tmpdir, "test.ckpt")

                f = self._pathmgr.opena(URI, "wb")
                i = "*"
                large = f"{i}" * 1000000000

                print("Starting `torch.save` call.")
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "large": large,
                    },
                    # pyre-fixme[6]: For 2nd param expected
                    #  `Union[PathLike[typing.Any], IO[bytes], str, BinaryIO]` but got
                    #  `IOBase`.
                    f,
                )
                f.close()
                start_time = time.time()

            finally:
                # We want this `join` call to take time. If it is instantaneous,
                # then our async write calls are not running asynchronously.
                print("Waiting for `torch.save` call to complete at `async_join()`.")
                self._pathmgr.async_join()

            print(
                "Time Python waited for `async_join()` call to finish: "
                f"{time.time() - start_time}s."
            )
            assert self._pathmgr.async_close()

            checkpoint = torch.load(URI)
            for key_item_1, key_item_2 in zip(
                model.state_dict().items(), checkpoint["model_state_dict"].items()
            ):
                assert torch.equal(key_item_1[1], key_item_2[1])
            assert optimizer.state_dict() == checkpoint["optimizer_state_dict"]
            assert large == checkpoint["large"]

            print("Async `torch.save` Test succeeded.")


if __name__ == "__main__":
    print("Async `torch.save` Test starting.")
    tst = TestDriver()
    tst.test()
