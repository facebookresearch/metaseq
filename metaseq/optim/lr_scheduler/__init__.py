# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""isort:skip_file"""

import importlib
import os

from metaseq import registry
from metaseq.optim.lr_scheduler.base_lr_scheduler import BaseLRScheduler
from omegaconf import DictConfig


(
    build_lr_scheduler_,
    register_lr_scheduler,
    LR_SCHEDULER_REGISTRY,
    LR_SCHEDULER_DATACLASS_REGISTRY,
) = registry.setup_registry(
    "--lr-scheduler", base_class=BaseLRScheduler, default="fixed"
)


def build_lr_scheduler(cfg: DictConfig, optimizer):
    return build_lr_scheduler_(cfg, optimizer)


# automatically import any Python files in the optim/lr_scheduler/ directory
for file in os.listdir(os.path.dirname(__file__)):
    if file.endswith(".py") and not file.startswith("_"):
        file_name = file[: file.find(".py")]
        importlib.import_module("metaseq.optim.lr_scheduler." + file_name)
