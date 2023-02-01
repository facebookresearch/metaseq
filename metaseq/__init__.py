# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""isort:skip_file"""

import os
import sys

try:
    from .version import __version__  # noqa
except ImportError:
    version_txt = os.path.join(os.path.dirname(__file__), "version.txt")
    with open(version_txt) as f:
        __version__ = f.read().strip()

__all__ = ["pdb"]

# backwards compatibility to support `from metaseq.X import Y`
from metaseq.distributed import utils as distributed_utils
from metaseq.logging import meters, metrics  # noqa

sys.modules["metaseq.distributed_utils"] = distributed_utils
sys.modules["metaseq.meters"] = meters
sys.modules["metaseq.metrics"] = metrics

# initialize hydra
from metaseq.dataclass.initialize import hydra_init  # noqa: E402

hydra_init()

import metaseq.criterions  # noqa
import metaseq.distributed  # noqa
import metaseq.models  # noqa
import metaseq.modules  # noqa
import metaseq.optim  # noqa
import metaseq.optim.lr_scheduler  # noqa
import metaseq.pdb  # noqa
import metaseq.tasks  # noqa

import metaseq.benchmark  # noqa
