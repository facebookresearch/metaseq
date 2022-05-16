# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os
import logging

from .distributed_timeout_wrapper import DistributedTimeoutWrapper
from .module_proxy_wrapper import ModuleProxyWrapper

logger = logging.getLogger(__name__)
if os.environ.get("USE_PTD_FSDP", "False") == "True":
    logger.info("Use PyTorch Distributed FSDP")
    from .ptd_fully_sharded_data_parallel import (
        fsdp_enable_wrap,
        fsdp_wrap,
        FullyShardedDataParallel,
    )
else:
    logger.info("Use FairScale FSDP")
    from .fully_sharded_data_parallel import (
        fsdp_enable_wrap,
        fsdp_wrap,
        FullyShardedDataParallel,
    )



__all__ = [
    "DistributedTimeoutWrapper",
    "fsdp_enable_wrap",
    "fsdp_wrap",
    "FullyShardedDataParallel",
    "ModuleProxyWrapper",
]
