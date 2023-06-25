# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import logging
import os
from typing import Optional

import torch

from metaseq.dataclass.configs import DistributedTrainingConfig
from metaseq.distributed import utils as distributed_utils

logger = logging.getLogger(__name__)

try:
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy
    from fairscale.utils.testing import DummyProcessGroup

    has_FSDP = True
except ImportError:
    FSDP = torch.nn.Module
    has_FSDP = False


class FullyShardedDataParallel(FSDP):
    """
    A small wrapper around fairscale's FullyShardedDataParallel (FSDP) with some
    metaseq-specific checkpoint saving/loading logic.

    Args:
        use_sharded_state (bool): if True, then ``state_dict`` will return
            ``FSDP.local_state_dict`` and ``load_state_dict`` will call
            ``FSDP.load_local_state_dict``. Otherwise, ``state_dict`` will
            return the full model weights on data parallel rank 0 (empty on
            other ranks) and ``load_state_dict`` will broadcast model weights
            from rank 0 to other ranks.
    """

    def __init__(self, *args, use_sharded_state: bool = False, **kwargs):
        if not has_FSDP:
            raise ImportError(
                "Cannot find FullyShardedDataParallel. "
                "Please install fairscale with: pip install fairscale"
            )
        super().__init__(*args, **kwargs)
        self.use_sharded_state = use_sharded_state

    @property
    def unwrapped_module(self) -> torch.nn.Module:
        if self.flatten_parameters:
            return self.module.module
        else:
            return self.module

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        return super().state_dict(
            destination=destination, prefix=prefix, keep_vars=keep_vars
        )
        # if self.use_sharded_state:
        #     return super().local_state_dict(
        #         destination=destination, prefix=prefix, keep_vars=keep_vars
        #     )
        # else:
        #     if self.rank == 0:
        #         return super().state_dict(
        #             destination=destination, prefix=prefix, keep_vars=keep_vars
        #         )
        #     else:
        #         # We must call state_dict() due to use of communication
        #         # primitives. But we don't use the result.
        #         super().state_dict()
        #         return destination or {}

    def load_state_dict(self, state_dict, strict=True):
        # if self.use_sharded_state:
        #     return super().load_local_state_dict(state_dict, strict=strict)
        # else:
        if not isinstance(self.process_group, DummyProcessGroup):
            state_dict = distributed_utils.broadcast_object(
                state_dict, src_rank=0, group=self.process_group
            )
        return super().load_state_dict(state_dict, strict=strict)


@contextlib.contextmanager
def fsdp_enable_wrap(
    cfg: DistributedTrainingConfig, use_sharded_state: bool = False, **kwargs
):
    try:
        from torch.distributed.fsdp.wrap import enable_wrap
        from torch.distributed.fsdp import MixedPrecision
    except ImportError:
        raise ImportError(
            "Cannot find FullyShardedDataParallel. "
            "Please install fairscale with: pip install fairscale"
        )
    if cfg.memory_efficient_fp16:
        assert cfg.fp16  # memory_efficient_fp16 should imply fp16
    group = distributed_utils.get_data_parallel_group()
    if group is None and cfg.distributed_world_size == 1:
        group = DummyProcessGroup(rank=0, size=1)
    if cfg.fp16:
        compute_dtype = torch.bfloat16 if cfg.bf16 else torch.float16
    else:
        compute_dtype = torch.float32
    mixed_precision = MixedPrecision(param_dtype=compute_dtype,
                                     reduce_dtype=torch.float32 if cfg.fp32_reduce_scatter else compute_dtype,
                                     buffer_dtype=compute_dtype,
                                     keep_low_precision_grads=cfg.fp16 and cfg.memory_efficient_fp16)
    fsdp_config = {
        "process_group": group,
        "sharding_strategy": ShardingStrategy.NO_SHARD,
        "mixed_precision": mixed_precision,
        "cpu_offload": cfg.cpu_offload and not cfg.memory_efficient_fp16,
        # "use_orig_params": True,
        **kwargs,
    }
    with enable_wrap(
        wrapper_cls=FullyShardedDataParallel,
        use_sharded_state=use_sharded_state,
        **fsdp_config,
    ):
        yield


def fsdp_wrap(module, min_num_params: Optional[int] = None, **kwargs):
    """
    Helper to wrap layers/modules in FSDP. This falls back to a no-op if
    fairscale is not available.

    Args:
        module (nn.Module): module to (maybe) wrap
        min_num_params (int, Optional): minimum number of layer params to wrap
    """
    try:
        from torch.distributed.fsdp.wrap import wrap

        if os.environ.get("RESHARD_OVERRIDE_PROCESS_GROUP", "False") == "True":
            logger.info("Process group was None, overriding to DummyProcessGroup")
            kwargs["process_group"] = DummyProcessGroup(rank=0, size=1)

        if min_num_params is not None:
            num_params = sum(p.numel() for p in module.parameters())
            if num_params >= min_num_params:
                return wrap(module, **kwargs)
            else:
                return module
        else:
            return wrap(module, **kwargs)
    except ImportError:
        return module
