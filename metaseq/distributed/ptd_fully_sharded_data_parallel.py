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
from metaseq.distributed import utils as dist_utils

logger = logging.getLogger(__name__)

# TODO (mingzhe):
# update state_dict interface.

try:
    from torch.distributed.fsdp.fully_sharded_data_parallel  import (
        FullyShardedDataParallel as FSDP,
        StateDictType,
        FullStateDictConfig
    )
    from fairscale.utils.testing import DummyProcessGroup

    has_FSDP = True
except ImportError:
    FSDP = torch.nn.Module
    has_FSDP = False


class FullyShardedDataParallel(FSDP):
    """
    A small wrapper around PyTorch Distributed FullyShardedDataParallel (FSDP) with some
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
                "Please install PyTorch with: pip3 install torch torchvision torchaudio"
            )
        super().__init__(*args, **kwargs)
        self.use_sharded_state = use_sharded_state

    @property
    def unwrapped_module(self) -> torch.nn.Module:
        return self.module.module

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        # TODO (mingzhe)
        # this needs to be updated.
        if self.use_sharded_state:
            with FSDP.state_dict_type(self, StateDictType.LOCAL_STATE_DICT):
                return super().state_dict(
                    destination=destination, prefix=prefix, keep_vars=keep_vars
                )
        else:
            # TODO (mingzhe)
            # this needs to be updated.
            full_state_dict_config = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            with FSDP.state_dict_type(self, StateDictType.FULL_STATE_DICT, full_state_dict_config):
                return super().state_dict(
                    destination=destination, prefix=prefix, keep_vars=keep_vars
                )

    def load_state_dict(self, state_dict, strict=True, model_cfg=None):
        # TODO (mingzhe)
        # update
        if self.use_sharded_state:
            with FSDP.state_dict_type(self, StateDictType.LOCAL_STATE_DICT):
                return super().load_local_state_dict(state_dict, strict=strict)
        else:
            with FSDP.state_dict_type(self, StateDictType.FULL_STATE_DICT):
                return super().load_state_dict(state_dict, strict=strict)


@contextlib.contextmanager
def fsdp_enable_wrap(
    cfg: DistributedTrainingConfig, use_sharded_state: bool = False, **kwargs
):
    try:
        from torch.distributed.fsdp.wrap import enable_wrap
    except ImportError:
        raise ImportError(
            "Cannot find FullyShardedDataParallel. "
            "Please install PyTorch with: pip3 install torch torchvision torchaudio"
        )
    if cfg.memory_efficient_fp16:
        assert cfg.fp16  # memory_efficient_fp16 should imply fp16
    group = dist_utils.get_data_parallel_group()
    if group is None and cfg.distributed_world_size == 1:
        group = DummyProcessGroup(rank=0, size=1)
    # TODO (mingzhe)
    # use PTD FSDP configs.
    fsdp_config = {
        "process_group": group,
        # "reshard_after_forward": not cfg.no_reshard_after_forward,
        # "mixed_precision": cfg.fp16 and not cfg.memory_efficient_fp16,
        # "fp32_reduce_scatter": cfg.fp32_reduce_scatter,
        # "flatten_parameters": True,
        # "cpu_offload": cfg.cpu_offload and not cfg.memory_efficient_fp16,
        # "compute_dtype": torch.float16 if cfg.fp16 else torch.float32,
        # "bucket_cap_mb": cfg.bucket_cap_mb,
        # "state_dict_device": torch.device("cpu"),
        # "gradient_predivide_factor": cfg.gradient_predivide_factor,
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
