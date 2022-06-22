# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging

import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel

from metaseq.distributed import (
    DistributedTimeoutWrapper,
    ModuleProxyWrapper,
)

logger = logging.getLogger(__name__)


_GOSSIP_DISABLED = False
try:
    import gossip  # noqa: F401
except ImportError:
    _GOSSIP_DISABLED = True


def DistributedModel(args, model, process_group, device):
    """
    Wrap a *model* to support distributed data parallel training.

    This is similar to the built-in DistributedDataParallel, but allows
    additional configuration of the DistributedDataParallel class to
    use, and also provides easier access to the wrapped model by
    forwarding requests for missing attributes to the wrapped model.

    Args:
        args (argparse.Namespace): metaseq args
        model (BaseModel): model to wrap
        process_group: the c10d process group to be used for distributed data
            parallel all-reduction.
        device: device to move model to
    """
    assert isinstance(model, nn.Module)
    if args.ddp_backend in {"c10d", "pytorch_ddp"}:
        wrapped_model = DistributedDataParallel(
            module=model.to(device),
            device_ids=[args.device_id],
            output_device=args.device_id,
            broadcast_buffers=args.broadcast_buffers,
            bucket_cap_mb=args.bucket_cap_mb,
            process_group=process_group,
            find_unused_parameters=args.find_unused_parameters,
        )
        # forward missing getattr and state_dict/load_state_dict to orig model
        wrapped_model = ModuleProxyWrapper(wrapped_model)
    elif args.ddp_backend == "fully_sharded":
        try:
            from fairscale.nn.data_parallel import FullyShardedDataParallel as FSDP
        except ImportError:
            raise ImportError(
                "Cannot find FullyShardedDataParallel. "
                "Please install fairscale with: pip install fairscale"
            )
        assert isinstance(model, FSDP), "expected model to already be wrapped in FSDP"
        wrapped_model = model
        if args.memory_efficient_fp16:
            if args.bf16:
                wrapped_model = wrapped_model.bfloat16()
            else:
                wrapped_model = wrapped_model.half()
        if not args.cpu_offload:
            wrapped_model = wrapped_model.to(device=device)
    elif args.ddp_backend == "ptd_fully_sharded":
        try:
            from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP
        except ImportError:
            raise ImportError(
                "Cannot find FullyShardedDataParallel. "
                "Please install fairscale with:  pip3 install torch torchvision torchaudio"
            )
        assert isinstance(model, FSDP), "expected model to already be wrapped in FSDP"
        wrapped_model = model
        if args.memory_efficient_fp16:
            wrapped_model = wrapped_model.half()
        if not args.cpu_offload:
            wrapped_model = wrapped_model.to(device=device)
    else:
        raise ValueError("Unknown --ddp-backend: " + args.ddp_backend)

    # kill hung distributed jobs after a timeout
    if getattr(args, "heartbeat_timeout", -1) > 0:
        wrapped_model = DistributedTimeoutWrapper(
            wrapped_model, timeout=getattr(args, "heartbeat_timeout", -1)
        )

    return wrapped_model
