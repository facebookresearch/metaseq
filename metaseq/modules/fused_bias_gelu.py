# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os

import torch

logger = logging.getLogger(__name__)


try:
    from megatron.model.fused_bias_gelu import bias_gelu_impl

    has_fused_bias_gelu = True
except ImportError:
    has_fused_bias_gelu = False


def load_megatron_fused_kernel():
    """Compile and load fused kernels from Megatron."""
    if getattr(load_megatron_fused_kernel, "has_run", False):
        return
    load_megatron_fused_kernel.has_run = True

    from megatron import fused_kernels
    from argparse import Namespace

    if not hasattr(fused_kernels, "load"):
        # verion of megatron that has them precompiled, we cans skip this...
        return

    if not torch.distributed.is_initialized():
        # gradient_accumulation_fusion=False is an arg added to latest megatron for
        # fused gradient accumulation, we ccurrently dont need it as we almost always
        # use update-freq=1
        args = Namespace(
            rank=0, masked_softmax_fusion=True, gradient_accumulation_fusion=False
        )
        fused_kernels.load(args)
        return

    global_rank = torch.distributed.get_rank()
    args = Namespace(
        rank=global_rank, masked_softmax_fusion=True, gradient_accumulation_fusion=False
    )

    # Always build on rank zero first.
    if global_rank == 0:
        build_dir = os.path.join(os.path.dirname(fused_kernels.__file__), "build")
        logger.info(
            "Compiling and loading fused kernels\n\n"
            "NOTE: If this hangs here, your megatron fused kernels may be corrupted. "
            "This can happen if a previous job is interrupted during a build. "
            "In that case, delete the megatron build directory and relaunch training. "
            f"The megatron build directory is located at: {build_dir}"
        )
        fused_kernels.load(args)
        torch.distributed.barrier()
    else:
        torch.distributed.barrier()
        fused_kernels.load(args)

    # Simple barrier to make sure all ranks have passed the
    # compilation phase successfully before moving on to the
    # rest of the program. We think this might ensure that
    # the lock is released.
    torch.distributed.barrier()

    logger.info("Done with compiling and loading fused kernels.")


def fused_bias_gelu(x, bias):
    if not has_fused_bias_gelu:
        raise ImportError(
            "Cannot find fused Megatron kernels, please install Megatron from: "
            "github.com/NVIDIA/Megatron-LM"
        )
    load_megatron_fused_kernel()
    return bias_gelu_impl(x, bias)
