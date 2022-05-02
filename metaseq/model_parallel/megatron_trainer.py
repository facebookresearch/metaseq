# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Train a network across multiple GPUs.
"""

import functools

import torch.distributed as dist

from metaseq.dataclass.configs import MetaseqConfig
from metaseq.distributed import utils as distributed_utils
from metaseq.trainer import Trainer

try:
    from megatron.mpu import (
        get_cuda_rng_tracker,
    )

    has_megatron_submodule = True
except (ImportError, ModuleNotFoundError):
    has_megatron_submodule = False


class MegatronTrainer(Trainer):
    """Main class for model parallel with data parallel training."""

    def __init__(self, cfg: MetaseqConfig, task, model, criterion, **kwargs):
        if not has_megatron_submodule:
            raise ImportError(
                "\n\nPlease install megatron using the setup instructions!"
            )
        super().__init__(cfg, task, model, criterion, **kwargs)

    def clip_grad_norm(
        self, clip_norm, norm_type="l2", skip_gradient_update_on_clip_norm=False
    ):
        def _aggregate_model_parallel_grad_norm(norm_type, total_norm):
            norm_type2_reduce_op = {"l2": dist.ReduceOp.SUM, "inf": dist.ReduceOp.MAX}
            reduce_op = norm_type2_reduce_op[norm_type]
            if norm_type == "l2":
                total_norm.pow_(2)
            dist.all_reduce(
                total_norm,
                group=distributed_utils.get_model_parallel_group(),
                op=reduce_op,
            )
            if norm_type == "l2":
                total_norm.sqrt_()
            return total_norm

        return self.optimizer.clip_grad_norm(
            clip_norm,
            norm_type,
            aggregate_norm_fn=functools.partial(
                _aggregate_model_parallel_grad_norm, norm_type
            ),
            skip_gradient_update_on_clip_norm=skip_gradient_update_on_clip_norm,
        )

    def save_checkpoint(self, filename, extra_state, **kwargs):
        """Save all training state in a checkpoint file."""
        extra_state["rng_tracker_states"] = get_cuda_rng_tracker().get_states()
        super().save_checkpoint(filename, extra_state, **kwargs)

    def load_checkpoint(
        self,
        filename,
        reset_optimizer=False,
        reset_lr_scheduler=False,
        optimizer_overrides=None,
        reset_meters=False,
    ):
        extra_state = super().load_checkpoint(
            filename,
            reset_optimizer=reset_optimizer,
            reset_lr_scheduler=reset_lr_scheduler,
            optimizer_overrides=optimizer_overrides,
            reset_meters=reset_meters,
        )
        if extra_state is not None and "rng_tracker_states" in extra_state:
            get_cuda_rng_tracker().set_states(extra_state["rng_tracker_states"])
        return extra_state
