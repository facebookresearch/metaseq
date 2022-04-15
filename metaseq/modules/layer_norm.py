# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
from metaseq import distributed_utils as dist_utils
from typing import Tuple
import torch.distributed as dist

try:
    from apex.normalization import FusedLayerNorm as _FusedLayerNorm

    has_fused_layernorm = True

    class FusedLayerNorm(_FusedLayerNorm):
        @torch.jit.unused
        def forward(self, x):
            if not x.is_cuda:
                return super().forward(x)
            else:
                with torch.cuda.device(x.device):
                    return super().forward(x)

except ImportError:
    has_fused_layernorm = False


def LayerNorm(normalized_shape, eps=1e-5, elementwise_affine=True, export=False):
    if torch.jit.is_scripting():
        export = True
    if not export and torch.cuda.is_available() and has_fused_layernorm:
        return FusedLayerNorm(normalized_shape, eps, elementwise_affine)
    return torch.nn.LayerNorm(normalized_shape, eps, elementwise_affine)


class Fp32LayerNorm(nn.LayerNorm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input):
        output = F.layer_norm(
            input.float(),
            self.normalized_shape,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps,
        )
        return output.type_as(input)


class SyncedModelParallelFusedLayerNorm(nn.Module):
    def __init__(
        self,
        hidden_size,
        mp_size,
        mp_rank=0,
        eps=1e-5,
        initialize_params_on_gpu=False,
        use_bias=True,
        mean_center=True,
    ):
        super().__init__()
        assert hidden_size % mp_size == 0
        partition_size = hidden_size // mp_size
        self.use_bias = use_bias
        self.mean_center = mean_center
        self.weight = nn.Parameter(torch.ones(partition_size, dtype=torch.float32))
        self.bias = (
            nn.Parameter(torch.zeros(partition_size, dtype=torch.float32))
            if self.use_bias
            else None
        )
        self.variance_epsilon = eps
        self.mp_world_size = float(dist_utils.get_model_parallel_world_size())
        self.mp_rank = mp_rank
        if initialize_params_on_gpu:
            self.weight.cuda().half()
            if self.bias is not None:
                self.bias.cuda().half()

    @staticmethod
    def get_statistics_from_all_workers(stat, rank, world_size):
        """Retuns tensor shaped (world_size, *stat_size)"""
        buffer_size: Tuple[int] = (int(world_size),) + tuple(stat.size())
        assert isinstance(
            buffer_size, tuple
        ), f"b{buffer_size} {world_size} {stat.size()}"
        buffer = torch.zeros(buffer_size, dtype=stat.dtype, device=stat.device)
        buffer[rank] = stat
        dist.all_reduce(buffer, group=dist_utils.get_model_parallel_group())
        return buffer

    def forward(self, hidden_states):
        hid_fp32 = hidden_states.float()
        local_variance = torch.var(hid_fp32, -1, keepdim=True, unbiased=True)
        local_mean = hid_fp32.mean(-1, keepdim=True)
        vs = SyncedModelParallelFusedLayerNorm.get_statistics_from_all_workers(
            local_variance, self.mp_rank, self.mp_world_size
        )
        ms = SyncedModelParallelFusedLayerNorm.get_statistics_from_all_workers(
            local_mean, self.mp_rank, self.mp_world_size
        )
        variance, mean = variance_formula(
            ms, vs, self.mp_world_size, hidden_states.size(-1)
        )
        denom = torch.rsqrt(variance + self.variance_epsilon).to(hidden_states.dtype)
        mean = mean.to(hidden_states.dtype)

        if self.mean_center:
            hidden_states = (hidden_states - mean) * denom
        else:
            hidden_states = hidden_states * denom

        if self.use_bias:
            return (self.weight * hidden_states) + self.bias
        else:
            return self.weight * hidden_states


def variance_formula(means, vs, g, k) -> Tuple[torch.Tensor]:
    """This only works with unbiased=False (No Bessel Correction)"""
    d = g * k
    var_ej = means.var(0)  # Need unbiased True here (at least in  toy example)
    summation = vs.sum(0)
    inner_coeff: float = (k * (g - 1)) / (k - 1)
    outer_coeff: float = (k - 1) / (d - 1)
    out: torch.Tensor = outer_coeff * (summation + (inner_coeff * var_ej))
    return out, means.mean(0)
