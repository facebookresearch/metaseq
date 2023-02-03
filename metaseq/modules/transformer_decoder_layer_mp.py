# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
from torch import nn

from metaseq import utils
from metaseq.modules import ModelParallelMultiheadAttention
from metaseq.modules import TransformerDecoderLayer

try:
    from megatron.mpu import (
        ColumnParallelLinear,
        RowParallelLinear,
    )

    has_megatron_submodule = True
except (ImportError, ModuleNotFoundError):
    has_megatron_submodule = False


def _weight_init(weight):
    return nn.init.kaiming_uniform_(weight, a=math.sqrt(5))


class ModelParallelTransformerDecoderLayer(TransformerDecoderLayer):
    """Decoder layer block.

    See "Megatron-LM: https://arxiv.org/pdf/1909.08053.pdf" for more details.
    """

    # TODO[susanz]: unify method signatures with non-model-parallel version.
    def build_fc1(
        self,
        input_dim,
        output_dim,
        initialize_params_on_gpu,
        full_megatron_init,
        megatron_init_sigma,
        dtype,
        disable_bias=False,
        truncate_init=False,
    ):
        if not has_megatron_submodule:
            raise ImportError(
                "\n\nPlease install megatron using the setup instructions!"
            )

        def _init_method_bias(bias):
            fan_in = input_dim
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(bias, -bound, bound)

        if full_megatron_init:
            # Setting bias init method to None, initializes biases with zero.
            init_method_weights = utils.init_method_normal(
                megatron_init_sigma, truncate_init=truncate_init
            )
            init_method_bias = None
        else:
            init_method_weights = _weight_init
            init_method_bias = _init_method_bias

        return ColumnParallelLinear(
            input_dim,
            output_dim,
            gather_output=False,
            init_method=init_method_weights,
            skip_bias_add=self.skip_bias_add,
            init_method_bias=init_method_bias,
            use_cpu_initialization=not initialize_params_on_gpu,
            dtype=dtype,
            bias=not disable_bias,
        )

    # TODO[susanz]: unify method signatures with non-model-parallel version.
    # Note: only fc2 includes the full_megatron_init_scalar arg, given the scaled_init_method_normal call.
    def build_fc2(
        self,
        input_dim,
        output_dim,
        initialize_params_on_gpu,
        full_megatron_init,
        full_megatron_init_scalar,
        megatron_init_sigma,
        num_layers,
        dtype,
        disable_bias=False,
        truncate_init=False,
    ):
        if not has_megatron_submodule:
            raise ImportError(
                "\n\nPlease install megatron using the setup instructions!"
            )

        skip_bias_add = self.skip_bias_add
        if full_megatron_init:
            init_method_weights = utils.scaled_init_method_normal(
                megatron_init_sigma * full_megatron_init_scalar,
                num_layers,
                truncate_init=truncate_init,
            )
        else:
            init_method_weights = _weight_init

        fc2 = RowParallelLinear(
            input_dim,
            output_dim,
            input_is_parallel=True,
            init_method=init_method_weights,
            skip_bias_add=skip_bias_add,
            use_cpu_initialization=not initialize_params_on_gpu,
            bias=not disable_bias,
            dtype=dtype,
        )
        if not full_megatron_init:
            # Copy nn.linear initialization to get same initialization as of non-model-parallel.
            # fan_in, _ = nn.init._calculate_fan_in_and_fan_out(fc2.weight)
            fan_in = input_dim
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(fc2.bias, -bound, bound)
        return fc2

    def build_self_attention(self, embed_dim, args, **unused_kwargs):
        return ModelParallelMultiheadAttention(
            embed_dim=embed_dim,
            num_heads=args.decoder_attention_heads,
            dropout=args.attention_dropout,
            self_attention=True,
            use_cpu_initialization=not getattr(
                args, "tensor_parallel_init_model_on_gpu", False
            ),
            full_megatron_init=getattr(args, "full_megatron_init", False),
            full_megatron_init_scalar=getattr(args, "full_megatron_init_scalar", 1.0),
            megatron_init_sigma=getattr(args, "megatron_init_sigma", 0.006),
            num_layers=args.decoder_layers,
            dtype=utils.get_model_init_dtype(args),
            bias=not getattr(args, "disable_bias", False),
            attn_variant=getattr(args, "attn_variant", "default"),
            xf_attn_op=getattr(args, "xf_attn_op", None),
            truncate_init=getattr(args, "truncate_init", None),
        )

    def forward_attention(
        self,
        query,
        key,
        value,
        residual,
        key_padding_mask=None,
        incremental_state=None,
        attn_mask=None,
    ):
        # This is calling into ModelParallelMultiheadAttention.forward
        attn_output, attn_bias = self.self_attn(
            query=query,
            key=key,
            value=value,
            key_padding_mask=key_padding_mask,
            incremental_state=incremental_state,
            attn_mask=attn_mask,
        )
        # Note [naman]: got rid off fused bias, dropout and residual cause
        # now we dont use dropout. And we dont use jit scripting also cause
        # it seems to use additional gpu memory for activations for dropout
        # even when its disabled.
        if attn_bias is not None:
            attn_output = attn_output + attn_bias.view(1, 1, -1)

        x = torch.nn.functional.dropout(
            attn_output,
            p=self.args.dropout,
            training=self.training,
        )
        x = x + residual
        return x
