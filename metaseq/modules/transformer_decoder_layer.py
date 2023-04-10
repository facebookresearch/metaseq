# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch import Tensor

from metaseq import utils
from metaseq.modules import (
    ActivationFn,
    ModelParallelMultiheadAttention,
    Dropout,
    FeedForward,
    LayerNorm,
)
from metaseq.modules.megatron.mpu import (
    ColumnParallelLinear,
    RowParallelLinear,
)


def _weight_init(weight):
    return nn.init.kaiming_uniform_(weight, a=math.sqrt(5))


class ModelParallelTransformerDecoderLayer(nn.Module):
    """Decoder layer block.
    Note that we have found model training to require pre-norm to remain stable.
    See "Megatron-LM: https://arxiv.org/pdf/1909.08053.pdf" for more details.
    Args:
        args (argparse.Namespace): parsed command-line arguments
    """

    def __init__(
        self,
        args,
    ):
        super().__init__()
        initialize_params_on_gpu = getattr(
            args, "tensor_parallel_init_model_on_gpu", False
        )
        device = torch.cuda.current_device() if initialize_params_on_gpu else None
        dtype = utils.get_model_init_dtype(args)

        self.args = args
        self.embed_dim = args.decoder_embed_dim
        self.dropout_module = Dropout(args.dropout, module_name=self.__class__.__name__)
        self.self_attn = self.build_self_attention(self.embed_dim, args)
        self.head_dim = int(self.embed_dim / args.decoder_attention_heads)
        affine_ln = not getattr(args, "disable_affine_ln", False)
        self.self_attn_layer_norm = LayerNorm(
            self.embed_dim, elementwise_affine=affine_ln
        )
        self.self_attn_layer_norm.to(device).to(dtype)

        self.activation_fn_name = getattr(args, "activation_fn", "relu") or "relu"

        # TODO[Susan]: Clean up these kwargs when unifying method signatures between model & non-model parallel.
        fc1_kwargs = {
            "initialize_params_on_gpu": initialize_params_on_gpu,
            "full_megatron_init": getattr(args, "full_megatron_init", False),
            "megatron_init_sigma": getattr(args, "megatron_init_sigma", 0.006),
            "dtype": utils.get_model_init_dtype(args),
            "disable_bias": getattr(args, "disable_bias", False),
            "truncate_init": getattr(args, "truncate_init", False),
        }

        self.fc1 = self.build_fc1(
            self.embed_dim,
            args.decoder_ffn_embed_dim,
            **fc1_kwargs,
        )

        self.activation_fn = ActivationFn(
            self.activation_fn_name,
            self.build_fc1,
            self.embed_dim,
            args.decoder_ffn_embed_dim,
            **fc1_kwargs,
        )

        self.fc2 = self.build_fc2(
            args.decoder_ffn_embed_dim,
            self.embed_dim,
            initialize_params_on_gpu=initialize_params_on_gpu,
            full_megatron_init=getattr(args, "full_megatron_init", False),
            full_megatron_init_scalar=getattr(args, "full_megatron_init_scalar", 1.0),
            megatron_init_sigma=getattr(args, "megatron_init_sigma", 0.006),
            num_layers=args.decoder_layers,
            dtype=utils.get_model_init_dtype(args),
            disable_bias=getattr(args, "disable_bias", False),
            truncate_init=getattr(args, "truncate_init", False),
        )

        self.final_layer_norm = LayerNorm(self.embed_dim, elementwise_affine=affine_ln)
        self.final_layer_norm.to(device).to(dtype)

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
            init_method_bias=init_method_bias,
            use_cpu_initialization=not initialize_params_on_gpu,
            dtype=dtype,
            bias=not disable_bias,
        )

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
            bias=not disable_bias,
            input_is_parallel=True,
            init_method=init_method_weights,
            use_cpu_initialization=not initialize_params_on_gpu,
            dtype=dtype,
        )
        if not full_megatron_init:
            # Copy nn.linear initialization to get same initialization as of non-model-parallel.
            # fan_in, _ = nn.init._calculate_fan_in_and_fan_out(fc2.weight)
            fan_in = input_dim
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(fc2.bias, -bound, bound)
        return fc2

    def build_self_attention(self, embed_dim, args):
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

    def forward(
        self,
        x,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        self_attn_mask: Optional[torch.Tensor] = None,
        self_attn_padding_mask: Optional[torch.Tensor] = None,
        recompute_fc1: bool = False,
    ):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        if getattr(self.args, "sequence_parallel", False):
            from metaseq.modules import SequeuceParallelTransformerBlock

            x = SequeuceParallelTransformerBlock.apply(
                x,
                self.self_attn.qkv_proj.weight,
                self.self_attn.out_proj.weight,
                self.fc1.weight,
                self.fc2.weight,
                self.self_attn.head_dim,
                recompute_fc1,
                self.activation_fn_name,
            )
            return x

        residual = x
        x = self.self_attn_layer_norm(x)
        x = self.forward_attention(
            query=x,
            key=x,
            value=x,
            residual=residual,
            key_padding_mask=self_attn_padding_mask,
            incremental_state=incremental_state,
            attn_mask=self_attn_mask,
        )
        residual = x
        x = self.final_layer_norm(x)
        x = FeedForward(
            x,
            fc1=self.fc1,
            activation_fn=self.activation_fn,
            fc2=self.fc2,
            dropout_module=self.dropout_module,
        )
        x = residual + x
        return x

    def make_generation_fast_(self, **kwargs):
        pass
