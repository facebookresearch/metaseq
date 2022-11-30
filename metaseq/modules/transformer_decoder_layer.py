# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, Optional

import torch
import torch.nn as nn
from torch import Tensor

from metaseq import utils
from metaseq.modules import (
    ActivationFn,
    MultiheadAttention,
    Dropout,
    FeedForwardNetwork,
    LayerNorm,
    Linear,
)
from metaseq.modules.fused_bias_gelu import (
    has_fused_bias_gelu,
    load_megatron_fused_kernel,
)


class TransformerDecoderLayer(nn.Module):
    """Pre-norm Decoder layer block.

    Note that we have found model training to require pre-norm to remain stable.

    Args:
        args (argparse.Namespace): parsed command-line arguments
    """

    def __init__(
        self,
        args,
        add_bias_kv=False,
        add_zero_attn=False,
    ):
        super().__init__()
        load_megatron_fused_kernel()
        self.args = args
        self.embed_dim = args.decoder_embed_dim
        self.dropout_module = Dropout(args.dropout, module_name=self.__class__.__name__)
        self.self_attn = self.build_self_attention(
            self.embed_dim,
            args,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
        )
        initialize_params_on_gpu = getattr(
            args, "tensor_parallel_init_model_on_gpu", False
        )
        device = torch.cuda.current_device() if initialize_params_on_gpu else None
        dtype = utils.get_model_init_dtype(args)

        self.nh = args.decoder_attention_heads
        self.head_dim = int(self.embed_dim / self.nh)
        affine_ln = not getattr(args, "disable_affine_ln", False)

        self.self_attn_layer_norm = LayerNorm(
            self.embed_dim, elementwise_affine=affine_ln
        )
        self.self_attn_layer_norm.to(device).to(dtype)

        ffn_dim = args.decoder_ffn_embed_dim

        self.activation_fn_name = getattr(args, "activation_fn", "relu") or "relu"
        self.skip_bias_add = (self.activation_fn_name == "gelu") and has_fused_bias_gelu

        # TODO[Susan]: Clean up these kwargs when unifying method signatures between model & non-model parallel.
        fc1_kwargs = {
            "initialize_params_on_gpu": initialize_params_on_gpu,
            "full_megatron_init": getattr(args, "full_megatron_init", False),
            "megatron_init_sigma": getattr(args, "megatron_init_sigma", 0.006),
            "dtype": utils.get_model_init_dtype(args),
            "disable_bias": getattr(args, "disable_bias", False),
            "truncate_init": getattr(args, "truncate_init", False),
        }

        # Note: ModelParallelTransformerDecoderLayer overrides build_fc1.
        self.fc1 = self.build_fc1(
            self.embed_dim,
            ffn_dim,
            **fc1_kwargs,
        )

        self.activation_fn = ActivationFn(
            self.activation_fn_name,
            self.build_fc1,
            self.embed_dim,
            ffn_dim,
            **fc1_kwargs,
        )

        # Note: ModelParallelTransformerDecoderLayer overrides build_fc2.
        self.fc2 = self.build_fc2(
            ffn_dim,
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
        self.args = args

    # Refer to model_parallel's transformer layer for why fc1 and fc2 are separate methods.
    def build_fc1(
        self,
        input_dim,
        output_dim,
        initialize_params_on_gpu=False,
        disable_bias=False,
        **unused_args
    ):
        return Linear(
            input_dim,
            output_dim,
            initialize_params_on_gpu=initialize_params_on_gpu,
            bias=not disable_bias,
            dtype=utils.get_model_init_dtype(self.args),
        )

    def build_fc2(
        self,
        input_dim,
        output_dim,
        initialize_params_on_gpu=False,
        disable_bias=False,
        **unused_args
    ):
        return Linear(
            input_dim,
            output_dim,
            initialize_params_on_gpu=initialize_params_on_gpu,
            dtype=utils.get_model_init_dtype(self.args),
        )

    def build_self_attention(
        self,
        embed_dim,
        args,
        add_bias_kv=False,
        add_zero_attn=False,
    ):
        return MultiheadAttention(
            embed_dim,
            args.decoder_attention_heads,
            dropout=args.attention_dropout,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
            self_attention=True,
            initialize_params_on_gpu=getattr(
                args, "tensor_parallel_init_model_on_gpu", False
            ),
            bias=not getattr(
                args,
                "disable_bias",
                False,
            ),
            dtype=utils.get_model_init_dtype(args),
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
        x, _ = self.self_attn(
            query=query,
            key=key,
            value=value,
            key_padding_mask=key_padding_mask,
            incremental_state=incremental_state,
            attn_mask=attn_mask,
        )
        x = self.dropout_module(x)
        x = residual + x
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
            from metaseq.model_parallel.modules import SequeuceParallelTransformerBlock

            x = SequeuceParallelTransformerBlock.apply(
                x,
                self.self_attn.qkv_proj.weight,
                self.self_attn.out_proj.weight,
                self.fc1.weight,
                self.fc2.weight,
                self.self_attn.head_dim,
                recompute_fc1,
                self.activation_fn_name,
                attn_variant=getattr(args, "attn_variant", "default")
                xf_attn_op=getattr(args, "xf_attn_op", None)
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
        x = FeedForwardNetwork(
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
