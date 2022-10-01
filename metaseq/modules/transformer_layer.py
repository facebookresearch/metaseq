# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch import Tensor

from metaseq import utils
from metaseq.modules import gelu, MultiheadAttention
from metaseq.modules.dropout import Dropout
from metaseq.modules.feedforward_network import FeedForwardNetwork
from metaseq.modules.fused_bias_gelu import (
    has_fused_bias_gelu,
    load_megatron_fused_kernel,
)
from metaseq.modules.layer_norm import LayerNorm
from metaseq.modules.linear import Linear


class TransformerDecoderLayer(nn.Module):
    """Decoder layer block.

    In the original paper each operation (multi-head attention, encoder
    attention or FFN) is postprocessed with: `dropout -> add residual ->
    layernorm`. In the tensor2tensor code they suggest that learning is more
    robust when preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`. We default to the approach in the paper, but the
    tensor2tensor approach can be enabled by setting
    *args.decoder_normalize_before* to ``True``.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        no_encoder_attn (bool, optional): whether to attend to encoder outputs
            (default: False).
    """

    def __init__(
        self,
        args,
        no_encoder_attn=False,
        add_bias_kv=False,
        add_zero_attn=False,
    ):
        super().__init__()
        load_megatron_fused_kernel()
        self.args = args
        self.embed_dim = args.decoder_embed_dim
        self.dropout_module = Dropout(args.dropout, module_name=self.__class__.__name__)
        self.cross_self_attention = getattr(args, "cross_self_attention", False)
        self.self_attn = self.build_self_attention(
            self.embed_dim,
            args,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
        )
        self.normalize_before = args.decoder_normalize_before

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

        if no_encoder_attn:
            self.encoder_attn = None
            self.encoder_attn_layer_norm = None
        else:
            self.encoder_attn = self.build_encoder_attention(self.embed_dim, args)
            self.encoder_attn_layer_norm = LayerNorm(self.embed_dim)
            self.encoder_attn_layer_norm = self.encoder_attn_layer_norm.to(device).to(
                dtype
            )

        ffn_dim = args.decoder_ffn_embed_dim

        self.activation_fn = utils.get_activation_fn(
            activation=str(args.activation_fn)
            if getattr(args, "activation_fn", None) is not None
            else "relu"
        )
        self.skip_bias_add = (self.activation_fn == gelu) and has_fused_bias_gelu
        self.fc1 = self.build_fc1(
            self.embed_dim,
            ffn_dim,
            initialize_params_on_gpu=initialize_params_on_gpu,
            full_megatron_init=getattr(args, "full_megatron_init", False),
            megatron_init_sigma=getattr(args, "megatron_init_sigma", 0.006),
            dtype=utils.get_model_init_dtype(args),
            disable_bias=getattr(args, "disable_bias", False),
        )

        self.fc2 = self.build_fc2(
            ffn_dim,
            self.embed_dim,
            initialize_params_on_gpu=initialize_params_on_gpu,
            full_megatron_init=getattr(args, "full_megatron_init", False),
            megatron_init_sigma=getattr(args, "megatron_init_sigma", 0.006),
            num_layers=args.decoder_layers,
            dtype=utils.get_model_init_dtype(args),
            disable_bias=getattr(args, "disable_bias", False),
        )

        self.final_layer_norm = LayerNorm(self.embed_dim, elementwise_affine=affine_ln)
        self.final_layer_norm.to(device).to(dtype)

        self.need_attn = True
        self.onnx_trace = False
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
            self_attention=not getattr(args, "cross_self_attention", False),
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

    def build_encoder_attention(self, embed_dim, args):
        return MultiheadAttention(
            embed_dim,
            args.decoder_attention_heads,
            kdim=getattr(args, "encoder_embed_dim", None),
            vdim=getattr(args, "encoder_embed_dim", None),
            dropout=args.attention_dropout,
            encoder_decoder_attention=True,
            initialize_params_on_gpu=getattr(
                args, "tensor_parallel_init_model_on_gpu", False
            ),
            dtype=utils.get_model_init_dtype(args),
        )

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def residual_connection(self, x, residual):
        return residual + x

    def forward_attention(
        self,
        query,
        key,
        value,
        residual,
        key_padding_mask=None,
        incremental_state=None,
        need_weights=False,
        attn_mask=None,
    ):
        x, attn = self.self_attn(
            query=query,
            key=key,
            value=value,
            key_padding_mask=key_padding_mask,
            incremental_state=incremental_state,
            need_weights=need_weights,
            attn_mask=attn_mask,
        )
        x = self.dropout_module(x)
        return self.residual_connection(x, residual), attn

    def forward(
        self,
        x,
        encoder_out: Optional[torch.Tensor] = None,
        encoder_padding_mask: Optional[torch.Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        prev_self_attn_state: Optional[List[torch.Tensor]] = None,
        prev_attn_state: Optional[List[torch.Tensor]] = None,
        self_attn_mask: Optional[torch.Tensor] = None,
        self_attn_padding_mask: Optional[torch.Tensor] = None,
        need_attn: bool = False,
        need_head_weights: bool = False,
    ):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor, optional): binary
                ByteTensor of shape `(batch, src_len)` where padding
                elements are indicated by ``1``.
            need_attn (bool, optional): return attention weights
            need_head_weights (bool, optional): return attention weights
                for each head (default: return average over heads).

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        if need_head_weights:
            need_attn = True

        residual = x

        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        if prev_self_attn_state is not None:
            prev_key, prev_value = prev_self_attn_state[:2]
            saved_state: Dict[str, Optional[Tensor]] = {
                "prev_key": prev_key,
                "prev_value": prev_value,
            }
            if len(prev_self_attn_state) >= 3:
                saved_state["prev_key_padding_mask"] = prev_self_attn_state[2]
            assert incremental_state is not None
            self.self_attn._set_input_buffer(incremental_state, saved_state)
        _self_attn_input_buffer = self.self_attn._get_input_buffer(incremental_state)
        if self.cross_self_attention and not (
            incremental_state is not None
            and _self_attn_input_buffer is not None
            and "prev_key" in _self_attn_input_buffer
        ):
            if self_attn_mask is not None:
                assert encoder_out is not None
                self_attn_mask = torch.cat(
                    (x.new_zeros(x.size(0), encoder_out.size(0)), self_attn_mask), dim=1
                )
            if self_attn_padding_mask is not None:
                if encoder_padding_mask is None:
                    assert encoder_out is not None
                    encoder_padding_mask = self_attn_padding_mask.new_zeros(
                        encoder_out.size(1), encoder_out.size(0)
                    )
                self_attn_padding_mask = torch.cat(
                    (encoder_padding_mask, self_attn_padding_mask), dim=1
                )
            assert encoder_out is not None
            y = torch.cat((encoder_out, x), dim=0)
        else:
            y = x

        x, attn = self.forward_attention(
            query=x,
            key=y,
            value=y,
            residual=residual,
            key_padding_mask=self_attn_padding_mask,
            incremental_state=incremental_state,
            need_weights=False,
            attn_mask=self_attn_mask,
        )
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        if self.encoder_attn is not None and encoder_out is not None:
            residual = x
            if self.normalize_before:
                x = self.encoder_attn_layer_norm(x)
            if prev_attn_state is not None:
                prev_key, prev_value = prev_attn_state[:2]
                saved_state: Dict[str, Optional[Tensor]] = {
                    "prev_key": prev_key,
                    "prev_value": prev_value,
                }
                if len(prev_attn_state) >= 3:
                    saved_state["prev_key_padding_mask"] = prev_attn_state[2]
                assert incremental_state is not None
                self.encoder_attn._set_input_buffer(incremental_state, saved_state)

            x, attn = self.encoder_attn(
                query=x,
                key=encoder_out,
                value=encoder_out,
                key_padding_mask=encoder_padding_mask,
                incremental_state=incremental_state,
                static_kv=True,
                need_weights=need_attn or (not self.training and self.need_attn),
                need_head_weights=need_head_weights,
            )
            x = self.dropout_module(x)
            x = self.residual_connection(x, residual)
            if not self.normalize_before:
                x = self.encoder_attn_layer_norm(x)
        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)
        x = FeedForwardNetwork(
            x,
            fc1=self.fc1,
            activation_fn=self.activation_fn,
            fc2=self.fc2,
            dropout_module=self.dropout_module,
        )
        l_aux = None
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.final_layer_norm(x)
        if self.onnx_trace and incremental_state is not None:
            saved_state = self.self_attn._get_input_buffer(incremental_state)
            assert saved_state is not None
            if self_attn_padding_mask is not None:
                self_attn_state = [
                    saved_state["prev_key"],
                    saved_state["prev_value"],
                    saved_state["prev_key_padding_mask"],
                ]
            else:
                self_attn_state = [saved_state["prev_key"], saved_state["prev_value"]]
            return x, attn, self_attn_state
        return x, attn, None, l_aux

    def make_generation_fast_(self, need_attn: bool = False, **kwargs):
        self.need_attn = need_attn
