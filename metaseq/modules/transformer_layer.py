# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from metaseq import distributed_utils as dist_utils, utils
from metaseq.modules import gelu, MultiheadAttention
from metaseq.modules.dropout import Dropout
from metaseq.modules.fused_bias_gelu import (
    fused_bias_gelu,
    has_fused_bias_gelu,
    load_megatron_fused_kernel,
)
from metaseq.modules.layer_norm import LayerNorm, SyncedModelParallelFusedLayerNorm
from metaseq.modules.linear import Linear


def _linear(x, weight, bias=None):
    return F.linear(x, weight, bias)


def _ffn(x, fc1, activation_fn, fc2, dropout_module, ffn_ln=None):
    x_shape = x.shape
    x = x.reshape(-1, x.size(-1))
    # apex fused bias gelu is not yet supported with megatron model parallel
    # TODO [namangoyal]: Find better way to do this
    model_parallel = not isinstance(fc1, nn.Linear) and not isinstance(fc1, Linear)
    if model_parallel and activation_fn == gelu and has_fused_bias_gelu:
        # here, we do the bias computation outside fc1 and fc2 to take advantage of fused_bias_gelu
        assert fc1.skip_bias_add
        x, bias_fc1 = fc1(x)
        x = fused_bias_gelu(x, bias_fc1)
        if ffn_ln is not None:
            x = ffn_ln(x)
        x, bias_fc2 = fc2(x)
        x = x + bias_fc2
    elif model_parallel:
        # here, we do the bias computation inside fc1 and fc2 AND gather_output
        x, _ = fc1(x)
        x = activation_fn(x)
        if ffn_ln is not None:
            x = ffn_ln(x)
        x, _ = fc2(x)
    elif has_fused_bias_gelu and activation_fn == gelu:
        x = _linear(x, fc1.weight)
        x = fused_bias_gelu(x, fc1.bias)
        if ffn_ln is not None:
            x = ffn_ln(x)
        x = _linear(x, fc2.weight, fc2.bias)
    else:
        x = fc1(x)
        x = activation_fn(x)
        if ffn_ln is not None:
            x = ffn_ln(x)
        x = fc2(x)
    x = x.view(x_shape)
    x = dropout_module(x)
    return x


class FeedForwardNetwork(nn.Module):
    """
    Feed Forward Network layer in the Transformer model
    """

    def __init__(self, args, embed_dim, ffn_dim, dropout_module=None):
        super().__init__()
        self.embed_dim = embed_dim
        self.activation_fn = utils.get_activation_fn(
            activation=str(args.activation_fn)
            if getattr(args, "activation_fn", None) is not None
            else "relu"
        )
        self.fc1 = Linear(self.embed_dim, ffn_dim)
        self.fc2 = Linear(ffn_dim, self.embed_dim)
        self.dropout_module = (
            Dropout(args.dropout, module_name=self.__class__.__name__)
            if not dropout_module
            else dropout_module
        )

    def forward(self, x):
        return _ffn(
            x,
            fc1=self.fc1,
            activation_fn=self.activation_fn,
            fc2=self.fc2,
            dropout_module=self.dropout_module,
        )


class TransformerEncoderLayer(nn.Module):
    """Encoder layer block.

    In the original paper each operation (multi-head attention or FFN) is
    postprocessed with: `dropout -> add residual -> layernorm`. In the
    tensor2tensor code they suggest that learning is more robust when
    preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`. We default to the approach in the paper, but the
    tensor2tensor approach can be enabled by setting
    *args.encoder_normalize_before* to ``True``.

    Args:
        args (argparse.Namespace): parsed command-line arguments
    """

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.embed_dim = args.encoder_embed_dim
        self.self_attn = self.build_self_attention(self.embed_dim, args)
        self.self_attn_layer_norm = LayerNorm(self.embed_dim)
        self.dropout_module = Dropout(args.dropout, module_name=self.__class__.__name__)
        self.normalize_before = args.encoder_normalize_before
        ffn_dim = args.encoder_ffn_embed_dim
        self.attn_ln = (
            LayerNorm(self.embed_dim) if getattr(args, "scale_attn", False) else None
        )

        self.activation_fn = utils.get_activation_fn(
            activation=getattr(args, "activation_fn", "relu") or "relu"
        )
        self.fc1 = Linear(self.embed_dim, ffn_dim)
        self.fc2 = Linear(ffn_dim, self.embed_dim)
        self.final_layer_norm = LayerNorm(self.embed_dim)

    def build_self_attention(self, embed_dim, args):
        return MultiheadAttention(
            embed_dim,
            args.encoder_attention_heads,
            dropout=args.attention_dropout,
            self_attention=True,
        )

    def residual_connection(self, x, residual):
        return residual + x

    def upgrade_state_dict_named(self, state_dict, name):
        """
        Rename layer norm states from `...layer_norms.0.weight` to
        `...self_attn_layer_norm.weight` and `...layer_norms.1.weight` to
        `...final_layer_norm.weight`
        """
        layer_norm_map = {"0": "self_attn_layer_norm", "1": "final_layer_norm"}
        for old, new in layer_norm_map.items():
            for m in ("weight", "bias"):
                k = "{}.layer_norms.{}.{}".format(name, old, m)
                if k in state_dict:
                    state_dict["{}.{}.{}".format(name, new, m)] = state_dict[k]
                    del state_dict[k]

    def forward(
        self,
        x,
        encoder_padding_mask: Optional[Tensor],
        attn_mask: Optional[Tensor] = None,
    ):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, seq_len)` where padding elements are indicated by ``1``.
            attn_mask (ByteTensor): binary tensor of shape `(tgt_len, src_len)`,
                where `tgt_len` is the length of output and `src_len` is the
                length of input, though here both are equal to `seq_len`.
                `attn_mask[tgt_i, src_j] = 1` means that when calculating the
                embedding for `tgt_i`, we exclude (mask out) `src_j`. This is
                useful for strided self-attention.

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        # anything in original attn_mask = 1, becomes -1e8
        # anything in original attn_mask = 0, becomes 0
        # Note that we cannot use -inf here, because at some edge cases,
        # the attention weight (before softmax) for some padded element in query
        # will become -inf, which results in NaN in model parameters
        if attn_mask is not None:
            attn_mask = attn_mask.masked_fill(attn_mask.to(torch.bool), -1e8)

        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        x, _ = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=encoder_padding_mask,
            need_weights=False,
            attn_mask=attn_mask,
        )

        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)
        x = _ffn(
            x,
            self.fc1,
            self.activation_fn,
            self.fc2,
            self.dropout_module,
            ffn_ln=self.ffn_layernorm,
        )
        l_aux = None
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.final_layer_norm(x)
        return x, l_aux


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
        self.embed_dim = args.decoder_embed_dim
        self.dropout_module = Dropout(args.dropout, module_name=self.__class__.__name__)
        self.cross_self_attention = getattr(args, "cross_self_attention", False)
        self.attn_ln = (
            LayerNorm(self.embed_dim) if getattr(args, "scale_attn", False) else None
        )

        self.self_attn = self.build_self_attention(
            self.embed_dim,
            args,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
        )
        self.normalize_before = args.decoder_normalize_before

        # use layerNorm rather than FusedLayerNorm for exporting.
        # char_inputs can be used to determint this.
        # TODO  remove this once we update apex with the fix
        export = getattr(args, "char_inputs", False)
        initialize_params_on_gpu = getattr(
            args, "tensor_parallel_init_model_on_gpu", False
        )
        if initialize_params_on_gpu and self.attn_ln is not None:
            self.attn_ln = self.attn_ln.cuda().half()
        self.nh = args.decoder_attention_heads
        self.head_dim = int(self.embed_dim / self.nh)
        scale_heads = getattr(args, "scale_heads", False)
        self.c_attn = None
        if scale_heads:
            if initialize_params_on_gpu:
                self.c_attn = nn.Parameter(
                    torch.ones((self.nh,), dtype=torch.float16).cuda(),
                    requires_grad=True,
                )
            else:
                self.c_attn = nn.Parameter(torch.ones((self.nh,)), requires_grad=True)

        self.self_attn_layer_norm = LayerNorm(self.embed_dim, export=export)

        if initialize_params_on_gpu:
            assert getattr(args, "memory_efficient_fp16", False)
            self.self_attn_layer_norm = self.self_attn_layer_norm.cuda().half()

        if no_encoder_attn:
            self.encoder_attn = None
            self.encoder_attn_layer_norm = None
        else:
            self.encoder_attn = self.build_encoder_attention(self.embed_dim, args)
            self.encoder_attn_layer_norm = LayerNorm(self.embed_dim, export=export)
            if initialize_params_on_gpu:
                self.encoder_attn_layer_norm = (
                    self.encoder_attn_layer_norm.cuda().half()
                )

        ffn_dim = args.decoder_ffn_embed_dim

        self.activation_fn = utils.get_activation_fn(
            activation=str(args.activation_fn)
            if getattr(args, "activation_fn", None) is not None
            else "relu"
        )
        # separate ffn_ln args.model_parallel_size
        mp_rank = (
            dist_utils.get_model_parallel_rank()
            if torch.distributed.is_initialized()
            else None
        )
        self.ffn_layernorm = None
        if getattr(args, "scale_fc", False):
            if args.model_parallel_size > 1:
                if not getattr(args, "sync_ln_variance", False):
                    self.ffn_layernorm = LayerNorm(ffn_dim // args.model_parallel_size)
                else:
                    self.ffn_layernorm = SyncedModelParallelFusedLayerNorm(
                        ffn_dim,
                        args.model_parallel_size,
                        mp_rank=mp_rank,
                        initialize_params_on_gpu=initialize_params_on_gpu,
                    )
            else:
                self.ffn_layernorm = LayerNorm(ffn_dim)
                if initialize_params_on_gpu:
                    self.ffn_layernorm = self.ffn_layernorm.cuda().half()
        self.skip_bias_add = (self.activation_fn == gelu) and has_fused_bias_gelu
        self.fc1 = self.build_fc1(
            self.embed_dim,
            ffn_dim,
            initialize_params_on_gpu=initialize_params_on_gpu,
            full_megatron_init=getattr(args, "full_megatron_init", False),
            megatron_init_sigma=getattr(args, "megatron_init_sigma", 0.006),
        )

        self.fc2 = self.build_fc2(
            ffn_dim,
            self.embed_dim,
            initialize_params_on_gpu=initialize_params_on_gpu,
            full_megatron_init=getattr(args, "full_megatron_init", False),
            megatron_init_sigma=getattr(args, "megatron_init_sigma", 0.006),
            num_layers=args.decoder_layers,
        )

        self.final_layer_norm = LayerNorm(self.embed_dim, export=export)
        if initialize_params_on_gpu:
            self.final_layer_norm = self.final_layer_norm.cuda().half()
        self.need_attn = True

        self.onnx_trace = False

        self.args = args

    def build_fc1(
        self, input_dim, output_dim, initialize_params_on_gpu=False, **unused_args
    ):
        return Linear(
            input_dim, output_dim, initialize_params_on_gpu=initialize_params_on_gpu
        )

    def build_fc2(
        self, input_dim, output_dim, initialize_params_on_gpu=False, **unused_args
    ):
        return Linear(
            input_dim, output_dim, initialize_params_on_gpu=initialize_params_on_gpu
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
        if self.c_attn is not None:
            tgt_len, bsz = x.size(0), x.size(1)
            x = x.view(tgt_len, bsz, self.nh, self.head_dim)
            x = torch.einsum("tbhd,h->tbhd", x, self.c_attn)
            x = x.reshape(tgt_len, bsz, self.embed_dim)
        x = self.dropout_module(x)
        if self.attn_ln is not None:
            x = self.attn_ln(x)
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
        x = _ffn(
            x,
            fc1=self.fc1,
            activation_fn=self.activation_fn,
            ffn_ln=self.ffn_layernorm,
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
