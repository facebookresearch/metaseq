# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from functools import partial
from typing import Dict, Optional, Tuple
import math
import torch
from torch import Tensor, nn

import logging

logger = logging.getLogger(__name__)

from metaseq.dataclass.constants import AttentionVariants
from metaseq.incremental_decoding_utils import with_incremental_state
from metaseq.modules.dropout import Dropout
from metaseq import utils

from metaseq.modules.rope import apply_rotary_emb

from megatron.mpu import (
    ColumnParallelLinear,
    RowParallelLinear,
    split_tensor_along_last_dim,
    get_cuda_rng_tracker,
    get_tensor_model_parallel_world_size,
)
from megatron.model.fused_softmax import (
    ScaledUpperTriangMaskedSoftmax,
    ScaledMaskedSoftmax,
)

try:
    import xformers.ops as xops

    has_xformers = True
except (ImportError, ModuleNotFoundError):
    has_xformers = False


@with_incremental_state
class LlamaModelParallelMultiheadAttention(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        kdim=None,
        vdim=None,
        dropout=0.0,
        bias=True,
        self_attention=False,
        use_cpu_initialization=True,
        full_megatron_init=False,
        full_megatron_init_scalar=1.0,
        megatron_init_sigma=None,
        num_layers=None,
        dtype=torch.float32,
        attn_variant=False,
        xf_attn_op=None,
        truncate_init=False,
        use_rope=False,
        combine_qkv_proj=True,
        llama_init_order=False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.qkv_same_dim = self.kdim == embed_dim and self.vdim == embed_dim
        self.model_parallel_size = get_tensor_model_parallel_world_size()
        self.num_heads_partition = num_heads // self.model_parallel_size
        assert (
            self.num_heads_partition * self.model_parallel_size == num_heads
        ), "Number of heads must be divisible by model parallel size"

        self.dropout_module = Dropout(dropout, module_name=self.__class__.__name__)
        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim**-0.5
        self.self_attention = self_attention
        self.use_rope = use_rope
        self.combine_qkv_proj = combine_qkv_proj

        def _init_method_weight(weight):
            nn.init.xavier_uniform_(weight, gain=1 / math.sqrt(2))

        def _init_method_bias(fan_in, bias):
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(bias, -bound, bound)

        if not self.combine_qkv_proj and not llama_init_order:
            self.k_proj = ColumnParallelLinear(
                self.kdim,
                embed_dim,
                bias=bias,
                gather_output=False,
                init_method=_init_method_weight,
                init_method_bias=None
                if full_megatron_init
                else partial(_init_method_bias, self.kdim),
                use_cpu_initialization=use_cpu_initialization,
                dtype=dtype,
            )
            self.v_proj = ColumnParallelLinear(
                self.vdim,
                embed_dim,
                bias=bias,
                gather_output=False,
                init_method=_init_method_weight,
                init_method_bias=None
                if full_megatron_init
                else partial(_init_method_bias, self.vdim),
                use_cpu_initialization=use_cpu_initialization,
                dtype=dtype,
            )
            self.q_proj = ColumnParallelLinear(
                embed_dim,
                embed_dim,
                bias=bias,
                gather_output=False,
                init_method=_init_method_weight,
                init_method_bias=None
                if full_megatron_init
                else partial(_init_method_bias, embed_dim),
                use_cpu_initialization=use_cpu_initialization,
                dtype=dtype,
            )
        elif not self.combine_qkv_proj and llama_init_order:
            self.q_proj = ColumnParallelLinear(
                embed_dim,
                embed_dim,
                bias=bias,
                gather_output=False,
                init_method=_init_method_weight,
                init_method_bias=None
                if full_megatron_init
                else partial(_init_method_bias, embed_dim),
                use_cpu_initialization=use_cpu_initialization,
                dtype=dtype,
            )
            self.k_proj = ColumnParallelLinear(
                self.kdim,
                embed_dim,
                bias=bias,
                gather_output=False,
                init_method=_init_method_weight,
                init_method_bias=None
                if full_megatron_init
                else partial(_init_method_bias, self.kdim),
                use_cpu_initialization=use_cpu_initialization,
                dtype=dtype,
            )
            self.v_proj = ColumnParallelLinear(
                self.vdim,
                embed_dim,
                bias=bias,
                gather_output=False,
                init_method=_init_method_weight,
                init_method_bias=None
                if full_megatron_init
                else partial(_init_method_bias, self.vdim),
                use_cpu_initialization=use_cpu_initialization,
                dtype=dtype,
            )

        def _init_method_weights_2(weight):
            nn.init.xavier_uniform_(weight, gain=1)

        init_method_weights = _init_method_weights_2
        if full_megatron_init:
            assert megatron_init_sigma is not None
            assert num_layers is not None
            init_method_weights = utils.scaled_init_method_normal(
                megatron_init_sigma * full_megatron_init_scalar,
                num_layers,
                truncate_init=truncate_init,
            )
        self.out_proj = RowParallelLinear(
            embed_dim,
            embed_dim,
            bias=bias,
            input_is_parallel=True,
            init_method=init_method_weights,
            skip_bias_add=True,
            use_cpu_initialization=use_cpu_initialization,
            dtype=dtype,
        )
        # self.xf_eff_attn = attn_variant == AttentionVariants.XFORMERS
        self.xf_eff_attn = False #TODO: remove this later once we have proper way of installing xformers
        self.xf_op = None
        if self.xf_eff_attn and not has_xformers:
            raise ImportError(
                "\n\nPlease install xformers to use memory efficient attention"
            )
        if self.xf_eff_attn and xf_attn_op is not None:
            try:
                self.xf_op = getattr(xops, xf_attn_op)
            except AttributeError:
                logging.warning(f"Invalid xformers memorry efficient op specified.")

    def forward(
        self,
        query,
        key: Optional[Tensor],
        value: Optional[Tensor],
        key_padding_mask: Optional[Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        attn_mask: Optional[Tensor] = None,
        freqs_cis: Optional[Tensor] = None,
        **unused_kwargs,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Input shape: Time x Batch x Channel

        Args:
            key_padding_mask (ByteTensor, optional): mask to exclude
                keys that are pads, of shape `(batch, src_len)`, where
                padding elements are indicated by 1s.
            attn_mask (ByteTensor, optional): typically used to
                implement causal attention, where the mask prevents the
                attention from looking forward in time (default: None).
        """
        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == self.embed_dim
        assert list(query.size()) == [tgt_len, bsz, embed_dim]
        if key is not None:
            src_len, key_bsz, _ = key.size()
            if not torch.jit.is_scripting():
                assert key_bsz == bsz
                assert value is not None
                assert src_len, bsz == value.shape[:2]

        if incremental_state is not None:
            saved_state = self._get_input_buffer(incremental_state)
        else:
            saved_state = None

        # logger.info("query:" + str(query.float().norm().item()))
        if self.self_attention:
            if self.combine_qkv_proj:
                kvq, _ = self.qkv_proj(query)
                k, v, q = split_tensor_along_last_dim(
                    kvq, 3, contiguous_split_chunks=True
                )
            else:
                q, _ = self.q_proj(query)
                k, _ = self.k_proj(query)
                v, _ = self.v_proj(query)
        else:
            assert key is not None and value is not None
            q, _ = self.q_proj(query)
            k, _ = self.k_proj(key)
            v, _ = self.v_proj(value)

        if self.use_rope:
            assert freqs_cis is not None
            q_shape = q.shape
            k_shape = k.shape
            q, k = apply_rotary_emb(
                q.view(bsz, tgt_len, self.num_heads_partition, self.head_dim),
                k.view(bsz, tgt_len, self.num_heads_partition, self.head_dim),
                seq_dim=1,
                freqs_cis=freqs_cis,
                backward=False,
            )
            q = q.view(*q_shape)
            k = k.view(*k_shape)

        # Megatron's fused kernel: "ScaledUpperTriangMaskedSoftmax" seems to crash with odd shape across seq_len dimension.
        # This is okay for training cause training we have all seq_len nice power of 2s but during evaluation and generation,
        # we have seq_lens not power of 2.
        CHANGES = not getattr(self, "inference", False)

        if self.xf_eff_attn:
            q = q.view(
                tgt_len, bsz * self.num_heads_partition, self.head_dim
            ).transpose(0, 1)
            if k is not None:
                k = k.view(-1, bsz * self.num_heads_partition, self.head_dim).transpose(
                    0, 1
                )
            if v is not None:
                v = (
                    v.contiguous()
                    .view(-1, bsz * self.num_heads_partition, self.head_dim)
                    .transpose(0, 1)
                )
            attn = xops.memory_efficient_attention(
                q,
                k,
                v,
                attn_bias=xops.LowerTriangularMask(),
                op=self.xf_op,
            )
        elif CHANGES:
            output_size = (
                q.size(1),
                self.num_heads_partition,
                q.size(0),
                k.size(0),
            )

            q = q.view(tgt_len, bsz * self.num_heads_partition, self.head_dim)
            if k is not None:
                k = k.view(-1, bsz * self.num_heads_partition, self.head_dim)
            if v is not None:
                v = (
                    v.contiguous()
                    .view(-1, bsz * self.num_heads_partition, self.head_dim)
                    .transpose(0, 1)
                )
            matmul_result = torch.empty(
                output_size[0] * output_size[1],
                output_size[2],
                output_size[3],
                dtype=q.dtype,
                device=torch.cuda.current_device(),
            )

            # Scale q,k before matmul for stability see https://tinyurl.com/sudb9s96 for math
            matmul_result = torch.baddbmm(
                matmul_result,
                math.sqrt(self.scaling) * q.transpose(0, 1),  # [b * np, sq, hn]
                math.sqrt(self.scaling)
                * k.transpose(0, 1).transpose(1, 2),  # [b * np, hn, sk]
                beta=0.0,
            )

            # Replace any non-finite values with finite equivalents, since otherwise
            # we may get NaN when adding attn_mask or computing softmax.
            if attn_mask is not None:
                matmul_result = torch.nan_to_num(matmul_result)

            # The attn_mask shape can be either seq_length x seq_length, or batch_size x seq_length x seq_length
            # depending on whether we are broadcasting the same attention mask across the batch, or
            # masking dynamically based on the data (e.g. for document attention).
            # If we have a per sequence mask, the condition len(attn_mask.size()) == 3
            # is true.
            if (attn_mask is not None) and (len(attn_mask.size()) == 3):
                # Going back to original scaled_masked_softmax to accomodate
                # non-causal attention masking (use the given input attention)
                attention_scores = matmul_result.view(*output_size)
                attn_mask = attn_mask < -0.5
                attn_mask = attn_mask.unsqueeze(1)
                attn_probs = ScaledMaskedSoftmax.apply(attention_scores, attn_mask, 1.0)
                attn_probs = attn_probs.view(
                    output_size[0] * output_size[1], output_size[2], output_size[3]
                )
            else:
                try:
                    attn_probs = ScaledUpperTriangMaskedSoftmax.apply(
                        matmul_result, 1.0
                    )
                except RuntimeError as e:
                    raise RuntimeError(
                        "Looks like you may have hit the feared INTERNAL ASSERT "
                        "ERROR. You can either ensure your sequences are padded "
                        "to a nice length (usually a power of 2), or you can make "
                        "sure you call model.make_generation_fast_() at load. See "
                        "interactive_hosted.py for an example.\n\n"
                        f"Original Exception: {e}"
                    )

            with get_cuda_rng_tracker().fork():
                attn_probs = self.dropout_module(attn_probs)

        else:
            q *= self.scaling

            q = (
                q.contiguous()
                .view(tgt_len, bsz * self.num_heads_partition, self.head_dim)
                .transpose(0, 1)
            )
            if k is not None:
                k = (
                    k.contiguous()
                    .view(-1, bsz * self.num_heads_partition, self.head_dim)
                    .transpose(0, 1)
                )
            if v is not None:
                v = (
                    v.contiguous()
                    .view(-1, bsz * self.num_heads_partition, self.head_dim)
                    .transpose(0, 1)
                )

            if saved_state is not None:
                # saved states are stored with shape (bsz, num_heads_partition, seq_len, head_dim)
                if "prev_key" in saved_state:
                    _prev_key = saved_state["prev_key"]
                    assert _prev_key is not None
                    prev_key = _prev_key.view(
                        bsz * self.num_heads_partition, -1, self.head_dim
                    )
                    assert k is not None
                    k = torch.cat([prev_key, k], dim=1)
                    src_len = k.size(1)
                if "prev_value" in saved_state:
                    _prev_value = saved_state["prev_value"]
                    assert _prev_value is not None
                    prev_value = _prev_value.view(
                        bsz * self.num_heads_partition, -1, self.head_dim
                    )
                    assert v is not None
                    v = torch.cat([prev_value, v], dim=1)
                saved_state["prev_key"] = k.view(
                    bsz, self.num_heads_partition, -1, self.head_dim
                )
                saved_state["prev_value"] = v.view(
                    bsz, self.num_heads_partition, -1, self.head_dim
                )
                saved_state["prev_key_padding_mask"] = key_padding_mask
                # In this branch incremental_state is never None
                assert incremental_state is not None
                incremental_state = self._set_input_buffer(
                    incremental_state, saved_state
                )
            assert k is not None
            assert k.size(1) == src_len

            # This is part of a workaround to get around fork/join parallelism
            # not supporting Optional types.
            if key_padding_mask is not None and key_padding_mask.dim() == 0:
                key_padding_mask = None

            if key_padding_mask is not None:
                assert key_padding_mask.size(0) == bsz
                assert key_padding_mask.size(1) == src_len

            attn_weights = torch.bmm(q, k.transpose(1, 2))

            assert list(attn_weights.size()) == [
                bsz * self.num_heads_partition,
                tgt_len,
                src_len,
            ]

            if attn_mask is not None:
                # The attn_mask shape can be either seq_length x seq_length, or batch_size x seq_length x seq_length
                # depending on whether we are broadcasting the same attention mask across the batch, or
                # masking dynamically based on the data (e.g. for document attention).
                # If we have a per sequence mask, the condition len(attn_mask.size()) == 3
                # is true.
                if len(attn_mask.size()) == 3:
                    attn_mask = attn_mask.unsqueeze(1)
                    attn_mask = attn_mask.repeat(1, self.num_heads_partition, 1, 1)
                    attn_mask = attn_mask.view(
                        bsz * self.num_heads_partition, tgt_len, src_len
                    )
                    attn_weights = attn_weights.masked_fill(
                        attn_mask < -0.5,
                        float("-inf"),
                    )
                else:
                    attn_mask = attn_mask.unsqueeze(0)
                    attn_weights += attn_mask

            if key_padding_mask is not None:
                # don't attend to padding symbols
                attn_weights = attn_weights.view(
                    bsz, self.num_heads_partition, tgt_len, src_len
                )
                attn_weights = attn_weights.masked_fill(
                    key_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool),
                    float("-inf"),
                )
                attn_weights = attn_weights.view(
                    bsz * self.num_heads_partition, tgt_len, src_len
                )

            attn_weights_float = utils.softmax(attn_weights, dim=-1)
            attn_weights = attn_weights_float.type_as(attn_weights)

            with get_cuda_rng_tracker().fork():
                attn_probs = self.dropout_module(attn_weights)

        # logger.info("attn_probs:" + str(attn_probs.float().norm().item()))
        assert v is not None

        if not self.xf_eff_attn:
            attn = torch.bmm(attn_probs, v)
            # logger.info("attn:" + str(attn.float().norm().item()))

        assert list(attn.size()) == [
            bsz * self.num_heads_partition,
            tgt_len,
            self.head_dim,
        ]
        embed_dim_partition = embed_dim // self.model_parallel_size
        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim_partition)
        attn, attn_bias = self.out_proj(attn)
        # Note that this no longer matches the signature of non-model-parallel version, which returns
        # Tuple[Tensor, Optional[Tensor]]
        return attn, attn_bias

    def _get_input_buffer(
        self, incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]]
    ) -> Dict[str, Optional[Tensor]]:
        result = self.get_incremental_state(incremental_state, "attn_state")
        if result is not None:
            return result
        else:
            empty_result: Dict[str, Optional[Tensor]] = {}
            return empty_result

    def _set_input_buffer(
        self,
        incremental_state: Dict[str, Dict[str, Optional[Tensor]]],
        buffer: Dict[str, Optional[Tensor]],
    ):
        return self.set_incremental_state(incremental_state, "attn_state", buffer)

    # This hook used as proxy for tracking state if model is in eval or generation mode.
    def make_generation_fast_(self, **unused):
        self.inference = True