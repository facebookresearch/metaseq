# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from metaseq.modules.activation_functions import gelu, gelu_back, relu, relu_back

import importlib
import logging
import math
import torch
from types import SimpleNamespace
from metaseq.dataclass.constants import AttentionVariants

# Not importing here cause cpu tests don't like it
global fused_layer_norm_cuda
fused_layer_norm_cuda = None

try:
    import xformers.ops as xops

    has_xformers = True
except (ImportError, ModuleNotFoundError):
    has_xformers = False

try:
    from megatron.mpu.mappings import (
        _reduce_scatter_along_first_dim,
        _gather_along_first_dim,
    )
    from megatron.mpu.utils import split_tensor_along_last_dim
    from megatron.model.fused_softmax import scaled_upper_triang_masked_softmax_cuda

    has_megatron_submodule = True
except (ImportError, ModuleNotFoundError):
    has_megatron_submodule = False


class _FakeContext(SimpleNamespace):
    """
    Used to provide a temporary buffer for FlashAttention's saved buffers
    """

    saved_tensors = None

    def save_for_backward(self, *args):
        self.saved_tensors = args


class SequeuceParallelTransformerBlock(torch.autograd.Function):
    """
    This is custom FFN autograd function hardcoded for:
    bias: false,
    layernorm affine: false, ln eps: 1e-5
    sequence_parallel: true,
    activation: gelu,
    gelu, layernorm: always recomputed i.e. no activation memory for these
    """

    @staticmethod
    def forward_mha(q, k, v, bsz, seq_len, head_dim, embed_dim_per_partition, dtype):
        scaling = head_dim**-0.5
        matmul_result = torch.empty(
            bsz * (embed_dim_per_partition // head_dim),
            seq_len,
            seq_len,
            dtype=dtype,
            device=torch.cuda.current_device(),
        )
        # Scale q,k before matmul for stability see https://tinyurl.com/sudb9s96 for math
        matmul_result = torch.baddbmm(
            matmul_result,
            math.sqrt(scaling) * q.transpose(0, 1),
            math.sqrt(scaling) * k.transpose(0, 1).transpose(1, 2),
            beta=0.0,
        )
        # attn_probs = matmul_result
        scale_t = torch.tensor([1.0])
        attn_probs = scaled_upper_triang_masked_softmax_cuda.forward(
            matmul_result, scale_t[0]
        )
        attn = torch.bmm(attn_probs, v)
        attn = attn.transpose(0, 1).contiguous().view(seq_len, bsz, -1)
        return attn, attn_probs

    @staticmethod
    def backward_mha(grad_mha_output, q, k, v, attn_probs, seq_len, bsz, head_dim):
        scaling = head_dim**-0.5
        grad_mha_output = grad_mha_output.view(seq_len, -1, head_dim).transpose(0, 1)
        grad_v = (
            torch.bmm(attn_probs.transpose(1, 2), grad_mha_output)
            .transpose(0, 1)
            .contiguous()
            .view(seq_len, bsz, -1)
        )
        grad_attn_probs_out = torch.bmm(grad_mha_output, v.transpose(1, 2))

        grad_attn_probs_in = scaled_upper_triang_masked_softmax_cuda.backward(
            grad_attn_probs_out, attn_probs, 1.0
        )
        grad_q = torch.bmm(
            math.sqrt(scaling) * grad_attn_probs_in,
            math.sqrt(scaling) * k.transpose(0, 1),
        )
        grad_q = grad_q.transpose(0, 1).contiguous().view(seq_len, bsz, -1)
        grad_k = torch.bmm(
            math.sqrt(scaling) * grad_attn_probs_in.transpose(1, 2),
            math.sqrt(scaling) * q.transpose(0, 1),
        )
        grad_k = grad_k.transpose(0, 1).contiguous().view(seq_len, bsz, -1)
        grad_kvq_proj_output = torch.cat([grad_k, grad_v, grad_q], dim=-1)
        return grad_kvq_proj_output

    @staticmethod
    def forward(
        ctx,
        input,
        kvq_proj_weight,
        out_proj_weight,
        fc1_weight,
        fc2_weight,
        head_dim,
        recompute_fc1,
        activation_fn_name,  # "relu" or "gelu" for now
        attn_variant,
        xf_attn_op,
    ):
        assert (
            activation_fn_name == "relu" or activation_fn_name == "gelu"
        ), "Only relu/gelu is supported!"

        xf_eff_attn = attn_variant == AttentionVariants.XFORMERS
        if xf_eff_attn and not has_xformers:
            raise ImportError(
                "\n\nPlease install xformers to use memory efficient attention"
            )

        xf_op = xops.MemoryEfficientAttentionCutlassFwdFlashBwOp
        if xf_eff_attn and xf_attn_op is not None:
            try:
                xf_op = getattr(xops, xf_attn_op)
            except AttributeError:
                logging.warning(f"Invalid xformers memorry efficient op specified.")

        # import from apex
        global fused_layer_norm_cuda
        if fused_layer_norm_cuda is None:
            fused_layer_norm_cuda = importlib.import_module("fused_layer_norm_cuda")

        ctx.recompute_fc1 = recompute_fc1

        input = input.contiguous()

        # Take out residual connection for self attention
        residual = input
        dtype = input.dtype

        # Apply layer norm on (seq_len // #tp_size, bsz, embed_dim) tensor
        ctx.layer_norm_normalized_shape = torch.Size((input.size(-1),))
        ctx.eps = 1e-5

        # # Self attention layer norm
        mha_layer_norm_output, _, _ = fused_layer_norm_cuda.forward(
            input, ctx.layer_norm_normalized_shape, ctx.eps
        )

        # all gather output across first dim, i.e. seq_len dim for kvq_proj
        mha_layer_norm_output = _gather_along_first_dim(
            mha_layer_norm_output, cached_buffer_name="mpu"
        )

        # apply kvq, output is (seq_len, bsz, 3 * embed_dim // #tp_size)
        # The order of (k,v, q) here doesn't matter as much as long its consistent since initialization of all three is same.
        # just matching the order of metaseq MHA.
        kvq_out = torch.matmul(mha_layer_norm_output, kvq_proj_weight.t())

        k, v, q = split_tensor_along_last_dim(kvq_out, 3, contiguous_split_chunks=True)
        seq_len, bsz, embed_dim_per_partition = q.size()

        if xf_eff_attn:
            q = q.view(seq_len, bsz, -1, head_dim).transpose(0, 1)
            k = k.view(seq_len, bsz, -1, head_dim).transpose(0, 1)
            v = v.view(seq_len, bsz, -1, head_dim).transpose(0, 1)

            attn = xf_op.forward_no_grad(
                q, k, v, attn_bias=xops.LowerTriangularMask(), p=0.0, scale=None
            ).view(seq_len, bsz, -1)
        else:
            q = q.view(seq_len, -1, head_dim)
            k = k.view(seq_len, -1, head_dim)
            v = v.view(seq_len, -1, head_dim).transpose(0, 1)

            attn, _ = SequeuceParallelTransformerBlock.forward_mha(
                q, k, v, bsz, seq_len, head_dim, embed_dim_per_partition, dtype
            )

        out_proj_out = torch.matmul(attn, out_proj_weight.t())
        out_proj_out = _reduce_scatter_along_first_dim(out_proj_out)
        out_proj_out = out_proj_out.view_as(residual)

        out_proj_out = out_proj_out + residual

        # Take out residual connection for FFN
        residual = out_proj_out
        # No need to save mean and invvar cause we redo layernorm in backward
        ffn_layer_norm_output, _, _ = fused_layer_norm_cuda.forward(
            out_proj_out, ctx.layer_norm_normalized_shape, ctx.eps
        )

        # all gather output across first dim, i.e. seq_len dim
        ffn_layer_norm_output = _gather_along_first_dim(
            ffn_layer_norm_output, cached_buffer_name="mpu"
        )

        # apply fc1, output is (seq_len, bsz, 4 * embed_dim // #tp_size)
        fc1_out = torch.matmul(ffn_layer_norm_output, fc1_weight.t())

        # apply activation
        # TODO: split out to explicit if/else instead of defaulting to relu when not gelu
        actv_out = gelu(fc1_out) if activation_fn_name == "gelu" else relu(fc1_out)

        # apply fc2, output (seq_len, bsz, embed_dim) but needs to be
        # summed across tp for real output
        fc2_out = torch.matmul(actv_out, fc2_weight.t())

        if ctx.recompute_fc1:
            fc1_out = None
        ctx.save_for_backward(
            input,
            q,
            k,
            v,
            out_proj_out,
            kvq_proj_weight,
            out_proj_weight,
            fc1_out,
            fc1_weight,
            fc2_weight,
        )
        (
            ctx.bsz,
            ctx.seq_len,
            ctx.head_dim,
            ctx.embed_dim_per_partition,
            ctx.activation_fn_name,
            ctx.xf_eff_attn,
            ctx.xf_op,
        ) = (
            bsz,
            seq_len,
            head_dim,
            embed_dim_per_partition,
            activation_fn_name,
            xf_eff_attn,
            xf_op,
        )

        # apply scatter gather,
        # input: (seq_len, bsz, embed_dim)
        # output: (seq_len // #tp_size, bsz, embed_dim) (and embed_dim is summed across gpus)
        fc2_out_post_scatter_gather = _reduce_scatter_along_first_dim(fc2_out)
        final_out = fc2_out_post_scatter_gather + residual
        return final_out

    @staticmethod
    def backward(ctx, grad_output):
        (
            input,
            q,
            k,
            v,
            out_proj_out,
            kvq_proj_weight,
            out_proj_weight,
            fc1_out,
            fc1_weight,
            fc2_weight,
        ) = ctx.saved_tensors
        (
            bsz,
            seq_len,
            head_dim,
            embed_dim_per_partition,
            activation_fn_name,
            xf_eff_attn,
            xf_op,
        ) = (
            ctx.bsz,
            ctx.seq_len,
            ctx.head_dim,
            ctx.embed_dim_per_partition,
            ctx.activation_fn_name,
            ctx.xf_eff_attn,
            ctx.xf_op,
        )
        dtype = grad_output.dtype

        residual_grad = grad_output

        # gatther gradients async,
        # and we can overlap this with any recomptation.
        grad_output, handle = _gather_along_first_dim(grad_output, async_op=True)

        # Both of these operations are just recomputed from forward to save activation memory.
        (
            ffn_layer_norm_output,
            ffn_layer_norm_mean,
            ffn_layer_norm_invvar,
        ) = fused_layer_norm_cuda.forward(
            out_proj_out, ctx.layer_norm_normalized_shape, ctx.eps
        )
        # recompute gelu output for calculating fc2 weight gradient
        # note, remember "gelu_out = fc2_in"
        if not ctx.recompute_fc1:
            assert fc1_out is not None
            actv_out = gelu(fc1_out) if activation_fn_name == "gelu" else relu(fc1_out)

        # Now wait for reduce scatter
        handle.wait()

        ffn_layer_norm_output, handle = _gather_along_first_dim(
            ffn_layer_norm_output, async_op=True, cached_buffer_name="mpu"
        )

        grad_fc2_input = grad_output.matmul(fc2_weight)

        if ctx.recompute_fc1:
            handle.wait()
            assert fc1_out is None
            fc1_out = torch.matmul(ffn_layer_norm_output, fc1_weight.t())
            actv_out = gelu(fc1_out) if activation_fn_name == "gelu" else relu(fc1_out)

        # calculate gelu/relu backward
        grad_actv_input = (
            gelu_back(grad_fc2_input, fc1_out)
            if activation_fn_name == "gelu"
            else relu_back(grad_fc2_input, fc1_out)
        )

        # Reshape matrix and calculate gradient with respect to fc2 weight
        grad_output = SequeuceParallelTransformerBlock._collapse_first_dimensions(
            grad_output
        )
        actv_out = SequeuceParallelTransformerBlock._collapse_first_dimensions(actv_out)
        grad_fc2_weight = grad_output.t().matmul(actv_out)

        grad_fc1_input = grad_actv_input.matmul(fc1_weight)
        handle.wait()

        grad_actv_input = SequeuceParallelTransformerBlock._collapse_first_dimensions(
            grad_actv_input
        )
        ffn_layer_norm_output = (
            SequeuceParallelTransformerBlock._collapse_first_dimensions(
                ffn_layer_norm_output
            )
        )

        grad_fc1_input, handle = _reduce_scatter_along_first_dim(
            grad_fc1_input, async_op=True
        )

        grad_fc1_weight = grad_actv_input.t().matmul(ffn_layer_norm_output)

        handle.wait()

        grad_attention_output = fused_layer_norm_cuda.backward(
            grad_fc1_input.contiguous(),
            ffn_layer_norm_mean,
            ffn_layer_norm_invvar,
            out_proj_out,
            ctx.layer_norm_normalized_shape,
            ctx.eps,
        )
        grad_attention_output = grad_attention_output + residual_grad

        residual_grad = grad_attention_output

        grad_attention_output, handle = _gather_along_first_dim(
            grad_attention_output,
            async_op=True,
        )

        # recalculate attention
        if xf_eff_attn:
            fake_ctx = _FakeContext()
            attn = xf_op.forward(
                fake_ctx,
                q,
                k,
                v,
                attn_bias=xops.LowerTriangularMask(),
                p=0.0,
                scale=None,
            ).view(seq_len, bsz, -1)
        else:
            attn, attn_probs = SequeuceParallelTransformerBlock.forward_mha(
                q, k, v, bsz, seq_len, head_dim, embed_dim_per_partition, dtype
            )

        handle.wait()

        grad_out_proj_input = grad_attention_output.matmul(out_proj_weight)
        grad_attention_output = (
            SequeuceParallelTransformerBlock._collapse_first_dimensions(
                grad_attention_output
            )
        )
        attn = SequeuceParallelTransformerBlock._collapse_first_dimensions(attn)
        grad_out_proj_weight = grad_attention_output.t().matmul(attn)

        if xf_eff_attn:
            d_q, d_k, d_v, _, _ = xf_op.backward(fake_ctx, grad_out_proj_input)
            d_q = d_q.transpose(0, 1).view(seq_len, bsz, -1)
            d_k = d_k.transpose(0, 1).view(seq_len, bsz, -1)
            d_v = d_v.transpose(0, 1).view(seq_len, bsz, -1)
            grad_kvq_proj_output = torch.cat([d_k, d_v, d_q], dim=-1)
        else:
            grad_kvq_proj_output = SequeuceParallelTransformerBlock.backward_mha(
                grad_out_proj_input, q, k, v, attn_probs, seq_len, bsz, head_dim
            )

        (
            mha_layer_norm_output,
            mha_layer_norm_mean,
            mha_layer_norm_invvar,
        ) = fused_layer_norm_cuda.forward(
            input, ctx.layer_norm_normalized_shape, ctx.eps
        )
        mha_layer_norm_output, handle = _gather_along_first_dim(
            mha_layer_norm_output,
            async_op=True,
            cached_buffer_name="mpu",
        )
        grad_input = grad_kvq_proj_output.matmul(kvq_proj_weight)
        handle.wait()

        grad_input, handle = _reduce_scatter_along_first_dim(grad_input, async_op=True)
        mha_layer_norm_output = (
            SequeuceParallelTransformerBlock._collapse_first_dimensions(
                mha_layer_norm_output
            )
        )
        grad_kvq_proj_output = (
            SequeuceParallelTransformerBlock._collapse_first_dimensions(
                grad_kvq_proj_output
            )
        )
        grad_kvq_weight = grad_kvq_proj_output.t().matmul(mha_layer_norm_output)
        handle.wait()

        grad_input = fused_layer_norm_cuda.backward(
            grad_input.contiguous(),
            mha_layer_norm_mean,
            mha_layer_norm_invvar,
            input,
            ctx.layer_norm_normalized_shape,
            ctx.eps,
        )
        grad_input = grad_input + residual_grad
        return (
            grad_input,
            grad_kvq_weight,
            grad_out_proj_weight,
            grad_fc1_weight,
            grad_fc2_weight,
            None,
            None,
            None,
        )

    @staticmethod
    def _collapse_first_dimensions(tensor):
        return tensor.view(
            tensor.shape[0] * tensor.shape[1],
            tensor.shape[2],
        )
