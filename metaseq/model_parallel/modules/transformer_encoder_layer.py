# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

try:
    from megatron.mpu import (
        ColumnParallelLinear,
        RowParallelLinear,
    )

    has_megatron_submodule = True
except (ImportError, ModuleNotFoundError):
    has_megatron_submodule = False

from metaseq.model_parallel.modules import ModelParallelMultiheadAttention
from metaseq.modules import TransformerEncoderLayer


class ModelParallelTransformerEncoderLayer(TransformerEncoderLayer):
    """Encoder layer block over multiple gpus.

    See "Megatron-LM: https://arxiv.org/pdf/1909.08053.pdf" for more details.
    """

    def build_fc1(self, input_dim, output_dim):
        if not has_megatron_submodule:
            raise ImportError(
                "\n\nPlease install megatron using the setup instructions!"
            )
        return ColumnParallelLinear(
            input_dim, output_dim, gather_output=False, skip_bias_add=True
        )

    def build_fc2(self, input_dim, output_dim):
        if not has_megatron_submodule:
            raise ImportError(
                "\n\nPlease install megatron using the setup instructions!"
            )
        return RowParallelLinear(
            input_dim, output_dim, input_is_parallel=True, skip_bias_add=True
        )

    def build_self_attention(self, embed_dim, args, **unused_kwargs):
        return ModelParallelMultiheadAttention(
            embed_dim,
            args.encoder_attention_heads,
            dropout=args.attention_dropout,
            self_attention=True,
        )
