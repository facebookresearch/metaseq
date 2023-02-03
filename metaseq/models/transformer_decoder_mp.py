# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging

from metaseq.modules import (
    ModelParallelTransformerDecoderLayer,
)
from metaseq.models.transformer_decoder import TransformerDecoder

try:
    from megatron import mpu
    from megatron.mpu import (
        gather_from_tensor_model_parallel_region,
    )

    has_megatron_submodule = True
except (ImportError, ModuleNotFoundError):
    has_megatron_submodule = False


logger = logging.getLogger(__name__)


class ModelParallelTransformerDecoder(TransformerDecoder):
    """
    Model Parallel Transformer decoder consisting of *args.decoder_layers* layers. Each layer
    is a :class:`ModelParallelTransformerDecoderLayer`.
    """

    def build_base_decoder_layer(self, args, **kwargs):
        return ModelParallelTransformerDecoderLayer(args)

    def output_layer(self, features, **kwargs):
        """Project features to the vocabulary size."""
        if not self.share_input_output_embed:
            raise NotImplementedError(
                "Model parallel training currently requires --share-decoder-input-output-embed"
            )

        is_sequence_parallel = getattr(self.args, "sequence_parallel", False)
        if is_sequence_parallel:
            input_parallel = features
        else:
            input_parallel = mpu.copy_to_tensor_model_parallel_region(features)

        # project back to size of vocabulary
        x = mpu.LinearWithGradAccumulationAndAsyncCommunication.apply(
            input_parallel,
            self.output_projection.weight,
            None,
            False,  # gradient_accumulation_fusion
            False,  # async_grad_allreduce
            is_sequence_parallel,  # sequence_parallel
        )
        # Gather output if model is in inference mode (i.e. evallm or generation) cause both are not yet compatible with
        # parallel vocab embeddings
        if getattr(self.args, "criterion") != "vocab_parallel_cross_entropy" or getattr(
            self, "inference", False
        ):
            x = gather_from_tensor_model_parallel_region(x).contiguous()

        return x

    # This hook used as proxy for tracking state if model is in eval or generation mode.
    def make_generation_fast_(self, **unused):
        self.inference = True

    def forward_embedding(
        self,
        *args,
    ):
        x, embed, positions = super().forward_embedding(*args)
        is_sequence_parallel = getattr(self.args, "sequence_parallel", False)
        if is_sequence_parallel:
            x = mpu.scatter_to_sequence_parallel_region(x)
        return x, embed, positions
