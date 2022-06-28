# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging

import torch
from torch import Tensor
from typing import Any, Dict, List, Optional

from metaseq.model_parallel.modules import (
    ModelParallelTransformerDecoderLayer,
    ModelParallelTransformerEncoderLayer,
)
from metaseq.models.transformer import TransformerDecoder, TransformerEncoder

try:
    from megatron import mpu
    from megatron.mpu import (
        copy_to_tensor_model_parallel_region,
        gather_from_tensor_model_parallel_region,
    )

    has_megatron_submodule = True
except (ImportError, ModuleNotFoundError):
    has_megatron_submodule = False


logger = logging.getLogger(__name__)


class ModelParallelTransformerEncoder(TransformerEncoder):
    """
    Model parallel Transformer encoder consisting of *args.encoder_layers* layers. Each layer
    is a :class:`ModelParallelTransformerEncoderLayer`.
    """

    def __init__(self, args, dictionary, embed_tokens):
        super().__init__(args, dictionary, embed_tokens)

        if args.no_final_layer_norm:
            self.layer_norm = None

    def build_encoder_layer(self, args):
        return ModelParallelTransformerEncoderLayer(args)


class ModelParallelTransformerDecoder(TransformerDecoder):
    """
    Model Parallel Transformer decoder consisting of *args.decoder_layers* layers. Each layer
    is a :class:`ModelParallelTransformerDecoderLayer`.
    """

    def build_base_decoder_layer(self, args, no_encoder_attn=False, **kwargs):
        return ModelParallelTransformerDecoderLayer(args, no_encoder_attn)

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
            False,
            False,
            is_sequence_parallel,
        )
        # Gather output if model in in inference mode (i.e. evallm or generation) cause both are not yet compatible with
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
        tokens,
        token_embedding: Optional[torch.Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
    ):
        # embed tokens and positions
        positions = None
        if self.embed_positions is not None:
            positions = self.embed_positions(
                tokens, incremental_state=incremental_state
            )
        # see IncrementalDecoder for important information about
        # incremental state
        if incremental_state:
            tokens = tokens[:, -1:]
            if positions is not None:
                positions = positions[:, -1:]

        if token_embedding is None:
            token_embedding = self.embed_tokens(tokens)

        x = embed = self.embed_scale * token_embedding

        if self.project_in_dim is not None:
            x = self.project_in_dim(x)

        if positions is not None:
            x += positions

        x = x.transpose(0, 1).contiguous()

        is_sequence_parallel = getattr(self.args, "sequence_parallel", False)

        if is_sequence_parallel:
            x = mpu.scatter_to_sequence_parallel_region(x)

        if self.dropout_module is not None:
            if is_sequence_parallel:
                with mpu.get_cuda_rng_tracker().fork():
                    x = self.dropout_module(x)
            else:
                x = self.dropout_module(x)
        return x, embed, positions
