# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from metaseq.model_parallel.modules import ModelParallelTransformerEncoderLayer
from metaseq.models.transformer_encoder import TransformerEncoder


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
