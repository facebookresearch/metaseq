# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn

from .learned_positional_embedding import LearnedPositionalEmbedding
from .sinusoidal_positional_embedding import SinusoidalPositionalEmbedding


def PositionalEmbedding(
    num_embeddings: int,
    embedding_dim: int,
    padding_idx: int,
    learned: bool = False,
    learned_sinusoidal: bool = False,
    full_megatron_init=False,
    pos_init_scalar=1.0,
    megatron_init_sigma=None,
    truncate_init=False,
):
    def _init_emb(tensor, sigma):
        if sigma <= 1e-8:  # effectively 0
            return nn.init.zeros_(tensor)
        if truncate_init:
            return nn.init.trunc_normal_(
                tensor, mean=0.0, std=sigma, a=-3 * sigma, b=3 * sigma
            )
        else:
            return nn.init.normal_(tensor, mean=0.0, std=sigma)

    if learned:
        # if padding_idx is specified then offset the embedding ids by
        # this index and adjust num_embeddings appropriately
        # TODO: The right place for this offset would be inside
        # LearnedPositionalEmbedding. Move this there for a cleaner implementation.
        if padding_idx is not None:
            num_embeddings = num_embeddings + padding_idx + 1
        m = LearnedPositionalEmbedding(num_embeddings, embedding_dim, padding_idx)
        if full_megatron_init:
            _init_emb(m.weight, megatron_init_sigma * pos_init_scalar)
        else:
            _init_emb(m.weight, embedding_dim**-0.5 * pos_init_scalar)
        if padding_idx is not None:
            nn.init.constant_(m.weight[padding_idx], 0)
    elif learned_sinusoidal:
        if padding_idx is not None:
            num_embeddings = num_embeddings + padding_idx + 1
        m = LearnedPositionalEmbedding(num_embeddings, embedding_dim, padding_idx)
        with torch.no_grad():
            m.weight.copy_(
                SinusoidalPositionalEmbedding.get_embedding(
                    num_embeddings,
                    embedding_dim,
                    padding_idx,
                )
            )
    else:
        m = SinusoidalPositionalEmbedding(
            embedding_dim,
            padding_idx,
            init_size=num_embeddings + padding_idx + 1,
        )
    return m
