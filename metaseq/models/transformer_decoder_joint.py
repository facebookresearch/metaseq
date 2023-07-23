# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import logging
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch import Tensor
from metaseq.dataclass.constants import UNSPECIFIED_DOC_SEP

from metaseq import utils
from metaseq.distributed import utils as distributed_utils, fsdp_wrap
from metaseq.models import BaseDecoder
from metaseq.models.transformer_decoder import ModelParallelTransformerDecoder
from metaseq.modules import (
    Dropout,
    LayerNorm,
    PositionalEmbedding,
    ModelParallelTransformerDecoderLayer,
    Linear,
)
from metaseq.modules.checkpoint_activations import checkpoint_wrapper

try:
    from megatron import mpu
    from megatron.mpu import (
        gather_from_tensor_model_parallel_region,
    )

    has_megatron_submodule = True
except (ImportError, ModuleNotFoundError):
    has_megatron_submodule = False

logger = logging.getLogger(__name__)


DEFAULT_MAX_SOURCE_POSITIONS = 1024
DEFAULT_MAX_TARGET_POSITIONS = 1024
DEFAULT_MIN_PARAMS_TO_WRAP = int(1e8)


class TransformerDecoderMultiLayerBlockModule(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    # TODO[susanz]: Return signature seems off. Cleanup?
    #  fsdp_checkpoint_wrap_layer_frequency always 1 so this path is not called.
    def forward(self, x, **kwargs):
        inner_states = []
        for layer in self.layers:
            x = layer(x, **kwargs)
            inner_states.append(x)
        return x, inner_states


def log_weight_stats(tensor, name):
    logger.debug(
        f"{name}, mean: {tensor.mean():.5f}, std: {tensor.std():.5f}, min: {tensor.min():.5f}, max: {tensor.max():.5f}"
    )


class ModelParallelTransformerDecoder_xattn(ModelParallelTransformerDecoder):
    """
    Transformer decoder consisting of *args.decoder_layers* layers. Each layer
    is a :class:`ModelParallelTransformerDecoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~metaseq.data.Dictionary): decoding dictionary
        embed_tokens (torch.nn.Embedding): output embedding
    """

    def __init__(self, args, dictionary, embed_tokens, embed_tokens2):
        self.args = args
        super().__init__(args, dictionary, embed_tokens)
        self.cm3 = ModelParallelTransformerDecoder()
        self.llm = ModelParallelTransformerDecoder()
        
        
    def extract_features(
        self,
        prev_output_tokens,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        token_embeddings: Optional[torch.Tensor] = None,
        self_attn_padding_mask: Optional[Tensor] = None,
    ):
        # compute self-attention padding mask (involves device-to-host transfer,
        # so put it at the top of the forward)
        if (
            self_attn_padding_mask is None
            and prev_output_tokens.eq(self.padding_idx).any()
        ):
            self_attn_padding_mask = prev_output_tokens.eq(self.padding_idx)

        # embed tokens and positions
        # x is T x B x C
        x, tok, pos = self.forward_embedding(
            prev_output_tokens, token_embeddings, incremental_state
        )

        # see BaseDecoder for important information about
        # incremental state. Note that it may be an empty dictionary.
        if not incremental_state:
            self_attn_mask = self.buffered_future_mask(x, prev_output_tokens)
        else:
            self_attn_mask = None

        # decoder layers
        # store other representations for instrumentation in VocabParallelCrossEntCrit
        # Note: we are only storing the embeddings output and output of final transformer block
        # instead of all inner representations, as thats the only thing being logged and storing
        # all intermediate representation causes OOM for large models during validation.
        inner_states: List[Optional[Tensor]] = [{"tok": tok, "pos": pos, "emb": x}]
        for idx, layer in enumerate(self.layers):
            x = layer(
                x,
                incremental_state=incremental_state,
                self_attn_mask=self_attn_mask,
                self_attn_padding_mask=self_attn_padding_mask,
                recompute_fc1=(idx < getattr(self.args, "recompute_fc1_num_layers", 0)),
            )
        inner_states.append(x)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        # Returned x is T x B x C here, as sequence_parallel requires T to be first dim
        return x, {"inner_states": inner_states}

    def output_layer(self, features, **kwargs):
        """Project features to the vocabulary size."""
        if not self.share_input_output_embed:
            # TODO[Susan]: Remove this & make compatible.
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
        # Gather output if model is in inference mode (i.e. eval_lm or generation) cause both are not yet
        # compatible with vocab parallel embeddings
        if getattr(self.args, "criterion") != "vocab_parallel_cross_entropy" or getattr(
            self, "inference", False
        ):
            x = gather_from_tensor_model_parallel_region(x).contiguous()

        return x

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        if self.embed_positions is None:
            return self.max_target_positions
        return min(self.max_target_positions, self.embed_positions.max_positions)

    def buffered_future_mask(self, tensor, input_tokens=None):
        cur_seq_len, batch_size = tensor.size(0), tensor.size(1)
        max_seq_len = self.max_positions()
        need_to_make_new_mask = (
            self._future_mask.size(0) == 0
            or (not self._future_mask.device == tensor.device)
            or self._future_mask.size(1) < max_seq_len
            or (
                self.use_alibi
                and self._future_mask.size(0)
                != (batch_size * self.args.decoder_attention_heads)
            )
            or (self.self_attn_doc_sep != UNSPECIFIED_DOC_SEP)
        )

        # self._future_mask.device != tensor.device is not working in TorchScript. This is a workaround.
        if need_to_make_new_mask:
            self._future_mask = torch.triu(
                utils.fill_with_neg_inf(
                    torch.zeros([max_seq_len, max_seq_len], device=tensor.device)
                ),
                1,
            )
            if self.self_attn_doc_sep != UNSPECIFIED_DOC_SEP:
                # Code to accomodate dynamic attention when document seperator is used
                assert input_tokens is not None
                self._future_mask = self._future_mask[:cur_seq_len, :cur_seq_len]
                self._future_mask = self._future_mask.unsqueeze(0).repeat(
                    batch_size, 1, 1
                )
                doc_id_indices = (
                    (input_tokens == self.self_attn_doc_sep).nonzero().tolist()
                )
                for indices in doc_id_indices:
                    self._future_mask[
                        indices[0], indices[1] + 1 :, : indices[1] + 1
                    ] = float("-inf")

            if self.use_alibi:
                alibi = self.alibi.repeat(batch_size, 1, 1)  # batch_size, 1, 1
                self._future_mask = self._future_mask.unsqueeze(0) + alibi

        self._future_mask = self._future_mask.to(tensor)
        if self.use_alibi:
            return self._future_mask[
                : batch_size * self.args.decoder_attention_heads,
                :cur_seq_len,
                :cur_seq_len,
            ]
        elif self.self_attn_doc_sep != UNSPECIFIED_DOC_SEP:
            return self._future_mask
        else:
            return self._future_mask[:cur_seq_len, :cur_seq_len]

    # This hook used as proxy for tracking state if model is in eval or generation mode.
    def make_generation_fast_(self, **unused):
        self.inference = True

    def forward(
        self,
        prev_output_tokens,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        features_only: bool = False,
        src_lengths: Optional[Any] = None,
        token_embeddings: Optional[torch.Tensor] = None,
        self_attn_padding_mask: Optional[Tensor] = None,
    ):
        """
        Includes several features from "Jointly Learning to Align and
        Translate with Transformer Models" (Garg et al., EMNLP 2019).

        Args:
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for teacher forcing
            incremental_state (dict): dictionary used for storing state during
                :ref:`Incremental decoding`
            features_only (bool, optional): only return features without
                applying output layer (default: False).
            token_embeddings (torch.Tensor, optional): precomputed embeddings
                default `None` will recompute embeddings
            self_attn_padding_mask (torch.Tensor, optional): precomputed padding
                mask for self-attention (default None will recompute mask)

        Returns:
            tuple:
                - the decoder's output of shape `(batch, tgt_len, vocab)`
                - a dictionary with any model-specific outputs
        """

        # see BaseDecoder for important information about incremental state
        x, extra = self.extract_features(
            prev_output_tokens,
            incremental_state=incremental_state,
            token_embeddings=token_embeddings,
            self_attn_padding_mask=self_attn_padding_mask,
        )
        if not features_only:
            x = self.output_layer(x)

        # Transposing back to B x T x C, so that the interface stays the same.
        x = x.transpose(0, 1).contiguous()
        return x, extra
