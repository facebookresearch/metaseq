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


class ModelParallelTransformerDecoder(BaseDecoder):
    """
    Transformer decoder consisting of *args.decoder_layers* layers. Each layer
    is a :class:`ModelParallelTransformerDecoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~metaseq.data.Dictionary): decoding dictionary
        embed_tokens (torch.nn.Embedding): output embedding
    """

    def __init__(self, args, dictionary, embed_tokens):
        self.args = args
        super().__init__(dictionary)
        self.register_buffer("version", torch.Tensor([3]))
        self._future_mask = torch.empty(0)

        self.dropout_module = Dropout(args.dropout, module_name=self.__class__.__name__)

        if getattr(args, "no_emb_dropout", False):
            self.dropout_module = None

        self.share_input_output_embed = args.share_decoder_input_output_embed
        self.embed_dim = args.decoder_embed_dim
        self.padding_idx = embed_tokens.padding_idx
        self.max_target_positions = args.max_target_positions
        self.embed_tokens = embed_tokens
        self.embed_scale = 1.0 if args.no_scale_embedding else math.sqrt(self.embed_dim)

        initialize_params_on_gpu = getattr(
            args, "tensor_parallel_init_model_on_gpu", False
        )
        device = torch.cuda.current_device() if initialize_params_on_gpu else None
        dtype = utils.get_model_init_dtype(args)

        self.use_alibi: bool = getattr(args, "alibi", False)
        self.self_attn_doc_sep: int = getattr(
            args, "self_attn_doc_sep", UNSPECIFIED_DOC_SEP
        )

        self.embed_positions = (
            PositionalEmbedding(
                self.max_target_positions,
                self.embed_dim,
                self.padding_idx,
                learned=args.decoder_learned_pos,
                learned_sinusoidal=getattr(args, "decoder_learned_sinusoidal", False),
                full_megatron_init=getattr(args, "full_megatron_init", False),
                pos_init_scalar=getattr(args, "pos_init_scalar", 1.0),
                megatron_init_sigma=getattr(args, "megatron_init_sigma", 0.006),
                truncate_init=getattr(args, "truncate_init", False),
            )
            if args.decoder_learned_pos and not self.use_alibi
            else None
        )
        self.embed_positions.to(device).to(dtype)

        self.layers = nn.ModuleList([])
        layers = []
        for i in range(args.decoder_layers):
            layers.append(self.build_decoder_layer(args))

        if getattr(self.args, "fsdp_checkpoint_wrap_layer_frequency", 1) > 1:
            assert (
                len(layers) % self.args.fsdp_checkpoint_wrap_layer_frequency == 0
            ), "num layers should be divisible by checkpoint wrap frequency"
            for i in range(
                0, len(layers), self.args.fsdp_checkpoint_wrap_layer_frequency
            ):
                layer_block = TransformerDecoderMultiLayerBlockModule(
                    layers[i : i + self.args.fsdp_checkpoint_wrap_layer_frequency]
                )
                checkpoint = getattr(args, "checkpoint_activations", False)
                if checkpoint:
                    offload_to_cpu = getattr(args, "offload_activations", False)
                    distribute_checkpointed_activations = getattr(
                        args, "distribute_checkpointed_activations", False
                    )
                    layer_block = checkpoint_wrapper(
                        layer_block,
                        offload_to_cpu=offload_to_cpu,
                        distribute_checkpointed_activations=distribute_checkpointed_activations,
                    )
                # if we are checkpointing, enforce that FSDP always wraps the
                # checkpointed layer, regardless of layer size
                min_params_to_wrap = (
                    getattr(args, "min_params_to_wrap", DEFAULT_MIN_PARAMS_TO_WRAP)
                    if not checkpoint
                    else 0
                )
                layer_block = fsdp_wrap(
                    layer_block,
                    min_num_params=min_params_to_wrap,
                    process_group=distributed_utils.get_data_parallel_group(),
                )
                self.layers.append(layer_block)
        else:
            self.layers = nn.ModuleList(layers)

        log_weight_stats(self.embed_tokens.weight, "embed tokens")

        self.num_layers = len(self.layers)

        self.layer_norm = LayerNorm(
            self.embed_dim,
            elementwise_affine=not getattr(args, "disable_affine_ln", False),
        )
        self.layer_norm.to(device).to(dtype)

        self.output_projection = None
        if self.share_input_output_embed:
            self.output_projection = Linear(
                self.embed_tokens.weight.shape[1],
                self.embed_tokens.weight.shape[0],
                bias=False,
                initialize_params_on_gpu=initialize_params_on_gpu,
                dtype=dtype,
            )
            self.output_projection.weight = self.embed_tokens.weight
        else:
            self.output_projection = Linear(
                self.embed_dim,
                len(dictionary),
                bias=False,
                initialize_params_on_gpu=initialize_params_on_gpu,
                dtype=dtype,
            )
            nn.init.normal_(
                self.output_projection.weight, mean=0, std=self.embed_dim**-0.5
            )

        if self.use_alibi:
            self.alibi = self._build_alibi_tensor(
                self.max_positions(), args.decoder_attention_heads
            )

    @staticmethod
    def _build_alibi_tensor(max_seq_len: int, n_attention_heads: int):
        """Returns tensor shaped (n_head, 1, max_seq_len)"""

        def get_slopes(n):
            # In the paper, we only train models that have 2^a heads for some a. This function has some good
            # properties that only occur when the input is a power of 2. To maintain that even when the number of
            # heads is not a power of 2, we use this workaround.
            def get_slopes_power_of_2(n):
                start = 2 ** (-(2 ** -(math.log2(n) - 3)))
                ratio = start
                return [start * ratio**i for i in range(n)]

            if math.log2(n).is_integer():
                return get_slopes_power_of_2(n)
            else:
                closest_power_of_2 = 2 ** math.floor(math.log2(n))
                return (
                    get_slopes_power_of_2(closest_power_of_2)
                    + get_slopes(2 * closest_power_of_2)[0::2][: n - closest_power_of_2]
                )

        slopes = torch.Tensor(get_slopes(n_attention_heads))
        # In the next line, the part after the * is what constructs the diagonal matrix (right matrix in Figure 3 in
        # the paper).
        # It doesn't exactly print out the same matrix as we have in Figure 3, but one where all rows are identical.
        # This works because the softmax operation is invariant to translation, and our bias functions are always
        # linear.
        alibi = slopes.unsqueeze(1).unsqueeze(1) * torch.arange(max_seq_len).unsqueeze(
            0
        ).unsqueeze(0).expand(n_attention_heads, -1, -1)
        alibi = alibi.view(n_attention_heads, 1, max_seq_len)
        return alibi

    def build_decoder_layer(self, args):
        layer = ModelParallelTransformerDecoderLayer(args)
        for name, param in layer.named_parameters():
            log_weight_stats(param, name)
        if getattr(args, "fsdp_checkpoint_wrap_layer_frequency", 1) > 1:
            return layer
        checkpoint = getattr(args, "checkpoint_activations", False)
        if checkpoint:
            offload_to_cpu = getattr(args, "offload_activations", False)
            distribute_checkpointed_activations = getattr(
                args, "distribute_checkpointed_activations", False
            )
            layer = checkpoint_wrapper(
                layer,
                offload_to_cpu=offload_to_cpu,
                distribute_checkpointed_activations=distribute_checkpointed_activations,
            )
        # if we are checkpointing, enforce that FSDP always wraps the
        # checkpointed layer, regardless of layer size
        min_params_to_wrap = (
            getattr(args, "min_params_to_wrap", DEFAULT_MIN_PARAMS_TO_WRAP)
            if not checkpoint
            else 0
        )
        layer = fsdp_wrap(
            layer,
            min_num_params=min_params_to_wrap,
            process_group=distributed_utils.get_data_parallel_group(),
        )
        return layer

    def forward_embedding(
        self,
        tokens,
        token_embedding: Optional[torch.Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
    ):
        # embed tokens and positions
        if self.self_attn_doc_sep != UNSPECIFIED_DOC_SEP:
            # create own positions when self_attn_doc_sep is set
            # We are essentially resetting positions based on document separator tokens.
            # For instance, if the doc separator is 2, and the tokens are
            # 143 345 2 5435 2
            # The default positions would be
            # 2 3 4 5 6
            # But with document level attention, we would like to reset positions
            # as well on sentence boundaries. So, the positions become
            # 2 3 4 2 3

            mask = tokens.ne(self.padding_idx).int()
            mask_with_reset = tokens.ne(self.padding_idx).int()
            mask_with_reset[:, :] = 1
            doc_id_indices = (tokens == self.self_attn_doc_sep).nonzero().tolist()

            # Based on the location of document seperator token (batch_doc_indices), we would reset
            # positional embeddings. The document seperator token marks the end of the preceding
            # document.
            for batch_idx in range(tokens.size(0)):
                # The self_attn_doc_sep token marks the end of the previous document. Therefore,
                # we need to add 1 to the indices to mark the start of documents.
                batch_doc_indices = [
                    index[1] + 1
                    for index in doc_id_indices
                    # index[1] + 1 < tokens.size(1) to prevent overflow
                    if index[0] == batch_idx and index[1] + 1 < tokens.size(1)
                ]
                batch_doc_indices.sort()
                for k, doc_sep_idx in enumerate(batch_doc_indices):
                    if k == 0:
                        mask_with_reset[batch_idx, doc_sep_idx] = -doc_sep_idx + 1
                    else:
                        mask_with_reset[batch_idx, doc_sep_idx] = (
                            batch_doc_indices[k - 1] - doc_sep_idx + 1
                        )
            positions = (
                torch.cumsum(mask_with_reset, dim=1).type_as(mask) * mask
            ).long() + self.padding_idx

            # Since positions are pre-computed, padding_idx should not be set.
            # Ref metaseq/metaseq/modules/learned_positional_embedding.py
            if self.embed_positions is not None:
                self.embed_positions.padding_idx = None
        else:
            positions = None

        if self.embed_positions is not None:
            positions = self.embed_positions(
                tokens, incremental_state=incremental_state, positions=positions
            )

        # see BaseDecoder for important information about
        # incremental state
        if incremental_state:
            tokens = tokens[:, -1:]
            if positions is not None:
                positions = positions[:, -1:]

        if token_embedding is None:
            token_embedding = self.embed_tokens(tokens)

        x = embed = self.embed_scale * token_embedding

        if positions is not None:
            x += positions

        if self.dropout_module is not None:
            x = self.dropout_module(x)

        # Returning in T x B x C format as that makes integrating sequence parallelism easier.
        x = x.transpose(0, 1).contiguous()

        is_sequence_parallel = getattr(self.args, "sequence_parallel", False)
        if is_sequence_parallel:
            x = mpu.scatter_to_sequence_parallel_region(x)
        return x, embed, positions

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
