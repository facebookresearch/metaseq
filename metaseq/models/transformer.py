# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import math
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch import Tensor
from metaseq.dataclass.constants import UNSPECIFIED_DOC_SEP

from metaseq import utils
from metaseq.distributed import utils as distributed_utils, fsdp_wrap
from metaseq.models import BaseEncoder, IncrementalDecoder
from metaseq.modules import (
    Dropout,
    LayerNorm,
    PositionalEmbedding,
    TransformerDecoderLayer,
    TransformerEncoderLayer,
)
from metaseq.modules.checkpoint_activations import checkpoint_wrapper

logger = logging.getLogger(__name__)


DEFAULT_MAX_SOURCE_POSITIONS = 1024
DEFAULT_MAX_TARGET_POSITIONS = 1024
DEFAULT_MIN_PARAMS_TO_WRAP = int(1e8)


class TransformerEncoder(BaseEncoder):
    """
    Transformer encoder consisting of *args.encoder_layers* layers. Each layer
    is a :class:`TransformerEncoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~metaseq.data.Dictionary): encoding dictionary
        embed_tokens (torch.nn.Embedding): input embedding
    """

    def __init__(self, args, dictionary, embed_tokens):
        self.args = args
        super().__init__(dictionary)
        self.register_buffer("version", torch.Tensor([3]))

        self.dropout_module = Dropout(args.dropout, module_name=self.__class__.__name__)

        embed_dim = embed_tokens.embedding_dim
        self.padding_idx = embed_tokens.padding_idx
        self.max_source_positions = args.max_source_positions
        self.embed_tokens = embed_tokens
        self.embed_scale = 1.0 if args.no_scale_embedding else math.sqrt(embed_dim)

        self.embed_positions = (
            PositionalEmbedding(
                args.max_source_positions,
                embed_dim,
                self.padding_idx,
                learned=args.encoder_learned_pos,
            )
            if args.encoder_learned_pos
            else None
        )

        self.layers = nn.ModuleList([])
        for i in range(args.encoder_layers):
            self.layers.append(self.build_encoder_layer(args))
        self.num_layers = len(self.layers)

        if args.encoder_normalize_before:
            self.layer_norm = LayerNorm(embed_dim)
        else:
            self.layer_norm = None

    def build_encoder_layer(self, args):
        layer = TransformerEncoderLayer(args)
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
        self, src_tokens, token_embedding: Optional[torch.Tensor] = None
    ):
        # embed tokens and positions
        if token_embedding is None:
            token_embedding = self.embed_tokens(src_tokens)
        x = embed = self.embed_scale * token_embedding
        if self.embed_positions is not None:
            x = embed + self.embed_positions(src_tokens)
        x = self.dropout_module(x)
        return x, embed

    def forward(
        self,
        src_tokens,
        src_lengths: Optional[torch.Tensor] = None,
        token_embeddings: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (torch.LongTensor): lengths of each source sentence of
                shape `(batch)`
            token_embeddings (torch.Tensor, optional): precomputed embeddings
                default `None` will recompute embeddings

        Returns:
            dict:
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
                - **encoder_embedding** (Tensor): the (scaled) embedding lookup
                  of shape `(batch, src_len, embed_dim)`
        """
        return self.forward_scriptable(src_tokens, src_lengths, token_embeddings)

    # TorchScript doesn't support super() method so that the scriptable Subclass
    # can't access the base class model in Torchscript.
    # Current workaround is to add a helper function with different name and
    # call the helper function from scriptable Subclass.
    def forward_scriptable(
        self,
        src_tokens,
        src_lengths: Optional[torch.Tensor] = None,
        token_embeddings: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (torch.LongTensor): lengths of each source sentence of
                shape `(batch)`
            token_embeddings (torch.Tensor, optional): precomputed embeddings
                default `None` will recompute embeddings

        Returns:
            dict:
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
                - **encoder_embedding** (Tensor): the (scaled) embedding lookup
                  of shape `(batch, src_len, embed_dim)`
        """
        # compute padding mask
        encoder_padding_mask = src_tokens.eq(self.padding_idx)
        has_pads = encoder_padding_mask.any()

        x, encoder_embedding = self.forward_embedding(src_tokens, token_embeddings)

        # account for padding while computing the representation
        if has_pads:
            x = x * (1 - encoder_padding_mask.unsqueeze(-1).type_as(x))

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # encoder layers
        l_aux = []
        for layer in self.layers:
            x, l_aux_i = layer(
                x, encoder_padding_mask=encoder_padding_mask if has_pads else None
            )
            l_aux.append(l_aux_i)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        # The Pytorch Mobile lite interpreter does not supports returning NamedTuple in
        # `forward` so we use a dictionary instead.
        # TorchScript does not support mixed values so the values are all lists.
        # The empty list is equivalent to None.
        return {
            "encoder_out": [x],  # T x B x C
            "encoder_padding_mask": [encoder_padding_mask],  # B x T
            "encoder_embedding": [encoder_embedding],  # B x T x C
            "src_tokens": [],
            "src_lengths": [],
            "l_aux": l_aux,
        }

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        if self.embed_positions is None:
            return self.max_source_positions
        return min(self.max_source_positions, self.embed_positions.max_positions)


class TransformerDecoderMultiLayerBlockModule(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, x, **kwargs):
        l_aux = []
        inner_states = []
        for layer in self.layers:
            x, layer_attn, _, l_aux_i = layer(x, **kwargs)
            inner_states.append(x)
        return x, layer_attn, inner_states, l_aux


def _log_weight_stats(tensor, name):
    logger.debug(
        f"{name}, mean: {tensor.mean():.5f}, std: {tensor.std():.5f}, min: {tensor.min():.5f}, max: {tensor.max():.5f}"
    )


class TransformerDecoder(IncrementalDecoder):
    """
    Transformer decoder consisting of *args.decoder_layers* layers. Each layer
    is a :class:`TransformerDecoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~metaseq.data.Dictionary): decoding dictionary
        embed_tokens (torch.nn.Embedding): output embedding
        no_encoder_attn (bool, optional): whether to attend to encoder outputs
            (default: False).
    """

    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False):
        self.args = args
        super().__init__(dictionary)
        self.register_buffer("version", torch.Tensor([3]))
        self._future_mask = torch.empty(0)

        self.dropout_module = Dropout(args.dropout, module_name=self.__class__.__name__)

        if getattr(args, "no_emb_dropout", False):
            self.dropout_module = None

        self.share_input_output_embed = args.share_decoder_input_output_embed
        input_embed_dim = embed_tokens.embedding_dim
        embed_dim = args.decoder_embed_dim
        self.embed_dim = embed_dim
        self.output_embed_dim = args.decoder_output_dim
        self.padding_idx = embed_tokens.padding_idx
        self.max_target_positions = args.max_target_positions
        self.embed_tokens = embed_tokens
        self.embed_scale = 1.0 if args.no_scale_embedding else math.sqrt(embed_dim)

        self.project_in_dim = (
            Linear(input_embed_dim, embed_dim, bias=False)
            if embed_dim != input_embed_dim
            else None
        )
        self.use_alibi: bool = getattr(args, "alibi", False)
        self.self_attn_doc_sep: int = getattr(
            args, "self_attn_doc_sep", UNSPECIFIED_DOC_SEP
        )
        initialize_params_on_gpu = getattr(
            args, "tensor_parallel_init_model_on_gpu", False
        )

        self.embed_positions = (
            PositionalEmbedding(
                self.max_target_positions,
                embed_dim,
                self.padding_idx,
                learned=args.decoder_learned_pos,
                learned_sinusoidal=getattr(args, "decoder_learned_sinusoidal", False),
                full_megatron_init=getattr(args, "full_megatron_init", False),
                megatron_init_sigma=getattr(args, "megatron_init_sigma", 0.006),
            )
            if args.decoder_learned_pos and not self.use_alibi
            else None
        )

        if initialize_params_on_gpu and self.embed_positions is not None:
            self.embed_positions = utils.floating_point_precision_convertor(
                self.embed_positions.cuda(),
                fp16=getattr(args, "fp16", False),
                memory_efficient_fp16=getattr(args, "memory_efficient_fp16", False),
                bf16=getattr(args, "bf16", False),
            )

        self.cross_self_attention = getattr(args, "cross_self_attention", False)

        self.layers = nn.ModuleList([])
        layers = []
        for i in range(args.decoder_layers):
            layers.append(
                self.build_decoder_layer(
                    args,
                    no_encoder_attn=no_encoder_attn,
                )
            )
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

        _log_weight_stats(self.embed_tokens.weight, "embed tokens")

        self.num_layers = len(self.layers)

        if args.decoder_normalize_before:
            self.layer_norm = LayerNorm(
                embed_dim,
                elementwise_affine=not getattr(args, "disable_affine_ln", False),
            )
            if initialize_params_on_gpu:
                self.layer_norm = utils.floating_point_precision_convertor(
                    self.layer_norm.cuda(),
                    fp16=getattr(args, "fp16", False),
                    memory_efficient_fp16=getattr(args, "memory_efficient_fp16", False),
                    bf16=getattr(args, "bf16", False),
                )

        else:
            self.layer_norm = None

        self.project_out_dim = (
            Linear(embed_dim, self.output_embed_dim, bias=False)
            if embed_dim != self.output_embed_dim
            else None
        )

        self.output_projection = None
        if self.share_input_output_embed:
            self.output_projection = nn.Linear(
                self.embed_tokens.weight.shape[1],
                self.embed_tokens.weight.shape[0],
                bias=False,
            )
            self.output_projection.weight = self.embed_tokens.weight
        else:
            self.output_projection = nn.Linear(
                self.output_embed_dim, len(dictionary), bias=False
            )
            nn.init.normal_(
                self.output_projection.weight, mean=0, std=self.output_embed_dim**-0.5
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

    def build_base_decoder_layer(self, args, no_encoder_attn=False):
        return TransformerDecoderLayer(args, no_encoder_attn=no_encoder_attn)

    def build_decoder_layer(self, args, no_encoder_attn=False):
        layer = self.build_base_decoder_layer(args, no_encoder_attn)
        for name, param in layer.named_parameters():
            _log_weight_stats(param, name)
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

            # If the positions are pre-computed, padding_idx should not be set.
            # Ref metaseq/metaseq/modules/learned_positional_embedding.py
            if self.embed_positions is not None:
                self.embed_positions.padding_idx = None
        else:
            positions = None
        if self.embed_positions is not None:
            positions = self.embed_positions(
                tokens, incremental_state=incremental_state, positions=positions
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

        if self.dropout_module is not None:
            x = self.dropout_module(x)

        return x, embed, positions

    # forward for TransformerDecoder
    def forward(
        self,
        prev_output_tokens,
        encoder_out: Optional[Dict[str, List[Tensor]]] = None,
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
            encoder_out (optional): output from the encoder, used for
                encoder-side attention
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

        # see IncrementalDecoder for important information about
        # incremental state
        x, extra = self.extract_features(
            prev_output_tokens,
            encoder_out=encoder_out,
            incremental_state=incremental_state,
            token_embeddings=token_embeddings,
            self_attn_padding_mask=self_attn_padding_mask,
        )
        if not features_only:
            x = self.output_layer(x)
        return x, extra

    def extract_features(
        self,
        prev_output_tokens,
        encoder_out: Optional[Dict[str, List[Tensor]]] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        token_embeddings: Optional[torch.Tensor] = None,
        self_attn_padding_mask: Optional[Tensor] = None,
    ):
        return self.extract_features_scriptable(
            prev_output_tokens,
            encoder_out=encoder_out,
            incremental_state=incremental_state,
            token_embeddings=token_embeddings,
            self_attn_padding_mask=self_attn_padding_mask,
        )

    def extract_features_scriptable(
        self,
        prev_output_tokens,
        encoder_out: Optional[Dict[str, List[Tensor]]],
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        token_embeddings: Optional[Tensor] = None,
        self_attn_padding_mask: Optional[Tensor] = None,
    ):
        """
        A scriptable subclass of this class has an extract_features method and calls
        super().extract_features, but super() is not supported in torchscript. A copy
        of this function is made to be used in the subclass instead.
        """
        last_layer_idx = self.num_layers - 1

        # compute self-attention padding mask (involves device-to-host transfer,
        # so put it at the top of the forward)
        if self_attn_padding_mask is None and (
            self.cross_self_attention or prev_output_tokens.eq(self.padding_idx).any()
        ):
            self_attn_padding_mask = prev_output_tokens.eq(self.padding_idx)

        # embed tokens and positions
        x, tok, pos = self.forward_embedding(
            prev_output_tokens, token_embeddings, incremental_state
        )

        # see IncrementalDecoder for important information about
        # incremental state. Note that it may be an empty dictionary.
        if not incremental_state:
            self_attn_mask = self.buffered_future_mask(x, prev_output_tokens)
        else:
            self_attn_mask = None

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # decoder layers
        attn: Optional[Tensor] = None
        # store other representations for instrumentation in VocabParallelCrossEntCrit
        # Note: we are only storing the embeddings output and output of final transformer block
        # instead of all inner representations, as thats the only thing being logged and storing
        # all intermediate representation causes OOM for large models during validation.
        inner_states: List[Optional[Tensor]] = [{"tok": tok, "pos": pos, "emb": x}]
        if encoder_out is None:
            l_aux = []
        else:
            l_aux = encoder_out["l_aux"] if "l_aux" in encoder_out else []
        for idx, layer in enumerate(self.layers):
            x, layer_attn, _, l_aux_i = layer(
                x,
                encoder_out=encoder_out["encoder_out"][0]
                if (encoder_out is not None and len(encoder_out["encoder_out"]) > 0)
                else None,
                encoder_padding_mask=encoder_out["encoder_padding_mask"][0]
                if (
                    encoder_out is not None
                    and len(encoder_out["encoder_padding_mask"]) > 0
                )
                else None,
                incremental_state=incremental_state,
                self_attn_mask=self_attn_mask,
                self_attn_padding_mask=self_attn_padding_mask,
                need_attn=bool((idx == last_layer_idx)),
                need_head_weights=bool((idx == last_layer_idx)),
            )
            l_aux.append(l_aux_i)
            if layer_attn is not None and idx == last_layer_idx:
                attn = layer_attn.float().to(x)

        inner_states.append(x)
        if attn is not None:
            # average probabilities over heads
            attn = attn.mean(dim=0)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if self.project_out_dim is not None:
            x = self.project_out_dim(x)

        return x, {"attn": [attn], "inner_states": inner_states, "l_aux": l_aux}

    def output_layer(self, features):
        """Project features to the vocabulary size."""
        return self.output_projection(features)

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        if self.embed_positions is None:
            return self.max_target_positions
        return min(self.max_target_positions, self.embed_positions.max_positions)

    def buffered_future_mask(self, tensor, input_tokens=None):
        batch_size, cur_seq_len = tensor.size(0), tensor.size(1)
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


def Embedding(
    num_embeddings, embedding_dim, padding_idx, initialize_params_on_gpu=False
):
    # Passing weights initialized on GPU.
    device = torch.cuda.current_device() if initialize_params_on_gpu else None
    dtype = torch.half if initialize_params_on_gpu else torch.float
    weight = torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
    nn.init.normal_(weight, mean=0, std=embedding_dim**-0.5)
    nn.init.constant_(weight[padding_idx], 0)
    m = nn.Embedding(
        num_embeddings, embedding_dim, padding_idx=padding_idx, _weight=weight
    )
    return m


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m
