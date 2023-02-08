# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn

try:
    from megatron.mpu import VocabParallelEmbedding

    has_megatron_submodule = True
except (ImportError, ModuleNotFoundError):
    has_megatron_submodule = False

from dataclasses import dataclass, field
from typing import Optional

from omegaconf import II

from metaseq.dataclass.constants import ATTN_CHOICES, UNSPECIFIED_DOC_SEP

from metaseq import utils
from metaseq.dataclass import ChoiceEnum, MetaseqDataclass
from metaseq.models import (
    LanguageModel,
    register_model,
    register_model_architecture,
)
from metaseq.models.transformer_decoder import (
    DEFAULT_MIN_PARAMS_TO_WRAP,
    TransformerDecoder,
    ModelParallelTransformerDecoder,
)
from metaseq.modules.embedding import Embedding
from metaseq.modules.activation_functions import get_available_activation_fns

import logging

DEFAULT_MAX_TARGET_POSITIONS = 1024
logger = logging.getLogger(__name__)


@dataclass
class TransformerLanguageModelConfig(MetaseqDataclass):
    activation_fn: ChoiceEnum(get_available_activation_fns()) = field(
        default="relu", metadata={"help": "activation function to use"}
    )
    dropout: float = field(default=0.1, metadata={"help": "dropout probability"})
    attention_dropout: float = field(
        default=0.0, metadata={"help": "dropout probability for attention weights"}
    )
    decoder_embed_dim: int = field(
        default=512, metadata={"help": "decoder embedding dimension"}
    )
    decoder_ffn_embed_dim: int = field(
        default=2048, metadata={"help": "decoder embedding dimension for FFN"}
    )
    decoder_layers: int = field(default=6, metadata={"help": "num decoder layers"})
    decoder_attention_heads: int = field(
        default=8, metadata={"help": "num decoder attention heads"}
    )
    share_decoder_input_output_embed: bool = field(
        default=False, metadata={"help": "share decoder input and output embeddings"}
    )
    decoder_learned_pos: bool = field(
        default=False,
        metadata={"help": "use learned positional embeddings in the decoder"},
    )
    decoder_learned_sinusoidal: bool = field(
        default=False,
        metadata={
            "help": "use learned positional embeddings init with sinusoidal in the decoder"
        },
    )
    no_scale_embedding: bool = field(
        default=False, metadata={"help": "if True, dont scale embeddings"}
    )
    checkpoint_activations: bool = field(
        default=False, metadata={"help": "checkpoint activations at each layer"}
    )
    offload_activations: bool = field(
        default=False,
        metadata={"help": "move checkpointed activations to CPU after they are used."},
    )
    # config for Fully Sharded Data Parallel (FSDP) training
    min_params_to_wrap: int = field(
        default=DEFAULT_MIN_PARAMS_TO_WRAP,
        metadata={
            "help": (
                "minimum number of params for a layer to be wrapped with FSDP() when "
                "training with --ddp-backend=fully_sharded. Smaller values will "
                "improve memory efficiency, but may make torch.distributed "
                "communication less efficient due to smaller input sizes. This option "
                "is set to 0 (i.e., always wrap) when --checkpoint-activations or "
                "--offload-activations are passed."
            )
        },
    )
    # ALiBi
    alibi: bool = field(
        default=False,
        metadata={
            "help": "use the ALiBi position method instead of regular position embeddings"
        },
    )
    # Dynamic Attention
    self_attn_doc_sep: int = field(
        default=UNSPECIFIED_DOC_SEP,
        metadata={
            "help": "use dynamic self attention masking when document separator ID is specified"
        },
    )
    fsdp_checkpoint_wrap_layer_frequency: int = field(
        default=1,
        metadata={
            "help": "group transformer blocks and wrap the group in checkpoint and FSDP wrapper together"
        },
    )
    distribute_checkpointed_activations: bool = field(
        default=False,
        metadata={
            "help": "distribute offloaded checkpoints to tensor parallel gpus. "
            "It adds extra within node all_reduce but reduces checkpointed activations significantly,"
            "so a good way to trade speed for gpu memory."
        },
    )
    tensor_parallel_init_model_on_gpu: bool = field(
        default=False,
        metadata={
            "help": "initialize model directly on gpu and possibly fp16 for tensor parallel, shoudl be faster to init model."
        },
    )
    full_megatron_init: bool = field(
        default=False,
        metadata={"help": "Exact same init as Megatron"},
    )
    full_megatron_init_scalar: float = field(
        default=1.0,
        metadata={
            "help": "Factor to scale sigma by for the second layer in FFN and out_proj of MHA"
        },
    )
    pos_init_scalar: float = field(
        default=1.0,
        metadata={"help": "Factor to scale positional embedding init by."},
    )
    truncate_init: bool = field(
        default=False,
        metadata={"help": "Truncate gaussian init to +/- 3 stddevs"},
    )
    megatron_init_sigma: float = field(
        default=0.006,
        metadata={"help": "Sigma for megatron initialization"},
    )
    no_emb_dropout: Optional[bool] = field(
        default=False, metadata={"help": "Avoid emb dropout for decoder"}
    )
    disable_bias: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Remove biases from all matrix projection, similar to PaLM paper,"
            " note this doesn't remove bias from layernorm"
        },
    )
    disable_affine_ln: Optional[bool] = field(
        default=False, metadata={"help": "disable weight and bias of layer norm"}
    )
    attn_variant: ATTN_CHOICES = field(
        default="default", metadata={"help": "variant to use for attention"}
    )
    xf_attn_op: str = field(
        default="None",
        metadata={
            "help": "which memory efficient attention operation to use from xFormers."
        },
    )
    recompute_fc1_num_layers: Optional[int] = field(
        default=0,
        metadata={
            "help": "Num layers for which to recompute FC1 in backwards, "
            "only applicable when --sequence-parallel option is set"
        },
    )
    # options from other parts of the config
    add_bos_token: bool = II("task.add_bos_token")
    tokens_per_sample: int = II("task.tokens_per_sample")
    max_target_positions: Optional[int] = II("task.max_target_positions")
    memory_efficient_fp16: bool = II("common.memory_efficient_fp16")
    fp16: bool = II("common.fp16")
    fp16_no_flatten_grads: bool = II("common.fp16_no_flatten_grads")
    ddp_backend: str = II("distributed_training.ddp_backend")
    world_size: int = II("distributed_training.distributed_world_size")
    distributed_rank: int = II("distributed_training.distributed_rank")
    batch_size: Optional[int] = II("dataset.batch_size")
    batch_size_valid: Optional[int] = II("dataset.batch_size_valid")
    model_parallel_size: int = II("common.model_parallel_size")


@register_model("transformer_lm", dataclass=TransformerLanguageModelConfig)
class TransformerLanguageModel(LanguageModel):
    def __init__(self, decoder):
        super().__init__(decoder)

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        if getattr(args, "max_target_positions", None) is None:
            args.max_target_positions = getattr(
                args, "tokens_per_sample", DEFAULT_MAX_TARGET_POSITIONS
            )

        embed_tokens = cls.build_embedding(
            args, task.source_dictionary, args.decoder_embed_dim
        )
        decoder = TransformerDecoder(
            args,
            task.target_dictionary,
            embed_tokens,
        )
        return cls(decoder)

    @classmethod
    def build_embedding(cls, args, dictionary, embed_dim, path=None):
        return Embedding(
            len(dictionary),
            embed_dim,
            dictionary.pad(),
            initialize_params_on_gpu=getattr(
                args, "tensor_parallel_init_model_on_gpu", False
            ),
            dtype=utils.get_model_init_dtype(args),
        )


@register_model("model_parallel_transformer_lm")
class ModelParallelTransformerLanguageModel(TransformerLanguageModel):
    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        if not has_megatron_submodule:
            raise ImportError(
                "\n\nPlease install megatron using the setup instructions!"
            )

        # make sure all arguments are present in older models
        base_lm_architecture(args)

        task.source_dictionary.pad_to_multiple_(8)
        task.target_dictionary.pad_to_multiple_(8)

        # task.source_dictionary.pad_to_multiple_(args.model_parallel_size * 8)
        # task.target_dictionary.pad_to_multiple_(args.model_parallel_size * 8)

        if getattr(args, "max_target_positions", None) is None:
            args.max_target_positions = getattr(
                args, "tokens_per_sample", DEFAULT_MAX_TARGET_POSITIONS
            )

        embed_tokens = cls.build_embedding(
            args, task.source_dictionary, args.decoder_embed_dim
        )
        assert getattr(
            args, "use_sharded_state", False
        ), "Use sharded state must be True for tensor parallel, otherwise model saving and loaded might be broken"

        if getattr(args, "sequence_parallel", False):
            assert (
                getattr(args, "model_parallel_size", 1) > 1
            ), "--sequence-parallel only works when --model-parallel-size is greater than 1"
            assert (
                getattr(args, "dropout", 0.0) == 0.0
            ), "havent yet tested if rng states are correct for dropout with seq_parallel"
            assert (
                getattr(args, "activation_fn", "gelu") == "gelu"
                or getattr(args, "activation_fn", "gelu") == "relu"
            ), "For now only supports gelu and relu"
            assert not getattr(
                args, "checkpoint_activations", False
            ), "Cannot set --checkpoint-activations with sequence parallel."
            assert not getattr(
                args, "distribute_checkpointed_activations", False
            ), "Cannot set --distribute-checkpointed-activations with sequence parallel."

        decoder = ModelParallelTransformerDecoder(
            args,
            task.target_dictionary,
            embed_tokens,
        )
        return cls(decoder)

    @staticmethod
    def add_args(parser):
        TransformerLanguageModel.add_args(parser)

    @classmethod
    def build_embedding(cls, args, dictionary, embed_dim, path=None):
        def _vocab_init(tensor, **kwargs):
            std = embed_dim**-0.5
            if getattr(args, "truncate_init", False):
                nn.init.trunc_normal_(tensor, mean=0, std=std, a=-3 * std, b=3 * std)
            else:
                nn.init.normal_(tensor, mean=0, std=std)
            nn.init.constant_(tensor[1], 0)

        def _vocab_init_megatron(tensor, **kwargs):
            std = getattr(args, "megatron_init_sigma", 0.006)
            if getattr(args, "truncate_init", False):
                nn.init.trunc_normal_(tensor, mean=0, std=std, a=-3 * std, b=3 * std)
            else:
                nn.init.normal_(tensor, mean=0, std=std)
            nn.init.constant_(tensor[1], 0)

        if getattr(args, "memory_efficient_fp16", False):
            dtype = torch.bfloat16 if getattr(args, "bf16", False) else torch.half
        else:
            dtype = torch.float32

        embed_tokens = VocabParallelEmbedding(
            len(dictionary),
            embed_dim,
            dictionary.pad(),
            init_method=_vocab_init_megatron
            if getattr(args, "full_megatron_init", False)
            else _vocab_init,
            use_cpu_initialization=not getattr(
                args, "tensor_parallel_init_model_on_gpu", False
            ),
            dtype=dtype,
        )
        return embed_tokens


def base_lm_architecture(args):
    args.activation_fn = getattr(args, "activation_fn", "relu")
    args.dropout = getattr(args, "dropout", 0.1)
    args.attention_dropout = getattr(args, "attention_dropout", 0.0)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 512)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 2048)
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", False
    )
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)
    args.decoder_learned_sinusoidal = getattr(args, "decoder_learned_sinusoidal", False)
    args.no_scale_embedding = getattr(args, "no_scale_embedding", False)
    args.add_bos_token = getattr(args, "add_bos_token", False)


@register_model_architecture("model_parallel_transformer_lm", "transformer_lm_megatron")
def transformer_lm_megatron(args):
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 3072)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 3072 * 4)
    args.decoder_layers = getattr(args, "decoder_layers", 72)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 32)
    args.dropout = getattr(args, "dropout", 0.1)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_fn = getattr(args, "activation_fn", "gelu")
    base_lm_architecture(args)


@register_model_architecture("model_parallel_transformer_lm", "transformer_lm_gpt")
def transformer_lm_gpt(args):
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 768)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 3072)
    args.decoder_layers = getattr(args, "decoder_layers", 12)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 12)
    args.dropout = getattr(args, "dropout", 0.1)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_fn = getattr(args, "activation_fn", "gelu")
    base_lm_architecture(args)


@register_model_architecture(
    "model_parallel_transformer_lm", "transformer_lm_gpt2_tiny"
)
def transformer_lm_gpt2_tiny(args):
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 64)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 64)
    args.decoder_layers = getattr(args, "decoder_layers", 2)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 1)
    args.dropout = getattr(args, "dropout", 0.1)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_fn = getattr(args, "activation_fn", "gelu")
    base_lm_architecture(args)
