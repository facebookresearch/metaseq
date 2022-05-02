# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import logging

from dataclasses import dataclass, field
from typing import Optional

from omegaconf import II

from metaseq import utils
from metaseq.dataclass import ChoiceEnum, MetaseqDataclass
from metaseq.models import (
    LanguageModel,
    register_model,
    register_model_architecture,
)
from metaseq.models.transformer import (
    DEFAULT_MIN_PARAMS_TO_WRAP,
    Embedding,
    TransformerDecoder,
)

DEFAULT_MAX_TARGET_POSITIONS = 1024

logger = logging.getLogger(__name__)


@dataclass
class TransformerLanguageModelConfig(MetaseqDataclass):
    activation_fn: ChoiceEnum(utils.get_available_activation_fns()) = field(
        default="relu", metadata={"help": "activation function to use"}
    )
    dropout: float = field(default=0.1, metadata={"help": "dropout probability"})
    attention_dropout: float = field(
        default=0.0, metadata={"help": "dropout probability for attention weights"}
    )
    decoder_embed_dim: int = field(
        default=512, metadata={"help": "decoder embedding dimension"}
    )
    decoder_output_dim: int = field(
        default=512, metadata={"help": "decoder output dimension"}
    )
    decoder_input_dim: int = field(
        default=512, metadata={"help": "decoder input dimension"}
    )
    decoder_ffn_embed_dim: int = field(
        default=2048, metadata={"help": "decoder embedding dimension for FFN"}
    )
    decoder_layers: int = field(default=6, metadata={"help": "num decoder layers"})
    decoder_attention_heads: int = field(
        default=8, metadata={"help": "num decoder attention heads"}
    )
    decoder_normalize_before: bool = field(
        default=False, metadata={"help": "apply layernorm before each decoder block"}
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
    use_stable_embedding: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Use bitsandbytes StableEmbeddingLayer which saves embedding state in fp32",
            "argparse_alias": "--stable-emb",
        },
    )
    # NormFormer
    scale_fc: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Insert LayerNorm between fully connected layers",
        },
    )
    scale_attn: Optional[bool] = field(
        default=False, metadata={"help": "Insert LayerNorm after attention"}
    )
    scale_heads: Optional[bool] = field(
        default=False,
        metadata={"help": "Learn a scale coefficient for each attention head"},
    )

    # ALiBi
    alibi: bool = field(
        default=False,
        metadata={
            "help": "use the ALiBi position method instead of regular position embeddings"
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
    megatron_init_sigma: float = field(
        default=0.006,
        metadata={"help": "Sigma for megatron initialization"},
    )
    sync_ln_variance: Optional[bool] = field(
        default=False,
        metadata={"help": "sync_ln_variance stats", "argparse_alias": "--sync-ln"},
    )

    no_emb_dropout: Optional[bool] = field(
        default=False, metadata={"help": "Avoid emb dropout for decoder"}
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
            args, task.source_dictionary, args.decoder_input_dim
        )
        decoder = TransformerDecoder(
            args,
            task.target_dictionary,
            embed_tokens,
            no_encoder_attn=True,
        )
        return cls(decoder)

    @classmethod
    def build_embedding(cls, args, dictionary, embed_dim, path=None):
        if getattr(args, "use_stable_embedding", False):
            import bitsandbytes as bnb

            if not args.no_scale_embedding:
                logger.warning(
                    "It is recommended to pass --no-scale-embedding with --use-stable-embedding"
                )
            return bnb.nn.StableEmbedding(len(dictionary), embed_dim, dictionary.pad())

        else:
            return Embedding(
                len(dictionary),
                embed_dim,
                dictionary.pad(),
                initialize_params_on_gpu=getattr(
                    args, "tensor_parallel_init_model_on_gpu", False
                ),
            )


def base_lm_architecture(args):
    args.dropout = getattr(args, "dropout", 0.1)
    args.attention_dropout = getattr(args, "attention_dropout", 0.0)

    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 512)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 2048)
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)
    args.decoder_learned_sinusoidal = getattr(args, "decoder_learned_sinusoidal", False)
    args.activation_fn = getattr(args, "activation_fn", "relu")

    args.add_bos_token = getattr(args, "add_bos_token", False)
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", False
    )

    args.decoder_output_dim = getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)

    # Model training is not stable without this
    args.decoder_normalize_before = True

    args.no_scale_embedding = getattr(args, "no_scale_embedding", False)
    args.checkpoint_activations = getattr(args, "checkpoint_activations", False)
    args.offload_activations = getattr(args, "offload_activations", False)
    if args.offload_activations:
        args.checkpoint_activations = True


@register_model_architecture("transformer_lm", "transformer_lm_gpt")
def transformer_lm_gpt(args):
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 768)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 3072)
    args.decoder_layers = getattr(args, "decoder_layers", 12)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 12)
    args.dropout = getattr(args, "dropout", 0.1)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_fn = getattr(args, "activation_fn", "gelu")
    base_lm_architecture(args)


@register_model_architecture("transformer_lm", "transformer_lm_gpt2_tiny")
def transformer_lm_gpt2_tiny(args):
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 64)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 64)
    args.decoder_layers = getattr(args, "decoder_layers", 2)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 1)
    args.dropout = getattr(args, "dropout", 0.1)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_fn = getattr(args, "activation_fn", "gelu")
    base_lm_architecture(args)


@register_model_architecture("transformer_lm", "transformer_lm_gpt2_bigger")
def transformer_lm_gpt2_bigger(args):
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 2048)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 8192)
    args.decoder_layers = getattr(args, "decoder_layers", 48)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 32)
    args.dropout = getattr(args, "dropout", 0.1)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_fn = getattr(args, "activation_fn", "gelu")
    base_lm_architecture(args)
