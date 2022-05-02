# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn as nn
from metaseq.model_parallel.models.transformer import ModelParallelTransformerDecoder
from metaseq.models import register_model, register_model_architecture
from metaseq.models.transformer_lm import TransformerLanguageModel


try:
    from megatron.mpu import VocabParallelEmbedding

    has_megatron_submodule = True
except (ImportError, ModuleNotFoundError):
    has_megatron_submodule = False


DEFAULT_MAX_TARGET_POSITIONS = 1024


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
            args, task.source_dictionary, args.decoder_input_dim
        )
        assert getattr(
            args, "use_sharded_state", False
        ), "Use sharded state must be True for tensor parallel, otherwise model saving and loaded might be broken"
        if getattr(args, "tensor_parallel_init_model_on_gpu", False):
            assert getattr(
                args, "memory_efficient_fp16", False
            ), "GPU initialization is only supported for full fp16 mode for now."

        decoder = ModelParallelTransformerDecoder(
            args,
            task.target_dictionary,
            embed_tokens,
            no_encoder_attn=True,
        )
        return cls(decoder)

    @staticmethod
    def add_args(parser):
        TransformerLanguageModel.add_args(parser)

    @classmethod
    def build_embedding(cls, args, dictionary, embed_dim, path=None):
        def _vocab_init(tensor, **kwargs):
            nn.init.normal_(tensor, mean=0, std=embed_dim**-0.5)
            nn.init.constant_(tensor[1], 0)

        def _vocab_init_megatron(tensor, **kwargs):
            nn.init.normal_(
                tensor, mean=0, std=getattr(args, "megatron_init_sigma", 0.006)
            )
            nn.init.constant_(tensor[1], 0)

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
        )
        return embed_tokens


def base_lm_architecture(args):
    args.activation_fn = getattr(args, "activation_fn", "relu")
    args.dropout = getattr(args, "dropout", 0.1)
    args.attention_dropout = getattr(args, "attention_dropout", 0.0)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 512)
    args.decoder_output_dim = getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 2048)
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    # Model training is not stable without this
    args.decoder_normalize_before = True
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
