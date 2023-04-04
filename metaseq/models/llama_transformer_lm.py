# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from dataclasses import dataclass, field
from fairscale.nn.model_parallel.layers import (
    ParallelEmbedding,
)
from omegaconf import II
import torch
import torch.nn as nn
from typing import Optional

from metaseq import utils
from metaseq.dataclass import ChoiceEnum
from metaseq.models.transformer_lm import (
    TransformerLanguageModelConfig,
    base_lm_architecture,
    has_megatron_submodule,
    DEFAULT_MAX_TARGET_POSITIONS,
)
from metaseq.models import (
    BaseModel,
    register_model,
)
from metaseq.modules.embedding import Embedding

from metaseq.models.llama_transformer_decoder import (
    LlamaModelParallelTransformerDecoder,
    LlamaTransformerDecoder,
)


LNORM_TYPES = ChoiceEnum(["layernorm", "rmsnorm"])
LSCALE_TYPES = ChoiceEnum(["disabled", "current", "global", "one"])


@dataclass
class LlamaTransformerLanguageModelConfig(TransformerLanguageModelConfig):
    separate_qkv_proj: Optional[bool] = field(
        default=False, metadata={"help": "set True to separate qkv proj"}
    )
    use_rope: Optional[bool] = field(
        default=False, metadata={"help": "whether to use rope embeddings"}
    )
    layernorm_type: Optional[LNORM_TYPES] = field(  # noqa
        default="layernorm", metadata={"help": "which layer norm to use"}
    )
    layer_scale: Optional[LSCALE_TYPES] = field(  # noqa
        default="disabled", metadata={"help": "what type of LayerScale to use, if any"}
    )
    llama_init_order: Optional[bool] = field(
        default=False,
        metadata={
            "help": "initialize parameters in llama order, to load FSDP shards correctly"
        },
    )
    arch: str = II("common.arch")
    criterion: str = II("criterion.criterion")
    bf16: bool = II("common.bf16")
    use_sharded_state: Optional[bool] = II("distributed.use_sharded_state")


@register_model("llama_transformer_lm", dataclass=LlamaTransformerLanguageModelConfig)
class LlamaTransformerLanguageModel(BaseModel):
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
        decoder = LlamaTransformerDecoder(
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


@register_model(
    "llama_transformer_lm_megatron", dataclass=LlamaTransformerLanguageModelConfig
)
class LlamaModelParallelTransformerLanguageModel(LlamaTransformerLanguageModel):
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

        if getattr(args, "max_target_positions", None) is None:
            args.max_target_positions = getattr(
                args, "tokens_per_sample", DEFAULT_MAX_TARGET_POSITIONS
            )

        embed_tokens = cls.build_embedding(
            args, task.source_dictionary, args.decoder_embed_dim
        )
        print(args)
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

        decoder = LlamaModelParallelTransformerDecoder(
            args,
            task.target_dictionary,
            embed_tokens,
        )
        return cls(decoder)

    @staticmethod
    def add_args(parser):
        LlamaTransformerLanguageModel.add_args(parser)

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

        embed_tokens = ParallelEmbedding(
            len(dictionary),
            embed_dim,
            dictionary.pad(),
            init_method=_vocab_init_megatron
            if getattr(args, "full_megatron_init", False)
            else _vocab_init,
        )
        embed_tokens = embed_tokens.to(dtype)
        if torch.cuda.is_available():
            embed_tokens = embed_tokens.cuda()
        return embed_tokens