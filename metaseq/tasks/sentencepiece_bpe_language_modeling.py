# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Streaming Language Modeling task that loads corpora in src-tgt format and performs
on-the-fly tokenization.
"""
import logging
import numpy as np
from dataclasses import dataclass, field
from omegaconf import open_dict

from metaseq.tasks.streaming_language_modeling import StreamingLanguageModelingConfig
from metaseq.tasks.streaming_language_modeling import StreamingLanguageModelingTask
from metaseq.tasks import register_task
from metaseq.dataclass import MetaseqDataclass


from metaseq.data.encoders.sentencepiece_bpe import Tokenizer

from metaseq.tasks.base_task import BaseTask, BaseDataset

logger = logging.getLogger(__name__)

from metaseq.data import (
    AppendTokenDataset,
    IdDataset,
    NestedDictionaryDataset,
    NumelDataset,
    PadDataset,
    PrependTokenDataset,
    StripTokenDataset,
    TokenBlockDataset,
    Dictionary,
)

from sentencepiece import SentencePieceProcessor


TOK_TO_METASEQ_MAPPING = {
    0: 3,  # unk,
    1: 0,  # bos,
    2: 2,  # eos,
    -1: 1,  # pad
}


METASEQ_TO_TOK_MAPPING = {3: 0, 0: 1, 2: 2, 1: 0}


@dataclass
class SentencepieceBpeTaskConfig(StreamingLanguageModelingConfig):
    sentencepiece_model_path: str = field(
        default="", metadata={"help": "path to tokenizer"}
    )


@register_task("sentencepiece_bpe_task", dataclass=SentencepieceBpeTaskConfig)
class SentencepieceBpeTask(StreamingLanguageModelingTask):
    dictionary: Dictionary

    def __init__(self, cfg: MetaseqDataclass, **kwargs):
        super().__init__(cfg, **kwargs)

        # reset special tokens  
        self.dictionary.pad_index = 0
        self.dictionary.bos_index = 1
        self.dictionary.eos_index = 2
        self.dictionary.unk_index = 3
        self.dictionary.indices[self.dictionary.pad_word] = 0
        self.dictionary.indices[self.dictionary.bos_word] = 1
        self.dictionary.indices[self.dictionary.eos_word] = 2
        self.dictionary.indices[self.dictionary.unk_word] = 3

    def _init_tokenizer(self, args):
        return Tokenizer(args)

    def build_tokenizer(self, args):
        """Build the pre-tokenizer for this task."""
        return None

    def build_model(self, cfg: MetaseqDataclass):
        with open_dict(cfg):
            cfg._name = "llama_transformer_lm_megatron"
            cfg.arch = "llama_transformer_lm_megatron"
        if not getattr(cfg, "in_training", False):
            cfg.memory_efficient_fp16 = True
        return super().build_model(cfg)

    def build_dataset_for_inference(self, src_tokens, src_lengths, **kwargs):
        """
        Generate batches for inference. We prepend an eos token to src_tokens
        (or bos if `--add-bos-token` is set) and we append a <pad> to target.
        This is convenient both for generation with a prefix and LM scoring.
        """
        dataset = StripTokenDataset(
            TokenBlockDataset(
                src_tokens,
                src_lengths,
                block_size=None,  # ignored for "eos" break mode
                pad=self.source_dictionary.pad(),
                eos=self.source_dictionary.eos(),
                break_mode="eos",
            ),
            # remove eos from (end of) target sequence
            self.source_dictionary.eos(),
        )
        src_dataset = PrependTokenDataset(
            dataset,
            token=(self.tokenizer.bos_id),
        )
        tgt_dataset = AppendTokenDataset(dataset, token=self.source_dictionary.pad())
        return NestedDictionaryDataset(
            {
                "id": IdDataset(),
                "net_input": {
                    "src_tokens": PadDataset(
                        src_dataset,
                        pad_idx=self.source_dictionary.pad(),
                        left_pad=False,
                    ),
                    "src_lengths": NumelDataset(src_dataset, reduce=False),
                },
                "target": PadDataset(
                    tgt_dataset, pad_idx=self.source_dictionary.pad(), left_pad=False
                ),
            },
            sizes=[np.array(src_lengths)],
        )