# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional
from dataclasses import dataclass, field

from metaseq import file_utils
from metaseq.data.encoders import register_bpe
from metaseq.dataclass import MetaseqDataclass


@dataclass
class HuggingFaceByteLevelBPEConfig(MetaseqDataclass):
    bpe_merges: str = field(default="???", metadata={"help": "path to merges.txt"})
    bpe_vocab: str = field(default="???", metadata={"help": "path to vocab.json"})
    bpe_add_prefix_space: bool = field(
        default=False, metadata={"help": "add prefix space before encoding"}
    )
    hf_tokenizer: Optional[str] = field(
        default=None, metadata={"help": "path to tokenizer file."}
    )


@register_bpe("hf_byte_bpe", dataclass=HuggingFaceByteLevelBPEConfig)
class HuggingFaceByteLevelBPE(object):
    def __init__(self, cfg):
        try:
            from tokenizers import ByteLevelBPETokenizer, Tokenizer
        except ImportError:
            raise ImportError(
                "Please install huggingface/tokenizers with: " "pip install tokenizers"
            )

        if cfg.hf_tokenizer:
            self.bpe = Tokenizer.from_file(cfg.hf_tokenizer)
        else:
            bpe_vocab = file_utils.cached_path(cfg.bpe_vocab)
            bpe_merges = file_utils.cached_path(cfg.bpe_merges)

            self.bpe = ByteLevelBPETokenizer(
                bpe_vocab,
                bpe_merges,
                add_prefix_space=cfg.bpe_add_prefix_space,
            )

    def encode(self, x: str) -> str:
        return " ".join(map(str, self.bpe.encode(x).ids))

    def decode(self, x: str) -> str:
        return self.bpe.decode(
            [int(tok) if tok not in {"<unk>", "<mask>"} else tok for tok in x.split()]
        )

    def is_beginning_of_word(self, x: str) -> bool:
        return self.decode(x).startswith(" ")
