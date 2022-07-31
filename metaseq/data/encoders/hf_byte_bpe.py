# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field
from typing import List

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


@dataclass
class HuggingFaceCM3UnigramConfig(MetaseqDataclass):
    spm_path: str = field(default="???", metadata={"help": "path to spm_path"})


@register_bpe("hf_byte_bpe", dataclass=HuggingFaceByteLevelBPEConfig)
class HuggingFaceByteLevelBPE(object):
    def __init__(self, cfg):
        try:
            from tokenizers import ByteLevelBPETokenizer
        except ImportError:
            raise ImportError(
                "Please install huggingface/tokenizers with: " "pip install tokenizers"
            )

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


@register_bpe("hf_cm3_unigram", dataclass=HuggingFaceCM3UnigramConfig)
class HuggingFaceCM3Unigram(object):
    def __init__(self, cfg):
        try:
            from tokenizers import (
                Tokenizer,
                decoders,
                models,
                normalizers,
                pre_tokenizers,
            )
            from tokenizers.pre_tokenizers import ByteLevel, Digits
        except ImportError:
            raise ImportError(
                "Please install huggingface/tokenizers with: " "pip install tokenizers"
            )

        spm_path = file_utils.cached_path(cfg.spm_path)
        tokenizer = Tokenizer(models.Unigram()).from_file(spm_path)
        tokenizer.normalizer = normalizers.NFKC()
        tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
            [ByteLevel(), Digits(individual_digits=True)]
        )
        tokenizer.decoder = decoders.ByteLevel()
        self.bpe = tokenizer

    def encode(self, x: str) -> str:
        x = " ".join(map(str, self.encode_raw(x)))
        print(x)
        return x

    def encode_raw(self, x: str) -> List[int]:
        return self.bpe.encode(x).ids

    def decode(self, x: str) -> str:
        print(x)
        return self.bpe.decode(
            [int(tok) if tok not in {"<unk>", "<mask>"} else tok for tok in x.split()],
            skip_special_tokens=False,
        )

    def is_beginning_of_word(self, x: str) -> bool:
        return self.decode(x).startswith(" ")
