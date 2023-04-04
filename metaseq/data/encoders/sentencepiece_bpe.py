# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from dataclasses import dataclass, field
import logging
import os
from typing import List, Union
from collections import namedtuple

from metaseq.data.encoders import register_bpe
from metaseq.dataclass import MetaseqDataclass


logger = logging.getLogger(__name__)


from sentencepiece import SentencePieceProcessor

TOK_TO_METASEQ_MAPPING = {
    0: 3,  # unk,
    1: 0,  # bos,
    2: 2,  # eos
    0: 1,  # pad
}

METASEQ_TO_TOK_MAPPING = {3: 0, 0: 1, 2: 2, 1: 0}


@dataclass
class TokenizerConfig(MetaseqDataclass):
    sentencepiece_model_path: str = field(
        default="", metadata={"help": "path to tokenizer"}
    )


@register_bpe("sentencepiece_bpe", dataclass=TokenizerConfig)
class BPETokenizer(object):
    def __init__(self, cfg) -> None:
        self.bpe = Tokenizer(cfg)


class Tokenizer(object):
    def __init__(self, cfg):

        model_path = cfg.sentencepiece_model_path
        assert os.path.isfile(model_path), model_path
        self.sp_model = SentencePieceProcessor(model_file=model_path)
        logger.info(f"Loaded SentencePiece model from {model_path}")

        # BOS / EOS token IDs
        self.n_words: int = self.sp_model.vocab_size()
        self.bos_id: int = self.sp_model.bos_id()
        self.eos_id: int = self.sp_model.eos_id()
        self.pad_id: int = 0  # set manually, as default is -1
        assert self.sp_model.vocab_size() == self.sp_model.get_piece_size()

    def encode(self, s: str):
        assert type(s) is str
        t = self.sp_model.encode(s)
        t_str = [str(tok) for tok in t]
        out = namedtuple("out", ["ids", "offsets"])
        return out(ids=t, offsets=zip(t, t_str))

    def encode_str(self, s: str) -> List[str]:
        assert type(s) is str
        return self.sp_model.encode(s, out_type=str)

    def decode(self, x: Union[str, List[int]]) -> str:
        toks = x
        if isinstance(x, str):
            toks = [
                int(tok) if tok not in {"<unk>", "<mask>"} else tok for tok in x.split()
            ]
        toks = [tok if tok != -1 else 0 for tok in toks]
        return self.sp_model.decode(toks)

    def is_beginning_of_word(self, x: str) -> bool:
        return self.decode(x).startswith(" ")

    def token_to_id(self, x: str) -> int:
        tok_id = self.sp_model.piece_to_id(x)
        return TOK_TO_METASEQ_MAPPING.get(tok_id, tok_id)

    def get_vocab_size(self) -> int:
        return self.sp_model.vocab_size()

    def id_to_token(self, x: int) -> str:
        if x == 1:
            return "<pad>"
        return self.sp_model.id_to_piece(METASEQ_TO_TOK_MAPPING.get(x, x))