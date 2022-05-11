# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Streaming Language Modeling task that loads corpora in src-tgt format and performs
on-the-fly tokenization.
"""

import logging
import os
from typing import Any, Dict, List

import torch

from metaseq.data import (
    JsonlDataset,
    StreamingShuffleDataset,
    StreamingSrcTgtDataset,
    data_utils,
)
from metaseq.tasks.streaming_language_modeling import (
    StreamingLanguageModelingTask,
    StreamingLanguageModelingConfig,
)
from metaseq.tasks import register_task

logger = logging.getLogger(__name__)


@register_task(
    "streaming_finetune_language_modeling", dataclass=StreamingLanguageModelingConfig
)
class StreamingFinetuneLanguageModelingTask(StreamingLanguageModelingTask):
    def _tokenize_src_tgt_json(self, json):
        src = json["src"].rstrip(" ")
        tgt = json["tgt"].rstrip()
        full_tokens = torch.LongTensor(
            self.tokenizer.encode(" ".join([src, tgt])).ids + [self.eod]
        )
        src_tokens_len = len(self.tokenizer.encode(src).ids)
        tgt_tokens = torch.clone(full_tokens)
        tgt_tokens[:src_tokens_len] = self.dictionary.pad_index
        return (full_tokens, tgt_tokens)

    def load_dataset(self, split: str, epoch=1, combine=False, **kwargs):
        """Load a given dataset split.

        The folder structure is assumed to look like:

            /path/to/data/train/00/foo.jsonl
            /path/to/data/train/00/bar.jsonl
            /path/to/data/train/01/foo.jsonl
            /path/to/data/train/01/bar.jsonl
            /path/to/data/valid/00/foo.jsonl
            /path/to/data/valid/00/bar.jsonl

        In this example, we have two "shards" of training data, which will be
        iterated over in epochs 1 and 2, respectively. Subsequent epochs will
        cycle back over the same data. We also have two different data sources
        in each shard (foo and bar), which will be combined and shuffled.

        Each jsonl entry is a dict with "src" and "tgt" keys. Loss is computed
        only on the tgt tokens.

        Args:
            split (str): name of the split (e.g., train, valid, valid1, test)
        """
        # This function reads a bunch of jsonl files, concats them together,
        # shuffles them, then chunks them into blocks of tokens (e.g., 2048).

        # determine number of shards for this split shards = {}
        cur_shard_str = self.get_shard_str(epoch, split)

        # concatenate any jsonl files that are part of the shard
        datasets, corpora = [], []
        for file in sorted(
            os.listdir(os.path.join(self.args.data, split, cur_shard_str))
        ):
            if not file.endswith(".jsonl"):
                continue
            datasets.append(
                JsonlDataset(
                    path=os.path.join(self.args.data, split, cur_shard_str, file),
                    tokenizer=self._tokenize_src_tgt_json,
                )
            )
            corpora.append(os.path.splitext(file)[0])
        assert len(datasets) > 0

        if self.args.multicorpus_sampling_alpha != 1:
            raise NotImplementedError

        dataset = torch.utils.data.ConcatDataset(datasets)

        # shuffle order across epochs
        dataset = StreamingShuffleDataset(dataset, seed=self.args.seed)

        self.datasets[split] = StreamingSrcTgtDataset(
            dataset,
            # We generate blocks with one extra token, so that we have a target
            # for the final input token. This results in slight data loss.
            block_size=self.args.tokens_per_sample + 1,
            break_mode=self.args.sample_break_mode,
            # we drop the remainder block during training
            drop_last=(split == "train"),
            padding_idx=self.source_dictionary.pad(),
            # 1284 is a randomly-generated offset to decouple the seed used here
            # from the seed used above in StreamingShuffleDataset
            # TODO: Track this seed to avoid collisions. See issue #65
            seed=1284 + self.args.seed,
        )

    def _collate_fn(self, items: List[Dict[str, Any]]):
        # StreamingTokenBlockDataset returns None as filler
        if len([x for x in items if x is not None]) == 0:
            return {}

        src_tokens = data_utils.collate_tokens(
            [x["src_block"] for x in items if x is not None],
            pad_idx=self.source_dictionary.pad(),
            pad_to_bsz=self.args.batch_size,
        )
        tgt_tokens = data_utils.collate_tokens(
            [x["tgt_block"] for x in items if x is not None],
            pad_idx=self.source_dictionary.pad(),
            pad_to_bsz=self.args.batch_size,
        )

        # generate inputs and targets
        input = src_tokens[:, :-1].contiguous()
        target = tgt_tokens[:, 1:].contiguous()

        ids = torch.cat([x["ids"] for x in items if x is not None])
        if ids.numel() != torch.unique(ids).numel():
            n_duplicate = ids.numel() - torch.unique(ids).numel()
            logger.error(
                f"found {n_duplicate}/{ids.numel()} duplicate document IDs in the same batch!"
            )

        # metaseq expects batches to have the following structure
        return {
            "id": ids,
            "net_input": {
                "src_tokens": input,
            },
            "target": target,
            "nsentences": input.size(0),
            "ntokens": input.ne(self.dictionary.pad()).sum(),
            "ntokens_target": target.ne(self.dictionary.pad()).sum(),
        }
