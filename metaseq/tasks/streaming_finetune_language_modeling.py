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
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

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


@dataclass
class StreamingFinetuneLanguageModelingConfig(StreamingLanguageModelingConfig):
    valid_sample_break_mode: Optional[str] = field(
        default="none",
        metadata={"help": "control break model specific to valid splits"},
    )
    report_valid_accuracy: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Read negative examples while reading data for evaluations"
            "This is useful to calculate validation accuracy during training"
        },
    )
    left_truncation: Optional[bool] = field(
        default=False,
        metadata={
            "help": "If an example is more than the block size decide on truncation type"
            "if left_truncation is true, truncation is on left size otherwise on right side."
        }
    )


@register_task(
    "streaming_finetune_language_modeling", dataclass=StreamingFinetuneLanguageModelingConfig
)
class StreamingFinetuneLanguageModelingTask(StreamingLanguageModelingTask):
    """
    Fine-tune a language model on a stream of data with a source/input and a target/output
    """

    def _tokenize_src_tgt_withcands_json(self, json):
        assert "candidates" in json and isinstance(json["candidates"], list)
        src = json["src"].rstrip(" ")
        src_tokens_len = len(self.tokenizer.encode(src).ids)
        tgt = json["tgt"].rstrip()
        pos = None
        neg = []
        cands = json["candidates"]
        if len(cands) == 0:
            cands = [tgt]
        for cand in cands:
            cand = cand.rstrip()
            full_tokens = torch.LongTensor(
                self.tokenizer.encode(" ".join([src, cand])).ids + [self.eod]
            )
            cand_tokens = torch.clone(full_tokens)
            cand_tokens[:src_tokens_len] = self.dictionary.pad_index
            if cand == tgt:
                pos = (full_tokens, cand_tokens)
            else:
                neg.append((full_tokens, cand_tokens))
        assert pos is not None
        return zip(*([pos] + neg))

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
                    tokenizer=self._tokenize_src_tgt_withcands_json
                    if (split != "train" and self.args.report_valid_accuracy)
                    else self._tokenize_src_tgt_json,
                )
            )
            corpora.append(os.path.splitext(file)[0])
        assert len(datasets) > 0

        if (
            self.args.multicorpus_sampling_alpha != 1
            or self.args.multicorpus_sampling_maximum > 0
        ) and (split == "train"):
            datasets = self._alpha_sampling(datasets, corpora, epoch)

        dataset = torch.utils.data.ConcatDataset(datasets)

        # shuffle order across epochs
        dataset = StreamingShuffleDataset(dataset, seed=self.args.seed)

        logger.info(
            f"Enabling {'left' if self.args.left_truncation else 'right'} truncation in the blocks of {split} split"
        )
        self.datasets[split] = StreamingSrcTgtDataset(
            dataset,
            # We generate blocks with one extra token, so that we have a target
            # for the final input token. This results in slight data loss.
            block_size=self.args.tokens_per_sample + 1,
            break_mode=self.args.sample_break_mode
            if split == "train"
            else self.args.valid_sample_break_mode,  # use diferent mode for val/test
            # we drop the remainder block during training
            drop_last=(split == "train"),
            padding_idx=self.source_dictionary.pad(),
            left_truncation=self.args.left_truncation,
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
        collate_data = {
            "id": ids,
            "net_input": {
                "src_tokens": input,
            },
            "target": target,
            "nsentences": input.size(0),
            "ntokens": input.ne(self.dictionary.pad()).sum(),
            "ntokens_target": target.ne(self.dictionary.pad()).sum(),
        }

        if "is_positive" in items[0]:
            is_positive = torch.cat([x["is_positive"] for x in items if x is not None])
            collate_data.update({"is_positive": is_positive})
            num_cands = torch.cat([x["num_cands"] for x in items if x is not None])
            collate_data.update({"num_cands": num_cands})
            # update ntokens_target
            true_bsz = is_positive.size(0)
            collate_data.update(
                {
                    "ntokens_target": (
                        target[:true_bsz, :].ne(self.dictionary.pad())
                        * is_positive.unsqueeze(1).repeat(1, target.size(1))
                    ).sum()
                }
            )

        return collate_data
