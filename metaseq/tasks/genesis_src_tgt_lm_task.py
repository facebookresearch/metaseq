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
from omegaconf import II, open_dict

from metaseq.data import (
    JsonlDataset,
    PartitionedStreamingDataset,
    data_utils,
    StreamingSrcTgtDataset,
    StreamingShuffleDataset,
    iterators,
)
from metaseq.dataclass import MetaseqDataclass
from metaseq.tasks.sentencepiece_bpe_language_modeling import (
    SentencepieceBpeTask,
)
from metaseq.tasks.streaming_language_modeling import (
    DocumentToSequenceDataset,
    StreamingLanguageModelingConfig,
)

from metaseq.tasks import register_task

logger = logging.getLogger(__name__)


@dataclass
class GensisSrcTgtLmConfig(StreamingLanguageModelingConfig):
    sentencepiece_model_path: str = field(
        default="", metadata={"help": "path to tokenizer"}
    )
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
        },
    )


@register_task("genesis_src_tgt_lm", dataclass=GensisSrcTgtLmConfig)
class GenesisSrcTgtLmTask(SentencepieceBpeTask):
    def __init__(self, args) -> None:
        super().__init__(args)
        self.bod = self.dictionary.bos_index
        self.eod = self.dictionary.eos_index

    def build_model(self, cfg: MetaseqDataclass):
        with open_dict(cfg):
            cfg.in_training = True
        return super().build_model(cfg)

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
                [self.bod]
                + self.tokenizer.encode(" ".join([src, cand])).ids
                + [self.eod]
            )
            cand_tokens = torch.clone(full_tokens)
            cand_tokens[
                : src_tokens_len + 1
            ] = (
                self.dictionary.pad_index
            )  # TODO: check the +1 offset including bod is correct
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
            [self.bod] + self.tokenizer.encode(" ".join([src, tgt])).ids + [self.eod]
        )
        src_tokens_len = len(self.tokenizer.encode(src).ids)
        tgt_tokens = torch.clone(full_tokens)
        tgt_tokens[
            : src_tokens_len + 1
        ] = (
            self.dictionary.pad_index
        )  # TODO: check the +1 offset including bod is correct
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

        logger.info(
            f"Enabling {'left' if self.args.left_truncation else 'right'} truncation in the blocks of {split} split"
        )

        # shuffle order across epochs
        dataset = StreamingShuffleDataset(dataset, seed=self.args.seed)
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
            seed=self.args.seed,
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

    def get_batch_iterator(
            self,
            dataset,
            max_tokens=None,
            max_sentences=None,
            max_positions=None,
            ignore_invalid_inputs=False,
            required_batch_size_multiple=1,
            seed=1,
            num_shards=1,
            shard_id=0,
            num_workers=0,
            epoch=1,
            data_buffer_size=0,
            disable_iterator_cache=False,
            batch_by_size=True,
            skip_remainder_batch=True,
        ):
            """
            Get an iterator that yields batches of data from the given dataset.

            Args:
                dataset (torch.utils.data.Dataset): dataset to batch
                max_sentences (int, optional): max number of sentences in each
                    batch (default: None).
                num_shards (int, optional): shard the data iterator into N
                    shards (default: 1).
                shard_id (int, optional): which shard of the data iterator to
                    return (default: 0).
                num_workers (int, optional): how many subprocesses to use for data
                    loading. 0 means the data will be loaded in the main process
                    (default: 0).
                epoch (int, optional): the epoch to start the iterator from
                    (default: 1).
                data_buffer_size (int, optional): number of batches to
                    preload (default: 0).
                disable_iterator_cache (bool, optional): don't cache the
                    EpochBatchIterator
                    (default: False).
                batch_by_size (bool, optional):
                    batch sequences of similar length together to reduce padding.
                    If false, each batch will be of size max_sentences.
                skip_remainder_batch (bool, optional): if set, discard the last
                    batch in each training epoch, as the last batch is often smaller
                    than local_batch_size * distributed_word_size (default: ``True``).
            Returns:
                ~metaseq.iterators.EpochBatchIterator: a batched iterator over the
                    given dataset split
            """
            assert max_tokens is None

            # Up to this point, we have shuffled documents, flattened them into a 1D
            # tensor, then chunked into token blocks. But if documents are long, then
            # adjacent blocks may be from a single document, and naively distributed
            # sequential blocks to GPUs may cause entire updates to be dominated by a
            # handful of unique documents. Instead we have a readahead buffer that
            # reads in 10 full batches of data and shuffles sequences across them,
            # thus increasing randomness. This assumes that no single document spans
            # 10 full batches, which is reasonable when batch sizes are in the
            # millions and documents are on average much smaller.
            assert isinstance(dataset, DocumentToSequenceDataset) or isinstance(
                dataset, StreamingSrcTgtDataset
            )
            shuffle_buffer_size = 10 * max_sentences * num_shards
            logger.info(f"setting shuffle buffer size to {shuffle_buffer_size}")
            dataset.set_shuffle_buffer_size(shuffle_buffer_size)
            # dataset.set_num_workers(num_workers)

            # partition dataset across data parallel workers
            dataset = PartitionedStreamingDataset(
                dataset,
                num_shards=num_shards,
                shard_id=shard_id,
                drop_last=skip_remainder_batch,
            )

            # create a stateful/checkpointable iterator for the current data
            # parallel worker
            return iterators.StreamingEpochBatchIterator(
                dataset=dataset,
                batch_size=max_sentences,
                collate_fn=self._collate_fn,
                drop_last=skip_remainder_batch,
                num_workers=num_workers,
                epoch=epoch,
                num_shards=num_shards,
            )