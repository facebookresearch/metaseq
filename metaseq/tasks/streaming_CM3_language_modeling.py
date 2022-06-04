# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Streaming Language Modeling task that loads corpora in plaintext and performs
on-the-fly tokenization.
"""

import logging
import os
from dataclasses import dataclass, field
import torch

from metaseq.data import (
    Dictionary,
    CausalMaskedDataset,
    PartitionedStreamingDataset,
    StreamingShuffleDataset,
    JsonlDataset,
    iterators,
)
from metaseq.tasks.streaming_language_modeling import (
    StreamingLanguageModelingConfig,
    StreamingLanguageModelingTask,
)
from metaseq.tasks import register_task

try:
    from tokenizers import ByteLevelBPETokenizer

    has_hf_tokenizers = True
except ImportError:
    has_hf_tokenizers = False


logger = logging.getLogger(__name__)

IMAGE_PREFIX = "I"
SPEECH_PREFIX = "S"


@dataclass
class StreamingCM3LanguageModelingConfig(StreamingLanguageModelingConfig):
    image_tokens: int = field(
        default=8192,
        metadata={"help": "total number of vision tokens used"},
    )
    speech_tokens: int = field(
        default=512,
        metadata={"help": "total number of speech tokens used"},
    )
    num_sentinel_tokens: int = field(
        default=512,
        metadata={"help": "number of special sentinel tokens to add to the vocabulary"},
    )
    lambda_sentinel_tokens: int = field(
        default=1,
        metadata={"help": "poisson lambda for the cm3 objective"},
    )


@register_task(
    "streaming_CM3_language_modeling", dataclass=StreamingCM3LanguageModelingConfig
)
class StreamingCM3LanguageModelingTask(StreamingLanguageModelingTask):
    def __init__(self, args):
        self.args = args
        self.datasets = {}
        self.dataset_to_epoch_iter = {}

        if not has_hf_tokenizers:
            raise ImportError("Please install tokenizers with: pip install tokenizers")
        print(args.vocab_filename)
        print(args.merges_filename)
        tokenizer = ByteLevelBPETokenizer.from_file(
            args.vocab_filename, args.merges_filename
        )
        self.sentinel_tokens = [
            f"<sentinel:{i}>" for i in range(args.num_sentinel_tokens)
        ]
        self.sentinel_end = "<eoss>"

        tokenizer.add_special_tokens(
            [f"{IMAGE_PREFIX}{x} " for x in range(args.image_tokens)]
        )
        tokenizer.add_special_tokens(
            [f"{SPEECH_PREFIX}{x} " for x in range(args.speech_tokens)]
        )
        tokenizer.add_special_tokens(self.sentinel_tokens)
        tokenizer.add_special_tokens([self.sentinel_end])

        self.tokenizer = tokenizer

        if max(args.update_freq) > 1:
            raise NotImplementedError(
                "--update-freq is not compatible with StreamingLanguageModelingTask"
            )

        self.eod = self.tokenizer.token_to_id(args.end_of_document_symbol)
        if self.eod is None:
            # This will be executed for old models that do not have the args.end_of_document_symbol explicitly set
            # and do not use <s/> (the default) but <EOS>
            self.eod = self.tokenizer.token_to_id("<EOS>")

        assert (
            self.eod is not None
        ), "Cannot find end-of-document symbol ({}) in tokenizer".format(
            args.end_of_document_symbol
        )

        # construct a dummy metaseq Dictionary corresponding to the given tokenizer
        self.dictionary = Dictionary()
        tok_vocab_size = self.tokenizer.get_vocab_size()

        for id in range(self.dictionary.nspecial, tok_vocab_size):
            self.dictionary.add_symbol(self.tokenizer.id_to_token(id))
        final_vocab_size = args.final_vocab_size
        # final_vocab_size = 51200 for roberta dictionary
        if final_vocab_size is not None:
            if final_vocab_size < tok_vocab_size:
                raise ValueError(
                    f"incompatible: {final_vocab_size}, tok_vocab_size: {tok_vocab_size}"
                )
            self.dictionary.pad_to_multiple_(final_vocab_size)
        else:
            self.dictionary.pad_to_multiple_(8)

        # confirm that metaseq dictionary and BPE have matching special symbols
        assert self.dictionary.bos_index == 0
        assert self.tokenizer.id_to_token(0) in {"<BOS>", "<s>"}
        assert self.dictionary.pad_index == 1
        assert self.tokenizer.id_to_token(1) in {"<PAD>", "<pad>"}
        assert self.dictionary.eos_index == 2
        assert self.tokenizer.id_to_token(2) in {"<EOS>", "</s>"}
        assert self.dictionary.unk_index == 3
        assert self.tokenizer.id_to_token(3) in {"<UNK>", "<unk>"}

        self.criterion_weights = torch.ones(len(self.dictionary))
        self.sentinel_tokens_ind = []
        for token in self.sentinel_tokens:
            token_index = self.dictionary.index(token)
            assert token_index != self.dictionary.unk_index
            self.sentinel_tokens_ind.append(token_index)
            self.criterion_weights[token_index] = 0.0

        self.sentinel_end_ind = self.dictionary.index(self.sentinel_end)
        assert token_index != self.sentinel_end_ind

    def load_dataset(self, split: str, epoch=1, combine=False, **kwargs):
        # This function reads a bunch of jsonl files, concats them together,
        # shuffles them, then chunks them into blocks of tokens (e.g., 2048).

        # determine number of shards for this split
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
                    tokenizer=self._tokenize_one_json,
                )
            )
            corpora.append(os.path.splitext(file)[0])
        assert len(datasets) > 0

        if self.args.multicorpus_sampling_alpha != 1:
            datasets = self._alpha_sampling(datasets, corpora, epoch)

        dataset = torch.utils.data.ConcatDataset(datasets)

        # shuffle order across epochs
        dataset = StreamingShuffleDataset(dataset, seed=self.args.seed)

        # chunk into blocks of tokens
        self.datasets[split] = CausalMaskedDataset(
            self.args.lambda_sentinel_tokens,
            self.sentinel_tokens_ind,
            "poisson",
            self.args.tokens_per_sample + 1,
            self.sentinel_end_ind,
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
            seed=1284 + self.args.seed,
        )

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
        skip_remainder_batch=False,
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
        assert isinstance(dataset, CausalMaskedDataset)
        shuffle_buffer_size = 10 * max_sentences * num_shards
        logger.info(f"setting shuffle buffer size to {shuffle_buffer_size}")
        dataset.set_shuffle_buffer_size(shuffle_buffer_size)

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
        )
