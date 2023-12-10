# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Streaming Language Modeling task that loads corpora in plaintext and performs
on-the-fly tokenization.
"""

import logging
import random
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from omegaconf import II

from metaseq.data import (
    Dictionary,
    JsonlDataset,
    PartitionedStreamingDataset,
    ResamplingDataset,
    StreamingSrcTgtDataset,
    data_utils,
    iterators,
)

from metaseq.dataclass import MetaseqDataclass
from metaseq.tasks import LegacyTask, register_task
from metaseq.data.document_to_sequence import DocumentToSequenceDataset
from metaseq.data.cm3_dataset import CausalMaskedDocumentToSequenceDataset
from metaseq.dataclass import ChoiceEnum

try:
    from tokenizers import ByteLevelBPETokenizer, Tokenizer

    has_hf_tokenizers = True
except ImportError:
    has_hf_tokenizers = False


logger = logging.getLogger(__name__)

DEFAULT_MULTICORPUS_MAX = -1

LANGUAGE_MODELING_MODE = ChoiceEnum(["standard", "cm3", "racm3"])
CM3_MODE = ChoiceEnum(["poisson", "fixed", "fim"])


def map_old_image_token_to_new_image_token(text):
    text = text.replace("I", "IMGIMG")
    for i in range(10):
        text = text.replace(str(i), chr(ord("A") + i))
    return text.replace(" ", "Z")


def map_new_image_token_to_old_image_token(text):
    text = text.replace("Z", " ")
    for i in range(10):
        text = text.replace(chr(ord("A") + i), str(i))
    return text.replace("IMGIMG", "I")


def parse_doc(doc):
    obj = re.match(r'<img alt="(.*?)" src="(I\d.*)">', doc)
    if obj is None:
        raise ValueError(f"doc not correct formated: {doc}")
    text, image = obj.group(1), obj.group(2)
    result = {"text": text, "image": image}
    return result


@dataclass
class StreamingLanguageModelingConfig(MetaseqDataclass):
    data: Optional[str] = field(
        default=None, metadata={"help": "path to data directory with JSONL files"}
    )
    hf_tokenizer: Optional[str] = field(
        default="", metadata={"help": "path to a HF tokenizer json file."}
    )
    vocab_filename: Optional[str] = field(
        default="", metadata={"help": "path to bpe-vocab.json"}
    )
    merges_filename: Optional[str] = field(
        default="", metadata={"help": "path to bpe-merges.txt"}
    )
    end_of_document_symbol: Optional[str] = field(
        default="</s>", metadata={"help": "symbol indicating an end-of-document"}
    )
    sample_break_mode: Optional[str] = field(
        default="none",
        metadata={
            "help": 'If omitted or "none", fills each sample with tokens-per-sample '
            'tokens. If set to "complete", splits samples only at the end '
            "of sentence, but may include multiple sentences per sample. "
            '"complete_doc" is similar but respects doc boundaries. '
            'If set to "eos", includes only one sentence per sample.'
        },
    )
    tokens_per_sample: int = field(
        default=1024,
        metadata={"help": "max number of tokens per sample for LM dataset"},
    )
    max_source_positions: Optional[int] = field(
        default=None, metadata={"help": "max number of tokens in the source sequence"}
    )
    max_target_positions: Optional[int] = field(
        default=None, metadata={"help": "max number of tokens in the target sequence"}
    )
    final_vocab_size: Optional[int] = field(
        default=None, metadata={"help": "force vocab size to this"}
    )
    multicorpus_sampling_alpha: Optional[float] = field(
        default=1.0,
        metadata={
            "help": "smoothing alpha for sample rations across multiple datasets"
        },
    )
    multicorpus_sampling_maximum: Optional[float] = field(
        default=DEFAULT_MULTICORPUS_MAX,
        metadata={"help": "Maximum size for example proportional sampling"},
    )
    data_subshard_count: int = field(
        default=1,
        metadata={
            "help": "Number of data subshards to use while training."
            "Subsharding allows us to virtually split the dataset to speed up dataset fast forwarding."
        },
    )
    # language modeling type
    language_modeling_type: LANGUAGE_MODELING_MODE = field(
        default="standard",
        metadata={
            "help": "Number of data subshards to use while training."
            "Subsharding allows us to virtually split the dataset to speed up dataset fast forwarding."
        },
    )
    # CM3 Specific Parameters
    cm3_num_sentinel_tokens: int = field(
        default=512,
        metadata={"help": "Number of special sentinel tokens to add to the vocabulary"},
    )
    cm3_lambda_sentinel_tokens: int = field(
        default=1,
        metadata={
            "help": "if CM3_MODE is `poisson` then the Poisson Lambda for the cm3 objective."
            "if CM3_MODE is `fixed` then represents the number of fixed masks per example to use."
            "if CM3_MODE is `fim` then will be forced to be 1."
        },
    )
    cm3_mode: CM3_MODE = field(
        default="poisson",
        metadata={
            "help": "The type of infilling objective to do; poisson (original CM3),"
            "fixed (CM3 with fixed number of masks), fim (CM3 with 1 mask)."
        },
    )
    cm3_allow_across_eod_boundaries: bool = field(
        default=False,
        metadata={
            "help": "Whether or not we allow rotation of documents across documents"
            "(especially when training with token blocking set to None)."
            "By default the original CM3 objective allows rotation across document boundaries."
            "For FIM it's unclear whether or not they allow this."
        },
    )
    cm3_percent_full_document_rotation: float = field(
        default=0.0,
        metadata={
            "help": "What percent of the time to rotate full documents while still abiding by the number of sentinel tokens used."
        },
    )
    num_retrieved_doc: int = field(
        default=2, metadata={"help": "number of retrieved documents"}
    )
    # TODO common vars below add to parent
    seed: int = II("common.seed")
    batch_size: Optional[int] = II("dataset.batch_size")
    batch_size_valid: Optional[int] = II("dataset.batch_size_valid")
    data_buffer_size: int = II("dataset.data_buffer_size")
    update_freq: List[int] = II("optimization.update_freq")


@register_task("streaming_language_modeling", dataclass=StreamingLanguageModelingConfig)
class StreamingLanguageModelingTask(LegacyTask):
    """
    Train a language model on a stream of data. Currently we assume the stream
    is in JSONL format and we tokenize inputs on-the-fly.

    Note that we append an end-of-document symbol to the end of each document.

    Args:
        tokenizer (tokenizers.ByteLevelBPETokenizer): the BPE tokenizer to use
    """

    def __init__(self, args):
        super().__init__(args)

        if not has_hf_tokenizers:
            raise ImportError("Please install tokenizers with: pip install tokenizers")

        if args.hf_tokenizer:
            self.tokenizer = Tokenizer.from_file(args.hf_tokenizer)
        else:
            self.tokenizer = ByteLevelBPETokenizer.from_file(
                args.vocab_filename, args.merges_filename
            )

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

        # confirm that metaseq dictionary and BPE have matching special symbols
        assert self.dictionary.bos_index == 0
        assert self.tokenizer.id_to_token(0) in {"<BOS>", "<s>"}
        assert self.dictionary.pad_index == 1
        assert self.tokenizer.id_to_token(1) in {"<PAD>", "<pad>"}
        assert self.dictionary.eos_index == 2
        assert self.tokenizer.id_to_token(2) in {"<EOS>", "</s>"}
        assert self.dictionary.unk_index == 3
        assert self.tokenizer.id_to_token(3) in {"<UNK>", "<unk>"}

        self.has_cm3 = args.language_modeling_type in ["cm3", "racm3"]
        self.has_retrieval = args.language_modeling_type == "racm3"
        if self.has_cm3:
            self._check_cm3_parameterization()
            self._create_cm3_special_tokens()
            self.cm3_sentinel_type = self.args.cm3_mode

        final_vocab_size = args.final_vocab_size
        if final_vocab_size is not None:
            if final_vocab_size < tok_vocab_size:
                raise ValueError(
                    f"incompatible: {final_vocab_size}, tok_vocab_size: {tok_vocab_size}"
                )
            self.dictionary.pad_to_multiple_(final_vocab_size)
        else:
            self.dictionary.pad_to_multiple_(8)

    def _check_cm3_parameterization(self):
        assert (
            self.args.cm3_lambda_sentinel_tokens > 0
        ), "cm3_lambda_sentinel_tokens must be > 0"
        assert (
            self.args.cm3_num_sentinel_tokens > 0
        ), "cm3_num_sentinel_tokens must be > 0"
        assert (
            self.args.cm3_num_sentinel_tokens >= self.args.cm3_lambda_sentinel_tokens
        ), "cm3_lambda_sentinel_tokens must be > cm3_num_sentinel_tokens"
        if self.args.cm3_mode == "fim":
            assert (
                self.args.cm3_num_sentinel_tokens == 1
            ), "FIM requires cm3_num_sentinel_tokens to be 1"
            assert (
                self.args.cm3_lambda_sentinel_tokens == 1
            ), "FIM requires cm3_lambda_sentinel_tokens to be 1"
            self.cm3_sentinel_type = "fixed"

    def _create_cm3_special_tokens(self):
        self.cm3_sentinel_end = "<eoss>"
        self.cm3_break = "<racm3:break>"
        self.dictionary.add_symbol(self.cm3_break)
        self.dictionary.add_symbol(self.cm3_sentinel_end)
        # self.cm3_break_ind = self.dictionary.index(self.cm3_break)
        self.cm3_sentinel_tokens = [
            f"<sentinel:{i}>" for i in range(self.args.cm3_num_sentinel_tokens)
        ]
        self.cm3_sentinel_tokens_ind = []
        for token in self.cm3_sentinel_tokens:
            self.dictionary.add_symbol(token)
            token_index = self.dictionary.index(token)
            assert token_index != self.dictionary.unk_index
            self.cm3_sentinel_tokens_ind.append(token_index)
        self.cm3_sentinel_end_ind = self.dictionary.index(self.cm3_sentinel_end)
        self.cm3_break_ind = self.dictionary.index(self.cm3_break)

    @classmethod
    def setup_task(cls, args, **kwargs):
        return cls(args)

    def _tokenize_one_json(self, json):
        text = json["text"]
        return torch.LongTensor(
            # append an end-of-document symbol after each document
            self.tokenizer.encode(text.rstrip()).ids
            + [self.eod]
        )

    def tokenize_single_doc(self, doc, add_eod=False):
        doc = parse_doc(doc)
        text, image = doc["text"], doc["image"]
        image = map_old_image_token_to_new_image_token(image)
        text_indexes, image_indexes = (
            self.tokenizer.encode(text.rstrip()).ids,
            self.tokenizer.encode(image.rstrip()).ids,
        )
        assert (
            len(image_indexes) == 1024
        ), f"Each image must be 1024 tokens, instead we got {len(image_indexes)}"
        assert all(
            [i > 4 for i in image_indexes]
        ), f"Images should not have any special tokens: {image_indexes}"
        indexes = text_indexes + [self.cm3_break_ind] + image_indexes
        if add_eod:
            indexes = indexes + [self.eod]
        return indexes

    def _tokenize_ra_json(self, json):
        query_index = self.tokenize_single_doc(json["text"], add_eod=True)
        query_index = torch.LongTensor(query_index)
        ra_docs = json["retrieved_docs_from_img"] + json["retrieved_docs_from_txt"]
        random.shuffle(ra_docs)

        ra_docs = ra_docs[: self.args.num_retrieved_doc]
        ra_indexes = []
        for ra_doc in ra_docs:
            ra_index = self.tokenize_single_doc(ra_doc, add_eod=False)
            ra_index = torch.LongTensor(ra_index + [self.cm3_break_ind])
            ra_indexes.append(ra_index)
        final_indexes = torch.cat(
            [torch.LongTensor([self.eod])] + ra_indexes + [query_index]
        )
        return final_indexes

    def _get_sample_prob(self, dataset_lens):
        """
        Get smoothed sampling porbability by corpus. This helps small corpus by upsampling them.
        """
        if self.args.multicorpus_sampling_maximum == DEFAULT_MULTICORPUS_MAX:
            prob = dataset_lens / dataset_lens.sum()
            smoothed_prob = prob**self.args.multicorpus_sampling_alpha
            smoothed_prob = smoothed_prob / smoothed_prob.sum()
        else:
            dataset_lens = np.array(
                [min(l, self.args.multicorpus_sampling_maximum) for l in dataset_lens]
            )
            smoothed_prob = dataset_lens / sum(dataset_lens)
        return smoothed_prob

    def _alpha_sampling(self, datasets, corpora, epoch=1):
        """
        Up or down sample corpora with alpha sampling.
        """
        dataset_lengths = np.array(
            [len(d) for d in datasets],
            dtype=float,
        )
        logger.info(f"loaded total {dataset_lengths.sum()} blocks for all corpora")
        sample_probs = self._get_sample_prob(dataset_lengths)

        logger.info(
            "Sample probability by corpus: %s",
            {
                corpus: "{0:.4f}".format(sample_probs[id])
                for id, corpus in enumerate(corpora)
            },
        )
        size_ratio = (sample_probs * dataset_lengths.sum()) / dataset_lengths
        # TODO: add an option for shrinking all size ratios to below 1
        # if self.args.multicorpus_sampling_alpha != 1:
        #   size_ratio /= size_ratio.max()

        # Fix numeric errors in size ratio computation
        #   0.999999999999999999 -> 1
        #   1.000000000000000002 -> 1
        for i in range(len(size_ratio)):
            size_ratio[i] = round(size_ratio[i], 8)

        logger.info(
            "Up/Down Sampling ratio by corpus: %s",
            {
                corpus: "{0:.2f}".format(size_ratio[id])
                for id, corpus in enumerate(corpora)
            },
        )
        logger.info(
            "Actual dataset size by corpus: %s",
            {
                corpus: "{0:.2f}".format(len(datasets[id]))
                for id, corpus in enumerate(corpora)
            },
        )
        resampled_datasets = [
            ResamplingDataset(
                datasets[i],
                size_ratio=size_ratio[i],
                seed=self.args.seed,
                epoch=epoch,
                replace=size_ratio[i] > 1.0,
            )
            for i, d in enumerate(datasets)
        ]
        # TODO: estimate the actual steps or tokens seen in training before launching experiments.
        logger.info(
            "Resampled dataset size by corpus: %s",
            {
                corpus: "{0:.2f}".format(len(resampled_datasets[id]))
                for id, corpus in enumerate(corpora)
            },
        )
        return resampled_datasets

    def get_shard_str(self, epoch, split):
        shards = {}
        for shard_id in os.listdir(os.path.join(self.args.data, split)):
            assert (
                int(shard_id) not in shards
            ), f"shard id: {shard_id} not in shards: {shards}"
            shards[int(shard_id)] = shard_id
        assert min(shards.keys()) == 0
        assert max(shards.keys()) == len(shards) - 1

        data_subshard_count = self.args.data_subshard_count if split == "train" else 1

        shard_idx = ((epoch - 1) // data_subshard_count) % len(shards)
        cur_shard_str = shards[shard_idx]
        return cur_shard_str

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

        Args:
            split (str): name of the split (e.g., train, valid, valid1, test)
        """
        # This function reads a bunch of jsonl files, concats them together,
        # shuffles them, then chunks them into blocks of tokens (e.g., 2048).

        # determine number of shards for this split
        cur_shard_str = self.get_shard_str(epoch, split)

        # concatenate any jsonl files that are part of the shard
        datasets, corpora = [], []
        data_subshard_count = self.args.data_subshard_count if split == "train" else 1
        for file in sorted(
            os.listdir(os.path.join(self.args.data, split, cur_shard_str))
        ):
            if not file.endswith(".jsonl"):
                continue
            datasets.append(
                JsonlDataset(
                    path=os.path.join(self.args.data, split, cur_shard_str, file),
                    # tokenizer=self._tokenize_one_json,
                    tokenizer=self._tokenize_ra_json
                    if self.has_retrieval
                    else self._tokenize_one_json,
                    epoch=epoch,
                    data_subshard_count=data_subshard_count,
                )
            )
            corpora.append(os.path.splitext(file)[0])
        assert len(datasets) > 0

        if self.args.multicorpus_sampling_alpha != 1:
            datasets = self._alpha_sampling(datasets, corpora, epoch)

        dataset = torch.utils.data.ConcatDataset(datasets)

        # chunk into blocks of tokens
        if self.has_cm3:
            # We chose not to use compositional inheritance because there's a
            # lot of downstream code that has isinstance checks.
            # So just to be safe and not change anything we use proper inheritance.
            self.datasets[split] = CausalMaskedDocumentToSequenceDataset(
                sentinel_token_expectation=self.args.cm3_lambda_sentinel_tokens,
                sentinel_tokens=self.cm3_sentinel_tokens_ind,
                sentinel_method=self.cm3_sentinel_type,
                sentinel_eos=self.cm3_sentinel_end_ind,
                allow_rotation_across_eod=self.args.cm3_allow_across_eod_boundaries,
                eod=self.cm3_break_ind,
                dataset=dataset,
                # We generate blocks with one extra token, so that we have a target
                # for the final input token. This results in slight data loss.
                block_size=self.args.tokens_per_sample + 1,
                break_mode=self.args.sample_break_mode,
                # we drop the remainder block during training
                drop_last=(split == "train"),
                padding_idx=self.source_dictionary.pad(),
                seed=self.args.seed,
                percent_full_document_rotation=self.args.cm3_percent_full_document_rotation
            )
        else:
            self.datasets[split] = DocumentToSequenceDataset(
                dataset,
                block_size=self.args.tokens_per_sample + 1,
                break_mode=self.args.sample_break_mode,
                drop_last=(split == "train"),
                padding_idx=self.source_dictionary.pad(),
                seed=self.args.seed,
            )

    def _collate_fn(self, items: List[Dict[str, Any]]):
        # StreamingTokenBlockDataset returns None as filler
        if len([x for x in items if x is not None]) == 0:
            return {}

        tokens = data_utils.collate_tokens(
            [x["block"] for x in items if x is not None],
            pad_idx=self.source_dictionary.pad(),
            pad_to_bsz=self.args.batch_size,
        )
        # generate inputs and targets
        input = tokens[:, :-1].contiguous()
        target = tokens[:, 1:].contiguous()

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
        }

    def dataset(self, split):
        return self.datasets[split]

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
        dataset.set_num_workers(num_workers)

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

    @property
    def source_dictionary(self):
        return self.dictionary

    @property
    def target_dictionary(self):
        return self.dictionary
