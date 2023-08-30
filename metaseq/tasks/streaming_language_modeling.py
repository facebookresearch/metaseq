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
import re
import zipfile
from collections import defaultdict, namedtuple
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from metaseq.data import (
    data_utils,
    Dictionary,
    iterators,
    JsonlDataset,
    PartitionedStreamingDataset,
    ResamplingDataset,
    StreamingSrcTgtDataset,
)
from metaseq.data.cm3_dataset import CausalMaskedDocumentToSequenceDataset
from metaseq.data.document_to_sequence import DocumentToSequenceDataset

from metaseq.dataclass import ChoiceEnum, ChoiceEnum, MetaseqDataclass
from metaseq.tasks import LegacyTask, register_task
from metaseq.utils import print_r0
from omegaconf import II

try:
    from tokenizers import ByteLevelBPETokenizer, Tokenizer

    has_hf_tokenizers = True
except ImportError:
    has_hf_tokenizers = False


logger = logging.getLogger(__name__)

DEFAULT_MULTICORPUS_MAX = -1

LANGUAGE_MODELING_MODE = ChoiceEnum(["standard", "cm3", "racm3"])
CM3_MODE = ChoiceEnum(["poisson", "fixed", "fim"])

DatasetWithShardInformation = namedtuple(
    "DatasetWithShardInformation", ["dataset", "is_sharded", "shard_id", "num_shards"]
)

TEXT_DATA_EVALSETS = ["llama", "text_eval", "marmot"]
IMAGE_PREFIX = "IMGIMG"


def map_old_image_token_to_new_image_token(text):
    text = text.replace("I", IMAGE_PREFIX)
    for i in range(10):
        text = text.replace(str(i), chr(ord("A") + i))
    return text.replace(" ", "Z")


def map_new_image_token_to_old_image_token(text):
    text = text.replace("Z", " ")
    for i in range(10):
        text = text.replace(chr(ord("A") + i), str(i))
    return text.replace(IMAGE_PREFIX, "I")


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
    no_break_image: bool = field(
        default=False, metadata={"help": "don't break images across two data samples"}
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
        self._build_cm3_special_tokens()
        if self.has_cm3:
            self._check_cm3_parameterization()
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

        (
            self.start_image_token_index,
            self.end_image_token_index,
        ) = self._find_image_token_bounds()

        logger.info(
            f"First Image Token and Index: {self.dictionary[self.start_image_token_index]} -- {self.start_image_token_index}"
        )
        logger.info(
            f"Last Image Token and Index: {self.dictionary[self.end_image_token_index]} -- {self.end_image_token_index}"
        )
        assert (
            self.args.data_subshard_count == 1
        ), "We utilize subsharding as a way to do faster data processing therefore do not allow for another layer of subsharding."

    def _find_image_token_bounds(self):
        start = None
        end = None
        for i in range(len(self.dictionary)):
            token: str = self.dictionary[i]
            token_has_prefix = token.startswith(IMAGE_PREFIX)
            if start is None and token_has_prefix:
                start = i
            elif token_has_prefix:
                end = i

        assert IMAGE_PREFIX in self.dictionary[start]
        assert IMAGE_PREFIX in self.dictionary[end]
        return start, end

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

    def _build_cm3_special_tokens(self):
        self.cm3_sentinel_end = "<eoss>"
        self.cm3_break = "<racm3:break>"
        self.cm3_sentinel_tokens = [
            f"<sentinel:{i}>" for i in range(self.args.cm3_num_sentinel_tokens)
        ]
        self.cm3_sentinel_tokens_ind = []
        for token in self.cm3_sentinel_tokens:
            token_index = self.dictionary.index(token)
            assert token_index != self.dictionary.unk_index
            self.cm3_sentinel_tokens_ind.append(token_index)
        self.cm3_sentinel_end_ind = self.dictionary.index(self.cm3_sentinel_end)
        self.cm3_break_ind = self.dictionary.index(self.cm3_break)

    @classmethod
    def setup_task(cls, args, **kwargs):
        return cls(args)

    def tokenize_cm3_v2(self, json):
        if "dataset_name" not in json:
            return self._tokenize_text_json(json)
        else:
            if json["dataset_name"] in ["m2c2v2.5", "m2c2v3"]:
                return self._tokenize_m2c2_json(json)
            elif json["dataset_name"] == "shutterstock":
                return self._tokenize_ra_json(json)
            elif json["dataset_name"] == "openflamingo":
                return self._tokenize_interleaving_json(json)
            elif json["dataset_name"] == "m2c2-cm3":
                # TODO VALIDATE if m2c2-cm3 needs special handling
                return self._tokenize_interleaving_json(json)
            else:
                raise ValueError(f"dataset not valid: {json['dataset_name']}")

    def _tokenize_text_json(self, json):
        if "text" in json:
            text = json["text"]
        elif "content" in json:
            text = json["content"]
        else:
            text = str(json)
        return (
            torch.LongTensor(
                # append an end-of-document symbol after each document
                [self.eod]
                + self.tokenizer.encode(text.rstrip()).ids
                + [self.eod]
            ),
            None,
        )

    def _tokenize_image(self, image_str):
        image = map_old_image_token_to_new_image_token(image_str)
        image_indexes = self.tokenizer.encode(image.rstrip()).ids
        assert (
            len(image_indexes) == 1024
        ), f"Each image must be 1024 tokens, instead we got {len(image_indexes)}"
        # if  not len(image_indexes) == 1024 and torch.distributed.get_rank() == 0:
        #     from metaseq import pdb; pdb.set_trace()
        assert all(
            [i > 3 for i in image_indexes]
        ), f"Images should not have any special tokens: {image_indexes}"
        return image_indexes

    def tokenize_single_html_doc(self, doc, add_eod=False):
        doc = parse_doc(doc)
        text, image = doc["text"], doc["image"]
        text_indexes = self.tokenizer.encode(text.rstrip()).ids
        image = image.strip() + " "
        image_indexes = self._tokenize_image(image)
        indexes = text_indexes + [self.cm3_break_ind] + image_indexes
        image_span = (len(text_indexes + [self.cm3_break_ind]), len(indexes))  # [)
        if add_eod:
            indexes = indexes + [self.eod]
            image_span = (image_span[0], image_span[1] + 1)
        return indexes, image_span

    def _tokenize_m2c2_json(self, json):
        query_index, image_span = self.tokenize_single_html_doc(
            json["text"], add_eod=True
        )
        final_indexes = torch.LongTensor([self.eod] + query_index)
        image_span = [(image_span[0] + 1, image_span[1] + 1)]
        return final_indexes, image_span

    def _tokenize_interleaving_json(self, json):
        all_tokens = [self.eod]
        all_texts = json["text_list"]
        textid_2_image = defaultdict(list)
        image_spans = []
        for image in json["image_info"]:
            if "IMAGE_TOKENS" in image:
                textid_2_image[image["matched_text_index"]].append(
                    image["IMAGE_TOKENS"]
                )
        for text_idx, text in enumerate(all_texts):
            all_tokens += self.tokenizer.encode(text.rstrip()).ids
            if len(textid_2_image[text_idx]):
                for image in textid_2_image[text_idx]:
                    image = image.replace('"', "").strip() + " "
                    image_indexes = self._tokenize_image(image)
                    all_tokens += [self.cm3_break_ind]
                    # there is always a special token after the image tokens, thus we +1 for the end of image
                    image_spans.append(
                        (len(all_tokens), len(all_tokens) + len(image_indexes) + 1)
                    )
                    all_tokens += image_indexes
                all_tokens += [self.cm3_break_ind]

        all_tokens += [self.eod]
        all_tokens = [int(x) for x in all_tokens]
        final_indexes = torch.LongTensor(all_tokens)
        return final_indexes, image_spans

    def _tokenize_ra_json(self, json):
        image_spans = []
        query_index, query_image_span = self.tokenize_single_html_doc(
            json["text"], add_eod=True
        )
        query_index = torch.LongTensor(query_index)
        ra_docs = json["retrieved_docs_from_img"] + json["retrieved_docs_from_txt"]
        np.random.shuffle(ra_docs)

        ra_docs = ra_docs[: self.args.num_retrieved_doc]
        ra_indexes = []
        cur_len = (
            1  # this accounts for the initial eod token at line of final_indexes below
        )
        for ra_doc in ra_docs:
            ra_index, image_span = self.tokenize_single_html_doc(ra_doc, add_eod=False)
            ra_index = torch.LongTensor(ra_index + [self.cm3_break_ind])
            ra_indexes.append(ra_index)
            # there is always a special token after the image tokens, thus we +1 for the end of image
            image_span = (cur_len + image_span[0], cur_len + image_span[1] + 1)
            cur_len += len(ra_index)
            image_spans.append(image_span)

        final_indexes = torch.cat(
            [torch.LongTensor([self.eod])] + ra_indexes + [query_index]
        )
        image_spans.append(
            (cur_len + query_image_span[0], cur_len + query_image_span[1])
        )
        return final_indexes, image_spans

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
        return cur_shard_str, shards

    def get_previous_shard_str(self, epoch, split):
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
        prev_shard_str = shards[shard_idx - 1] if shard_idx > 0 else None
        return prev_shard_str

    def load_dataset(
        self,
        split: str,
        epoch=1,
        combine=False,
        num_shards=None,
        shard_id=None,
        **kwargs,
    ):
        is_sharded = True
        if num_shards is None or shard_id is None:
            num_shards = 1
            shard_id = 1
            is_sharded = False
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
        cur_shard_str, all_shards = self.get_shard_str(epoch, split)

        # if torch.distributed.get_rank() == 0:
        #     from metaseq import pdb; pdb.set_trace()

        if split == "train" and "zip" in self.args.data:
            # Delete jsonl files
            prev_shard_str = self.get_previous_shard_str(epoch, split)
            if prev_shard_str is not None and split == "train":
                path_to_unzip_files = os.path.join(
                    self.args.data, split, prev_shard_str
                )
                previous_extracted_files = [
                    f"{path_to_unzip_files}/{f}"
                    for f in os.listdir(path_to_unzip_files)
                    if f.endswith(".jsonl")
                ]
                print_r0("previous_extracted_files", previous_extracted_files)
                for f in previous_extracted_files:
                    if torch.distributed.get_rank() == 0:
                        print_r0("removing!: ", f)
                        os.remove(f)

            # Get list of all zip files
            path_to_zip_files = os.path.join(self.args.data, split, cur_shard_str)
            print_r0("path_to_zip_files", path_to_zip_files)
            zip_files = [f for f in os.listdir(path_to_zip_files) if f.endswith(".zip")]

            for zip_filename in zip_files:
                # Create a ZipFile object
                with zipfile.ZipFile(
                    os.path.join(path_to_zip_files, zip_filename), "r"
                ) as zip_ref:
                    # Extract all the contents of the zip file in specified directory
                    zip_ref.extractall(path=path_to_zip_files)
            # List the extracted files
            extracted_files = [
                f for f in os.listdir(path_to_zip_files) if f.endswith(".jsonl")
            ]
            print_r0("extracted_files", extracted_files)

        # concatenate any jsonl files that are part of the shard
        datasets, corpora = [], []
        data_subshard_count = self.args.data_subshard_count if split == "train" else 1
        for key, cur_shard_str in all_shards.items():
            for file in sorted(
                os.listdir(os.path.join(self.args.data, split, cur_shard_str))
            ):
                if not file.endswith(".jsonl"):
                    continue
                datasets.append(
                    JsonlDataset(
                        path=os.path.join(self.args.data, split, cur_shard_str, file),
                        tokenizer=self.tokenize_cm3_v2,
                        epoch=shard_id,
                        data_subshard_count=num_shards,
                    )
                )
                corpora.append(os.path.splitext(file)[0])
        assert len(datasets) > 0

        if self.args.multicorpus_sampling_alpha != 1:
            datasets = self._alpha_sampling(datasets, corpora, epoch)

        dataset = torch.utils.data.ConcatDataset(datasets)

        break_mode = "complete" if split != "train" else self.args.sample_break_mode
        no_image_break = False if split != "train" else self.args.no_image_break
        is_text = any([subset in split for subset in TEXT_DATA_EVALSETS])

        # chunk into blocks of tokens
        if self.has_cm3 and not is_text:
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
                break_mode=break_mode,
                # we drop the remainder block during training
                drop_last=(split == "train"),
                padding_idx=self.source_dictionary.pad(),
                seed=self.args.seed,
                percent_full_document_rotation=self.args.cm3_percent_full_document_rotation,
                no_break_image=no_image_break,
            )
        else:
            self.datasets[split] = DocumentToSequenceDataset(
                dataset,
                block_size=self.args.tokens_per_sample + 1,
                break_mode=break_mode,
                drop_last=(split == "train"),
                padding_idx=self.source_dictionary.pad(),
                seed=self.args.seed,
            )

        self.datasets[split] = DatasetWithShardInformation(
            self.datasets[split],
            is_sharded=is_sharded,
            shard_id=shard_id,
            num_shards=num_shards,
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
        assert isinstance(dataset, DatasetWithShardInformation)
        if dataset.is_sharded:
            # guarantee partitioning/shard consistency across load_dataset/get_iterator's.
            assert num_shards == dataset.num_shards
            assert shard_id == dataset.shard_id - 1

        assert isinstance(dataset.dataset, DocumentToSequenceDataset) or isinstance(
            dataset, StreamingSrcTgtDataset
        )
        shuffle_buffer_size = 10 * max_sentences * num_shards
        logger.info(f"setting shuffle buffer size to {shuffle_buffer_size}")
        dataset.dataset.set_shuffle_buffer_size(shuffle_buffer_size)
        dataset.dataset.set_num_workers(num_workers)

        if not dataset.is_sharded:
            # The dataset is not sharded, so we shard.
            dataset = PartitionedStreamingDataset(
                dataset.dataset,
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
        else:
            # We don't need to partition by worker ID because partitioning already happens in JSONLDataset
            # iterators.StreamingEpochBatchIterator expects and iterable dataset
            # we can trivially achieve this by using PartitionedStreamingDataset with
            # 0/1 shards.
            dataset = PartitionedStreamingDataset(
                dataset.dataset,
                num_shards=1,
                shard_id=0,
                drop_last=skip_remainder_batch,
            )
            return iterators.StreamingShardedEpochBatchIterator(
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
