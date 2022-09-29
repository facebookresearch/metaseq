# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from omegaconf import II

from metaseq import utils
from metaseq.data import (
    AppendTokenDataset,
    Dictionary,
    IdDataset,
    LMContextWindowDataset,
    MonolingualDataset,
    NestedDictionaryDataset,
    NumelDataset,
    PadDataset,
    PrependTokenDataset,
    StripTokenDataset,
    SortDataset,
    TokenBlockDataset,
    data_utils,
)
from metaseq.data.indexed_dataset import get_available_dataset_impl
from metaseq.data.shorten_dataset import maybe_shorten_dataset
from metaseq.dataclass import ChoiceEnum, MetaseqDataclass
from metaseq.tasks import LegacyTask, register_task

try:
    from tokenizers import ByteLevelBPETokenizer

    has_hf_tokenizers = True
except ImportError:
    has_hf_tokenizers = False

SAMPLE_BREAK_MODE_CHOICES = ChoiceEnum(["none", "complete", "complete_doc", "eos"])
SHORTEN_METHOD_CHOICES = ChoiceEnum(["none", "truncate", "random_crop"])
logger = logging.getLogger(__name__)


@dataclass
class LanguageModelingConfig(MetaseqDataclass):
    data: Optional[str] = field(
        default=None, metadata={"help": "path to data directory"}
    )
    # Begin args from StreamingLanguageModelingConfig
    vocab_filename: Optional[str] = field(
        default="", metadata={"help": "path to bpe-vocab.json"}
    )
    merges_filename: Optional[str] = field(
        default="", metadata={"help": "path to bpe-merges.txt"}
    )
    end_of_document_symbol: Optional[str] = field(
        default="</s>", metadata={"help": "symbol indicating an end-of-document"}
    )
    final_vocab_size: Optional[int] = field(
        default=None, metadata={"help": "force vocab size to this"}
    )  # End of args from StreamingLanguageModelingConfig
    sample_break_mode: SAMPLE_BREAK_MODE_CHOICES = field(
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
    add_bos_token: bool = field(
        default=False, metadata={"help": "prepend beginning of sentence token (<s>)"}
    )
    max_source_positions: Optional[int] = field(
        default=None, metadata={"help": "max number of tokens in the source sequence"}
    )
    max_target_positions: Optional[int] = field(
        default=None, metadata={"help": "max number of tokens in the target sequence"}
    )
    shorten_method: SHORTEN_METHOD_CHOICES = field(
        default="none",
        metadata={
            "help": "if not none, shorten sequences that exceed --tokens-per-sample"
        },
    )
    shorten_data_split_list: str = field(
        default="",
        metadata={
            "help": "comma-separated list of dataset splits to apply shortening to, "
            'e.g., "train,valid" (default: all dataset splits)'
        },
    )
    pad_to_fixed_length: Optional[bool] = field(
        default=False,
        metadata={"help": "pad to fixed length"},
    )
    pad_to_fixed_bsz: Optional[bool] = field(
        default=False,
        metadata={"help": "boolean to pad to fixed batch size"},
    )
    shuffle_docs: bool = field(
        default=False,
        metadata={
            "help": "Only for sample break mode EOS, first shuffle docs and then create blocks"
        },
    )

    # TODO common vars below add to parent
    seed: int = II("common.seed")
    batch_size: Optional[int] = II("dataset.batch_size")
    batch_size_valid: Optional[int] = II("dataset.batch_size_valid")
    dataset_impl: Optional[ChoiceEnum(get_available_dataset_impl())] = II(
        "dataset.dataset_impl"
    )
    data_buffer_size: int = II("dataset.data_buffer_size")
    use_plasma_view: bool = II("common.use_plasma_view")
    plasma_path: str = II("common.plasma_path")


# TODO(susanz): Deprecate this task. This pre-date StreamingLanguageModelingTask,
#  when tokenization happened offline.
@register_task("language_modeling", dataclass=LanguageModelingConfig)
class LanguageModelingTask(LegacyTask):
    """
    Train a language model.  To be deprecated.
    """

    def __init__(self, args):
        super().__init__(args)

        if not has_hf_tokenizers:
            raise ImportError("Please install tokenizers with: pip install tokenizers")

        self.tokenizer = ByteLevelBPETokenizer.from_file(
            args.vocab_filename, args.merges_filename
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

    @classmethod
    def setup_task(cls, args, **kwargs):
        return cls(args)

    def load_dataset(self, split: str, epoch=1, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, valid1, test)
        """
        paths = utils.split_paths(self.args.data)
        assert len(paths) > 0

        data_path = paths[(epoch - 1) % len(paths)]
        split_path = os.path.join(data_path, split)

        # each process has its own copy of the raw data (likely to be an np.memmap)
        dataset = data_utils.load_indexed_dataset(
            split_path, self.dictionary, self.args.dataset_impl, combine=combine
        )
        if dataset is None:
            raise FileNotFoundError(f"Dataset not found: {split} ({split_path})")

        dataset = maybe_shorten_dataset(
            dataset,
            split,
            self.args.shorten_data_split_list,
            self.args.shorten_method,
            self.args.tokens_per_sample,
            self.args.seed,
        )

        if self.args.shuffle_docs:
            assert (
                self.args.sample_break_mode == "none"
            ), "shuffle docs is only for sample break mode none"
            dataset = TokenBlockDataset(
                dataset,
                dataset.sizes,
                self.args.tokens_per_sample,
                pad=self.dictionary.pad(),
                eos=self.dictionary.eos(),
                break_mode="complete_doc",
                include_targets=False,
                use_plasma_view=self.args.use_plasma_view,
                split_path=split_path,
                plasma_path=self.args.plasma_path,
            )
            with data_utils.numpy_seed(self.args.seed + epoch):
                shuffle = np.random.permutation(len(dataset))

            dataset = SortDataset(
                dataset,
                sort_order=[
                    shuffle,
                    dataset.sizes,
                ],
            )
        dataset = TokenBlockDataset(
            dataset,
            dataset.sizes,
            self.args.tokens_per_sample,
            pad=self.dictionary.pad(),
            eos=self.dictionary.eos(),
            break_mode=self.args.sample_break_mode,
            include_targets=True,
            use_plasma_view=self.args.use_plasma_view,
            split_path=split_path,
            plasma_path=self.args.plasma_path,
        )
        add_eos_for_other_targets = (
            self.args.sample_break_mode is not None
            and self.args.sample_break_mode != "none"
        )
        fixed_pad_length = None
        if self.args.pad_to_fixed_length:
            fixed_pad_length = self.args.tokens_per_sample

        pad_to_bsz = None
        if self.args.pad_to_fixed_bsz:
            pad_to_bsz = (
                self.args.batch_size_valid if "valid" in split else self.args.batch_size
            )

        self.datasets[split] = MonolingualDataset(
            dataset=dataset,
            sizes=dataset.sizes,
            src_vocab=self.dictionary,
            tgt_vocab=self.dictionary,
            add_eos_for_other_targets=add_eos_for_other_targets,
            shuffle=True,
            add_bos_token=self.args.add_bos_token,
            fixed_pad_length=fixed_pad_length,
            pad_to_bsz=pad_to_bsz,
        )

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
            token=(
                self.source_dictionary.bos()
                if getattr(self.args, "add_bos_token", False)
                else self.source_dictionary.eos()
            ),
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

    def eval_lm_dataloader(
        self,
        dataset,
        max_tokens: Optional[int] = 36000,
        batch_size: Optional[int] = None,
        max_positions: Optional[int] = None,
        num_shards: int = 1,
        shard_id: int = 0,
        num_workers: int = 1,
        data_buffer_size: int = 10,
        # ensures that every evaluated token has access to a context of at least
        # this size, if possible
        context_window: int = 0,
    ):
        if context_window > 0:
            dataset = LMContextWindowDataset(
                dataset=dataset,
                tokens_per_sample=self.args.tokens_per_sample,
                context_window=context_window,
                pad_idx=self.source_dictionary.pad(),
            )
        return self.get_batch_iterator(
            dataset=dataset,
            max_tokens=max_tokens,
            max_sentences=batch_size,
            max_positions=max_positions,
            ignore_invalid_inputs=True,
            num_shards=num_shards,
            shard_id=shard_id,
            num_workers=num_workers,
            data_buffer_size=data_buffer_size,
        )

    @property
    def source_dictionary(self):
        return self.dictionary

    @property
    def target_dictionary(self):
        return self.dictionary
