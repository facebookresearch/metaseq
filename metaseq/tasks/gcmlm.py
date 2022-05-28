# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
from dataclasses import dataclass, field
from typing import Optional

from metaseq import utils
from metaseq.data import (
    Dictionary,
    MonolingualDataset,
    GCMLMDataset,
    TokenBlockDataset,
    data_utils,
)
from metaseq.dataclass import MetaseqDataclass, ChoiceEnum
from metaseq.tasks import LegacyTask, register_task
from omegaconf import II


SAMPLE_BREAK_MODE_CHOICES = ChoiceEnum(["none", "complete", "complete_doc", "eos"])
SHORTEN_METHOD_CHOICES = ChoiceEnum(["none", "truncate", "random_crop"])
logger = logging.getLogger(__name__)


@dataclass
class GCMLMConfig(MetaseqDataclass):
    # Data parsing related arguments
    data: Optional[str] = field(
        default=None, metadata={"help": "path to data directory with JSONL files"}
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
    max_source_positions: Optional[int] = field(
        default=None, metadata={"help": "max number of tokens in the source sequence"}
    )
    max_target_positions: Optional[int] = field(
        default=None, metadata={"help": "max number of tokens in the target sequence"}
    )
    final_vocab_size: Optional[int] = field(
        default=None, metadata={"help": "force vocab size to this"}
    )

    # GCMLM related args
    mask_prob: float = field(
        default=0.15,
        metadata={
            "help": "mask tokens with the given probability"
        }
    )
    clm_prob: float = field(
        default=0.0,
        metadata={
            "help": "use bidir_attn_prefix=0 and mask_prob=0 with the given probability"
        }
    )
    fully_causal_prob: float = field(
        default=0.0,
        metadata={
            "help": "use bidir_attn_prefix=1 with the given probability"
        }
    )
    fully_bidir_prob: float = field(
        default=0.0,
        metadata={
            "help": "use bidir_attn_prefix=seqlen with the given probability"
        }
    )
    predict_masked_only: Optional[bool] = field(
        default=False, metadata={"help": "only predict masked tokens"}
    )
    pad_to_fixed_length: Optional[bool] = field(
        default=False, metadata={"help": "pad to fixed length"},
    )
    pad_to_fixed_bsz: Optional[bool] = field(
        default=False, metadata={"help": "boolean to pad to fixed batch size"},
    )

    # TODO common vars below add to parent
    seed: int = II("common.seed")
    batch_size: Optional[int] = II("dataset.batch_size")
    batch_size_valid: Optional[int] = II("dataset.batch_size_valid")
    dataset_impl: Optional[ChoiceEnum(get_available_dataset_impl())] = II(
        "dataset.dataset_impl"
    )
    data_buffer_size: int = II("dataset.data_buffer_size")
    tpu: bool = II("common.tpu")
    use_plasma_view: bool = II("common.use_plasma_view")
    plasma_path: str = II("common.plasma_path")


@register_task("gcmlm", dataclass=GCMLMConfig)
class GCMLMTask(LegacyTask):

    def __init__(self, args, dictionary):
        super().__init__(args)
        self.dictionary = dictionary
        self.mask_idx = self.dictionary.add_symbol("<mask>")

    @classmethod
    def setup_dictionary(cls, args, **kwargs):
        dictionary = None
        if args.data:
            paths = utils.split_paths(args.data)
            assert len(paths) > 0
            dictionary = Dictionary.load(os.path.join(paths[0], "dict.txt"))
            logger.info("dictionary: {} types".format(len(dictionary)))
        return dictionary

    @classmethod
    def setup_task(cls, args, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            args (argparse.Namespace): parsed command-line arguments
        """
        dictionary = cls.setup_dictionary(args, **kwargs)
        return cls(args, dictionary)

    def load_dataset(
        self, split: str, epoch=1, combine=False, **kwargs
    ) -> MonolingualDataset:
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

        dataset = TokenBlockDataset(
            dataset,
            dataset.sizes,
            self.args.tokens_per_sample,
            pad=self.dictionary.pad(),
            eos=self.dictionary.eos(),
            break_mode=self.args.sample_break_mode,
            include_targets=False,
            use_plasma_view=self.args.use_plasma_view,
            split_path=split_path,
            plasma_path=self.args.plasma_path,
        )
        logger.info("loaded {} blocks from: {}".format(len(dataset), split_path))
        logger.info(f"average sequence length: {dataset.sizes.mean():.2f}")

        fixed_pad_length = None
        if self.args.pad_to_fixed_length:
            fixed_pad_length = self.args.tokens_per_sample

        pad_to_bsz = None
        if self.args.pad_to_fixed_bsz:
            pad_to_bsz = self.args.batch_size_valid if 'valid' in split else self.args.batch_size

        self.datasets[split] = GCMLMDataset(
            dataset=dataset,
            sizes=dataset.sizes,
            vocab=self.dictionary,
            mask_idx=self.mask_idx,
            mask_prob=self.args.mask_prob,
            clm_prob=self.args.clm_prob,
            fully_causal_prob=self.args.fully_causal_prob,
            fully_bidir_prob=self.args.fully_bidir_prob,
            predict_masked_only=self.args.predict_masked_only,
            shuffle=True,
            fixed_pad_length=fixed_pad_length,
            pad_to_bsz=pad_to_bsz,
        )

    def build_dataset_for_inference(self, src_tokens, src_lengths, **kwargs):
        # TODO: Not sure what we want to do here
        raise NotImplementedError()

    @property
    def source_dictionary(self):
        """Return the :class:`~fairseq.data.Dictionary` for the language
        model."""
        return self.dictionary

    @property
    def target_dictionary(self):
        """Return the :class:`~fairseq.data.Dictionary` for the language
        model."""
        return self.dictionary
