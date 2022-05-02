# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
from dataclasses import dataclass, field
from typing import Optional, List

import numpy as np
import torch
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
    PrependTokenDataset,
    StripTokenDataset,
    TokenBlockDataset,
    MultiplePadDataset,
    data_utils,
)
from metaseq.data.indexed_dataset import get_available_dataset_impl
from metaseq.data.shorten_dataset import maybe_shorten_dataset
from metaseq.dataclass import ChoiceEnum, MetaseqDataclass
from metaseq.tasks import LegacyTask, register_task

SAMPLE_BREAK_MODE_CHOICES = ChoiceEnum(["none", "complete", "complete_doc", "eos"])
SHORTEN_METHOD_CHOICES = ChoiceEnum(["none", "truncate", "random_crop"])
logger = logging.getLogger(__name__)

try:
    from tokenizers import ByteLevelBPETokenizer

    has_hf_tokenizers = True
except ImportError:
    has_hf_tokenizers = False


@dataclass
class LanguageModelingInferenceForModelsTrainedWithStreamingConfig(MetaseqDataclass):
    data: Optional[str] = field(
        default=None, metadata={"help": "path to data directory"}
    )
    vocab_filename: Optional[str] = field(
        default="", metadata={"help": "path to bpe-vocab.json"}
    )
    merges_filename: Optional[str] = field(
        default="", metadata={"help": "path to bpe-merges.json"}
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
    output_dictionary_size: int = field(
        default=-1, metadata={"help": "limit the size of output dictionary"}
    )
    self_target: bool = field(default=False, metadata={"help": "include self target"})
    future_target: bool = field(
        default=False, metadata={"help": "include future target"}
    )
    past_target: bool = field(default=False, metadata={"help": "include past target"})
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
    update_freq: List[int] = II("optimization.update_freq")


@register_task(
    "language_modeling_inference_for_models_trained_with_streaming",
    dataclass=LanguageModelingInferenceForModelsTrainedWithStreamingConfig,
)
class LanguageModelingInferenceForModelsTrainedWithStreamingTask(LegacyTask):
    """
    This class is specially developed for inference of models trained
    with the new StreamingLanguageModeling but follows closely the language_modeling implementation.

    Args:
        dictionary (~metaseq.data.Dictionary): the dictionary for the input of
            the language model
        output_dictionary (~metaseq.data.Dictionary): the dictionary for the
            output of the language model. In most cases it will be the same as
            *dictionary*, but could possibly be a more limited version of the
            dictionary (if ``--output-dictionary-size`` is used).
        targets (List[str]): list of the target types that the language model
            should predict.  Can be one of "self", "future", and "past".
            Defaults to "future".

    .. note::

        The language modeling task is compatible with :mod:`metaseq-train`,
        :mod:`metaseq-generate`, :mod:`metaseq-interactive` and
        :mod:`metaseq-eval-lm`.

    The language modeling task provides the following additional command-line
    arguments:

    .. argparse::
        :ref: metaseq.tasks.language_modeling_parser
        :prog:
    """

    def __init__(self, args, tokenizer, targets=None):
        super().__init__(args)

        self.tokenizer = tokenizer

        self.eod = self.tokenizer.token_to_id(args.end_of_document_symbol)
        if self.eod is None:
            # fix for 7B_punctsplit_bpe_relu2_229k
            self.eod = self.tokenizer.token_to_id("<endoftext|>")

        assert (
            self.eod is not None
        ), "Cannot find end-of-document symbol ({}) in tokenizer".format(
            args.end_of_document_symbol
        )

        # construct a dummy metaseq Dictionary corresponding to the given tokenizer
        self.dictionary = Dictionary()
        for id in range(self.dictionary.nspecial, self.tokenizer.get_vocab_size()):
            # This aims at producing a dict with first symbols being special
            # (<s> <pad>) and the rest '4', '5' to be backward compatible with
            # the current few_shot eval logic. See
            # examples/few_shot/predictor.py
            if id <= max(self.eod, 3):
                self.dictionary.add_symbol(self.tokenizer.id_to_token(id))
            else:
                self.dictionary.add_symbol(str(id))

        # confirm that metaseq dictionary and BPE have matching special symbols
        assert self.dictionary.bos_index == 0
        assert self.tokenizer.id_to_token(0) in {"<BOS>", "<s>"}
        assert self.dictionary.pad_index == 1
        assert self.tokenizer.id_to_token(1) in {"<PAD>", "<pad>"}
        assert self.dictionary.eos_index == 2
        assert self.tokenizer.id_to_token(2) in {"<EOS>", "</s>"}
        assert self.dictionary.unk_index == 3
        assert self.tokenizer.id_to_token(3) in {"<UNK>", "<unk>"}

        self.dictionary.pad_to_multiple_(8)

        self.output_dictionary = self.dictionary

        if targets is None:
            targets = ["future"]
        self.targets = targets

    @classmethod
    def setup_task(cls, args, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            args (argparse.Namespace): parsed command-line arguments
        """

        if not has_hf_tokenizers:
            raise ImportError("Please install tokenizers with: pip install tokenizers")

        # set bytelevel bpe tokenizer
        tokenizer = ByteLevelBPETokenizer.from_file(
            args.vocab_filename, args.merges_filename
        )

        # upgrade old checkpoints
        if getattr(args, "exclude_self_target", False):
            args.self_target = False

        targets = []
        if getattr(args, "self_target", False):
            targets.append("self")
        if getattr(args, "future_target", False):
            targets.append("future")
        if getattr(args, "past_target", False):
            targets.append("past")
        if len(targets) == 0:
            # standard language modeling
            targets = ["future"]

        return cls(args, tokenizer, targets=targets)

    def build_model(self, args):
        model = super().build_model(args)
        for target in self.targets:
            if target not in model.supported_targets:
                raise ValueError(
                    f"Unsupported language modeling target: {target} not in {model.supported_targets}"
                )

        return model

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
            tgt_vocab=self.output_dictionary,
            add_eos_for_other_targets=add_eos_for_other_targets,
            shuffle=True,
            targets=self.targets,
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
                else self.eod
                # In some models we accidently replaced this with <endoftext/>
                # (id:4) but it seems they work with eos as well so we will
                # keep this way in this experimental task until figure a better
                # flexible soluton.
            ),
        )
        src_dataset = MultiplePadDataset(
            src_dataset, pad_idx=self.source_dictionary.pad(), multiple=8
        )
        tgt_dataset = AppendTokenDataset(dataset, token=self.source_dictionary.pad())
        return NestedDictionaryDataset(
            {
                "id": IdDataset(),
                "net_input": {
                    "src_tokens": src_dataset,
                    "src_lengths": NumelDataset(src_dataset, reduce=False),
                },
                "target": MultiplePadDataset(
                    tgt_dataset, pad_idx=self.source_dictionary.pad(), multiple=8
                ),
            },
            sizes=[np.array(src_lengths)],
        )

    def inference_step(
        self, generator, models, sample, prefix_tokens=None, constraints=None
    ):
        with torch.no_grad():
            # Generation will always be conditioned on bos_token
            if getattr(self.args, "add_bos_token", False):
                bos_token = self.source_dictionary.bos()
            else:
                bos_token = self.source_dictionary.eos()

            if constraints is not None:
                raise NotImplementedError(
                    "Constrained decoding with the language_modeling task is not supported"
                )

            # SequenceGenerator doesn't use src_tokens directly, we need to
            # pass the `prefix_tokens` argument instead
            if prefix_tokens is None and sample["net_input"]["src_tokens"].nelement():
                prefix_tokens = sample["net_input"]["src_tokens"]
                if prefix_tokens[:, 0].eq(bos_token).all():
                    prefix_tokens = prefix_tokens[:, 1:]

            return generator.generate(
                models, sample, prefix_tokens=prefix_tokens, bos_token=bos_token
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
        """Return the :class:`~metaseq.data.Dictionary` for the language
        model."""
        return self.dictionary

    @property
    def target_dictionary(self):
        """Return the :class:`~metaseq.data.Dictionary` for the language
        model."""
        return self.output_dictionary
