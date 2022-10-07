# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from dataclasses import dataclass, field
from typing import Optional

import torch
from metaseq.data import Dictionary
from metaseq.tasks import register_task
from metaseq.tasks.language_modeling_inference_for_models_trained_with_streaming import (
    LanguageModelingInferenceForModelsTrainedWithStreamingConfig as LMInferenceStreamingConfig,
)
from metaseq.tasks.language_modeling_inference_for_models_trained_with_streaming import (
    LanguageModelingInferenceForModelsTrainedWithStreamingTask as LMInferenceStreamingTask,
)
from metaseq.tasks.streaming_CM3_language_modeling import IMAGE_PREFIX, SPEECH_PREFIX

logger = logging.getLogger(__name__)

try:
    from tokenizers import (
        ByteLevelBPETokenizer,
        Tokenizer,
        decoders,
        models,
        normalizers,
        pre_tokenizers,
        Regex,
    )
    from tokenizers.pre_tokenizers import ByteLevel, Digits

    has_hf_tokenizers = True
except ImportError:
    has_hf_tokenizers = False


@dataclass
class CM3LanguageModelingInferenceForModelsTrainedWithStreamingConfig(
    LMInferenceStreamingConfig
):
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
    spm_path: Optional[str] = field(
        default=None, metadata={"help": "path to data directory with JSONL files"}
    )
    final_vocab_size: Optional[int] = field(
        default=None, metadata={"help": "force vocab size to this"}
    )


@register_task(
    "cm3_language_modeling_inference_for_models_trained_with_streaming",
    dataclass=CM3LanguageModelingInferenceForModelsTrainedWithStreamingConfig,
)
class CM3LanguageModelingInferenceForModelsTrainedWithStreamingTask(
    LMInferenceStreamingTask
):
    """
    This class is specially developed for inference of models trained
    with the new StreamingLanguageModeling but follows closely the language_modeling implementation.
    """

    def _initialize_gpt2_tokenizer(self, args):
        tokenizer = ByteLevelBPETokenizer.from_file(
            args.vocab_filename, args.merges_filename
        )
        tokenizer.add_special_tokens(
            [f"{IMAGE_PREFIX}{x} " for x in range(args.image_tokens)]
        )
        tokenizer.add_special_tokens(
            [f"{SPEECH_PREFIX}{x} " for x in range(args.speech_tokens)]
        )
        tokenizer.add_special_tokens(self.sentinel_tokens)
        tokenizer.add_special_tokens([self.sentinel_end])

        return tokenizer

    def _initialize_unigram_tokenizer(self, args):
        # if "1.3" in args.spm_path:
        if "1.4" not in args.spm_path:  # tokenizers for 1.3 and previous
            tokenizer = Tokenizer(models.Unigram()).from_file(args.spm_path)
            tokenizer.normalizer = normalizers.NFKC()
            tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
                [ByteLevel(), Digits(individual_digits=True)]
            )
            tokenizer.decoder = decoders.ByteLevel()
            logger.error(
                "Using V1.3 Tokenizer. Please double check you are using the right tokenizer."
            )
            return tokenizer
        # elif "1.4" in args.spm_path:
        else:  # 1.4 tokenizers
            tokenizer = Tokenizer(models.Unigram()).from_file(args.spm_path)
            tokenizer.normalizer = normalizers.NFKC()
            tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
                [
                    pre_tokenizers.Split(Regex(r"[\r\n]+"), "isolated"),
                    pre_tokenizers.Split(Regex(r"(I|S)(\d{1,4}) "), "isolated"),
                    ByteLevel(use_regex=False),
                    Digits(individual_digits=True),
                ]
            )
            tokenizer.decoder = decoders.ByteLevel()
            return tokenizer
        # raise ValueError("What tokenizer are you trying to use?")

    def _initialize_metaseq_dictionary(self, args):
        dictionary = Dictionary()
        tok_vocab_size = self.tokenizer.get_vocab_size()

        for id in range(dictionary.nspecial, tok_vocab_size):
            dictionary.add_symbol(self.tokenizer.id_to_token(id))
        final_vocab_size = args.final_vocab_size
        # final_vocab_size = 51200 for roberta dictionary
        if final_vocab_size is not None:
            if final_vocab_size < tok_vocab_size:
                raise ValueError(
                    f"incompatible: {final_vocab_size}, tok_vocab_size: {tok_vocab_size}"
                )
            dictionary.pad_to_multiple_(final_vocab_size)
        else:
            dictionary.pad_to_multiple_(8)

        self.dictionary = dictionary

    def _initialize_eod(self, args):
        self.eod = self.tokenizer.token_to_id(args.end_of_document_symbol)
        if self.eod is None:
            # This will be executed for old models that do not have the args.end_of_document_symbol explicitly set
            # and do not use <s/> (the default) but <EOS>
            self.eod = self.tokenizer.token_to_id("<EOS>")

    def _initialize_boundary_tokens(self, args):
        self.image_modality_start_token = self.dictionary.index(f"{IMAGE_PREFIX}0 ")
        self.image_modality_end_token = self.dictionary.index(
            f"{IMAGE_PREFIX}{args.image_tokens - 1} "
        )

        self.speech_modality_start_token = self.dictionary.index(f"{SPEECH_PREFIX}0 ")
        self.speech_modality_end_token = self.dictionary.index(
            f"{SPEECH_PREFIX}{args.speech_tokens - 1} "
        )

    def _check_tokenizer_dictionary_invariants(self, args):
        assert (
            self.eod is not None
        ), "Cannot find end-of-document symbol ({}) in tokenizer".format(
            args.end_of_document_symbol
        )

        assert self.dictionary.bos_index == 0
        assert self.tokenizer.id_to_token(0) in {"<BOS>", "<s>"}
        assert self.dictionary.pad_index == 1
        assert self.tokenizer.id_to_token(1) in {"<PAD>", "<pad>"}
        assert self.dictionary.eos_index == 2
        assert self.tokenizer.id_to_token(2) in {"<EOS>", "</s>"}
        assert self.dictionary.unk_index == 3
        assert self.tokenizer.id_to_token(3) in {"<UNK>", "<unk>"}

        assert len(self.dictionary) == self.tokenizer.get_vocab_size()
        for token in self.sentinel_tokens + [self.sentinel_end]:
            assert self.tokenizer.token_to_id(token) != 3
            assert self.dictionary.index(token) != 3

        for i in range(args.image_tokens):
            token = f"{IMAGE_PREFIX}{i} "
            assert self.tokenizer.token_to_id(token) != 3
            assert self.dictionary.index(token) != 3

        for i in range(args.speech_tokens):
            token = f"{SPEECH_PREFIX}{i} "
            assert self.tokenizer.token_to_id(token) != 3
            assert self.dictionary.index(token) != 3

        assert len(self.tokenizer.encode("I1234 I1 I56 ").ids) == 3
        if args.spm_path:
            samp_tokens = self.tokenizer.encode("1234").ids
            assert (
                len(samp_tokens) == 5
            ), f"expect digit splitting for unigram tokenizer got {samp_tokens}"

        if args.spm_path:
            n = len(self.dictionary)
            assert (
                n & (n - 1) == 0
            ), "expect dictionary size for unigram tokenizer to be an exact power of two"

        assert self.image_modality_start_token != self.dictionary.unk_index
        assert self.image_modality_end_token != self.dictionary.unk_index

        assert self.speech_modality_start_token != self.dictionary.unk_index
        assert self.speech_modality_end_token != self.dictionary.unk_index

        assert (
            self.image_modality_start_token < self.image_modality_end_token
            and self.image_modality_end_token - self.image_modality_start_token
            == args.image_tokens - 1
        ), f"IMAGE START: {self.image_modality_start_token}, IMAGE END: {self.image_modality_end_token}"

        assert (
            self.speech_modality_start_token < self.speech_modality_end_token
            and self.speech_modality_end_token - self.speech_modality_start_token
            == args.speech_tokens - 1
        )

    def __init__(self, args):
        self.args = args
        self.datasets = {}
        self.dataset_to_epoch_iter = {}

        if not has_hf_tokenizers:
            raise ImportError("Please install tokenizers with: pip install tokenizers")

        self.sentinel_end = "<eoss>"
        self.sentinel_tokens = [
            f"<sentinel:{i}>" for i in range(args.num_sentinel_tokens)
        ]

        if args.spm_path is None or args.spm_path == "":
            logger.warn(
                "By default, CM3 should be using unigram tokenization. "
                "Please double check tokenization unless you really are sure of what you are doing."
            )
            self.tokenizer = self._initialize_gpt2_tokenizer(args)
        else:
            self.tokenizer = self._initialize_unigram_tokenizer(args)

        self._initialize_metaseq_dictionary(args)
        self._initialize_eod(args)
        self._initialize_boundary_tokens(args)

        self._check_tokenizer_dictionary_invariants(args)

        logger.info(f"Dictionary Size: {len(self.dictionary)}")
        # confirm that metaseq dictionary and BPE have matching special symbols

        self.criterion_weights = torch.ones(len(self.dictionary))
        self.sentinel_tokens_ind = []
        for token in self.sentinel_tokens:
            token_index = self.dictionary.index(token)
            assert token_index != self.dictionary.unk_index
            self.sentinel_tokens_ind.append(token_index)
            self.criterion_weights[token_index] = 0.0

        self.sentinel_end_ind = self.dictionary.index(self.sentinel_end)

        self.output_dictionary = self.dictionary
