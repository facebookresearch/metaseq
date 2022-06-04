# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from dataclasses import dataclass, field

import torch

from metaseq.data import Dictionary
from metaseq.tasks import register_task
from metaseq.tasks.language_modeling_inference_for_models_trained_with_streaming import (
    LanguageModelingInferenceForModelsTrainedWithStreamingConfig as LMInferenceStreamingConfig,
    LanguageModelingInferenceForModelsTrainedWithStreamingTask as LMInferenceStreamingTask)
from metaseq.tasks.streaming_CM3_language_modeling import IMAGE_PREFIX, SPEECH_PREFIX

logger = logging.getLogger(__name__)

try:
    from tokenizers import ByteLevelBPETokenizer

    has_hf_tokenizers = True
except ImportError:
    has_hf_tokenizers = False


@dataclass
class CM3LanguageModelingInferenceForModelsTrainedWithStreamingConfig(LMInferenceStreamingConfig):
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
    "cm3_language_modeling_inference_for_models_trained_with_streaming",
    dataclass=CM3LanguageModelingInferenceForModelsTrainedWithStreamingConfig,
)
class CM3LanguageModelingInferenceForModelsTrainedWithStreamingTask(LMInferenceStreamingTask):
    """
    This class is specially developed for inference of models trained
    with the new StreamingLanguageModeling but follows closely the language_modeling implementation.
    """

    def __init__(self, args):
        self.args = args
        self.datasets = {}
        self.dataset_to_epoch_iter = {}

        if not has_hf_tokenizers:
            raise ImportError("Please install tokenizers with: pip install tokenizers")

        tokenizer = ByteLevelBPETokenizer.from_file(
            args.vocab_filename,
            args.merges_filename
        )
        self.sentinel_tokens = [f"<sentinel:{i}>" for i in range(args.num_sentinel_tokens)]
        self.sentinel_end = "<eoss>"

        tokenizer.add_special_tokens([f"{IMAGE_PREFIX}{x} " for x in range(args.image_tokens)])
        tokenizer.add_special_tokens([f"{SPEECH_PREFIX}{x} " for x in range(args.speech_tokens)])
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

        self.criterion_weights = torch.ones(len(self.dictionary))
        self.sentinel_tokens_ind = []
        for token in self.sentinel_tokens:
            token_index = self.dictionary.index(token)
            assert token_index != self.dictionary.unk_index
            self.sentinel_tokens_ind.append(token_index)
            self.criterion_weights[token_index] = 0.0

        self.sentinel_end_ind = self.dictionary.index(self.sentinel_end)
        assert token_index != self.sentinel_end_ind
        self.output_dictionary = self.dictionary
