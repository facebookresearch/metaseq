"""
Streaming Distillation Language Modeling task that loads corpora considering the output format from the
inference script, and performs on-the-fly tokenization.
"""

import logging
import os
from argparse import Namespace
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import math
import torch

from metaseq.data import (
    JsonlDataset,
    data_utils,
)
from metaseq.tasks.streaming_language_modeling import (
    StreamingLanguageModelingTask,
    StreamingLanguageModelingConfig,
)
from metaseq.tasks.streaming_language_modeling import DocumentToSequenceDataset

from metaseq.tasks import register_task

logger = logging.getLogger(__name__)

DISTILLATION_MODES = {'logits_distillation', 'logprobs_distillation', 'hard_labels_distillation'}


@dataclass
class StreamingLanguageModelingDistillationConfig(StreamingLanguageModelingConfig):
    source_key: Optional[str] = field(
        default=None, metadata={"help": "Key in the JSONL input file to use as target. Default is \"src\"."}
    )
    target_key: Optional[str] = field(
        default=None, metadata={"help": "Key in the JSONL input file to use as target. Default is \"tgt\"."}
    )
    beam_results_key: Optional[str] = field(
        default="beam_results",
        metadata={"help": "Key in the JSONL that represents the beam results. Default is \"beam_results\"."}
    )
    distillation_mode: str = field(
        default="hard_labels_distillation",
        metadata={
            "help":
            "Choose the type of distillation that you want to run. Distilling from makes the student model learn from the teacher model \
        logits_distillation and requires that your input file comes with top K logits_distillation or top K logprobs_distillation for every teacher prediction. Distilling from teacher generated text directly makes the student learn from tokens \
        directly, not logits_distillation. Options are: \"logits_distillation\", \"logprobs_distillation\" and \"hard_labels_distillation\". Default is \"hard_labels_distillation\"."
        }
    )


@register_task("streaming_distillation_language_modeling", dataclass=StreamingLanguageModelingDistillationConfig)
class StreamingDistillationLanguageModelingTask(StreamingLanguageModelingTask):

    def __init__(self, args):
        super().__init__(args)

        if args.distillation_mode not in DISTILLATION_MODES:
            raise ValueError(
                f'Distillation mode not allowed. You must choose between one of the following options: {DISTILLATION_MODES}.'
            )

        self.distillation_mode = args.distillation_mode
        self.is_soft_distillation = self.distillation_mode == 'logits_distillation' or self.distillation_mode == 'logprobs_distillation'

        if args.source_key:
            self.source_key = args.source_key
        else:
            if args.distillation_mode == "hard_labels_distillation":
                self.source_key = "src"
            else:
                assert self.is_soft_distillation
                self.source_key = "src"

        if args.target_key:
            self.target_key = args.target_key
        else:
            if args.distillation_mode == "hard_labels_distillation":
                self.target_key = "tgt"
            if args.distillation_mode == "logits_distillation":
                self.target_key = "top_logits"
            if args.distillation_mode == "logprobs_distillation":
                self.target_key = "top_logprobs"

        assert self.target_key and self.source_key
        self.beam_results_key = args.beam_results_key

        self.likely_prediction_value = math.inf
        self.unlikely_prediction_value = -math.inf

        # For distillation we can't use the padding token as a mask because we can't
        # differentiate from paddings predicted by the teacher model and paddings
        # that are acting as masks tensors. However, if we use a negative target token
        # id or a large value for a token id as masks, we hit overflows during model
        # forward step. Thus our mask will be a combination of token id and prediction
        # values.
        self.mask_token_id = self.source_dictionary.pad()  # pad is 1
        self.mask_prediction_value = 0

    def build_criterion(self, args: Namespace):
        self.criterion_name = args.criterion
        if args.distillation_mode == 'hard_labels_distillation' and self.criterion_name != 'vocab_parallel_cross_entropy':
            raise ValueError(
                f'Criterion "{self.criterion_name}" not allowed. Distillation on teacher generated text requires the criterion "vocab_parallel_cross_entropy".'
            )
        elif args.distillation_mode == 'logits_distillation' and self.criterion_name != 'vocab_parallel_mse':
            raise ValueError(
                f'Criterion "{self.criterion_name}" not allowed. Distillation on teacher logits requires the criterion "vocab_parallel_mse".'
            )
        elif args.distillation_mode == 'logprobs_distillation' and self.criterion_name != 'vocab_parallel_soft_cross_entropy':
            raise ValueError(
                f'Criterion "{self.criterion_name}" not allowed. Distillation on teacher logprobs requires the criterion "vocab_parallel_soft_cross_entropy".'
            )

        return super().build_criterion(args)

    def _tokenize_source_target_json(self, json):
        """
        The purpose of the _tokenize_source_target_json method is to tokenize the source and target data in a JSON format.
        This function return a tuple containing "source_and_target_tokens" and "masked_source_and_target_tokens".
        Both "source_and_target_tokens" and "masked_source_and_target_tokens" tensors have exactly the same size, which is
        important because they will be matched internally in the model after some processing steps.
        """
        source = json[self.source_key].rstrip(" ")
        source_tokens = self.tokenizer.encode(source).ids

        prompt_eos_tokens = self.tokenizer.encode(self.args.prompt_eos_text).ids
        prompt_eos_tokens_len = len(prompt_eos_tokens)

        # Using inference output format
        if self.beam_results_key in json:
            target = json[self.beam_results_key]
            # Use only the first beam result.
            # TODO - add support for multiple beams
            if isinstance(target, list):
                target = json[self.beam_results_key][0]
        else:
            target = json

        target = target[self.target_key]

        if self.distillation_mode == 'hard_labels_distillation':
            target = target.rstrip(" ")
            target_tokens = self.tokenizer.encode(target).ids

            if len(target_tokens) > self.args.tokens_per_sample:
                target_tokens = data_utils.truncate_target(target_tokens, self.args.tokens_per_sample)

            # Limit by max sequence length (block length) so loss is computed for every block
            max_context_length = self.args.tokens_per_sample - len(target_tokens) - prompt_eos_tokens_len
            source_tokens, _ = data_utils.truncate_source(
                source_tokens, max_context_length, self.args.truncation, prompt_eos_tokens_len
            )
            source_tokens_len = len(source_tokens)

            source_and_target_tokens = torch.LongTensor(
                source_tokens + prompt_eos_tokens + target_tokens + [self.eod]
            )  # Shape (num_source_tokens+num_masked_source_and_target_tokens)
            masked_source_and_target_tokens = torch.clone(
                source_and_target_tokens
            )  # Shape (num_source_tokens+num_masked_source_and_target_tokens)
            masked_source_and_target_tokens[:(source_tokens_len + prompt_eos_tokens_len)] = self.mask_token_id
        elif self.is_soft_distillation:
            assert isinstance(target, list), 'Expected a list of logits or logprobs in the target key'
            score_key = 'logit_score' if self.distillation_mode == 'logits_distillation' else 'logprob_score'

            # target_list has <seq_len> items. Each item is a sublist that represents K predictions of
            # the teacher model in the format [token_id, score].
            target_list, num_k_predictions = self._get_target_tokens_and_num_k_predictions(target, score_key)

            if len(target_list) > self.args.tokens_per_sample:
                target_list = data_utils.truncate_target(target_list, self.args.tokens_per_sample)

            # source_tokens_list is a list of source tokens, in which each sublist represents K predictions of
            # the teacher model in the format [token_id, score]. The original source token will be
            # couple with a large prediction value, and we add dummy BOS tokens with a small prediction rate
            # to complete the K predictions.
            # masked_source_tokens_list is the same list but all source tokens are masked with the <pad> token.
            source_tokens_list, masked_source_tokens_list = self._get_source_tokens_and_masked_source_tokens_with_prediction_scores(
                source_tokens, num_k_predictions
            )
            prompt_eos_tokens, masked_prompt_eos_tokens = self._get_source_tokens_and_masked_source_tokens_with_prediction_scores(
                prompt_eos_tokens, num_k_predictions
            )

            # Limit by max sequence length (block length) so loss is computed for every block
            max_context_length = self.args.tokens_per_sample - len(target_list) - prompt_eos_tokens_len
            source_tokens_list, masked_source_tokens_list = data_utils.truncate_source(
                source_tokens_list, max_context_length, self.args.truncation, prompt_eos_tokens_len, masked_source_tokens_list
            )
            source_tokens_len = len(source_tokens_list)
            assert source_tokens_len <= max_context_length

            source_and_target_tokens, masked_source_and_target_tokens = self._get_concatenated_tokens_with_prediction_scores(
                source_tokens_list, masked_source_tokens_list, source_tokens_len, prompt_eos_tokens, masked_prompt_eos_tokens,
                target_list, num_k_predictions
            )  # Shape ((num_source_tokens+num_masked_source_and_target_tokens), num_top_k_predictions, 2) where 2 stands for tokens and scores
        else:
            raise RuntimeError(f'Unexpected distillation mode: "{self.distillation_mode}".')

        # "source_and_target_tokens" is composed by: source_tokens_list + target_list + EOS
        # "masked_source_and_target_tokens" is composed by: masked_source_tokens_list + target_list + EOS
        return (source_and_target_tokens, masked_source_and_target_tokens)

    def _get_target_tokens_and_num_k_predictions(self, target, score_key):
        """
        This method reads the target input (a list of top_logprobs) and creates a new list holding
        the pairs of K token ids/predictions for each token of the sequence.
        It also handles the case of having an Open AI teacher model that doesn't output token ids.
        """
        target_list = list()

        # Fill tensor with values to be used as target during
        # distillation
        num_k_predictions = -1
        for predicted_tokens in target:
            predicted_tokens_list = list()

            if num_k_predictions == -1:
                # Initialize K
                num_k_predictions = len(predicted_tokens)
            else:
                # Make sure we have K predictions for every token
                assert len(predicted_tokens) == num_k_predictions

            scores = [token_info['logprob_score'] for token_info in predicted_tokens.values()]
            max_score = max(scores)

            for token_txt, token_info in predicted_tokens.items():
                token_id = token_info['token_id']
                token_score = token_info[score_key]
                # OpenAI teacher models don't output token ids. We need to get
                # the token id of our own model tokenizer.
                if token_id is None:
                    ids = self.tokenizer.encode(token_txt).ids
                    if len(ids) > 1:
                        # Some tokens (e.g.: </s>) are tokenized as multiple tokens.
                        # We need to use a different method to tokenize it.
                        id = self.tokenizer.token_to_id(token_txt)
                        if id is None:
                            if token_txt.startswith("bytes:"):
                                # use the mask token id and mask prediction score to ignore this token
                                # during loss calculation
                                ids = [self.mask_token_id]
                                token_score = self.unlikely_prediction_value
                            elif token_txt.strip() == '':  # token_txt has only multiple space chars
                                ids = [ids[0]]  # use the first space token only
                            else:
                                raise RuntimeError(f"Unexpected token '{token_txt}' can't be converted to known token id.")
                        else:
                            ids = [id]  # use tokenizer encode output
                    assert ids is not None and len(ids) == 1
                    token_id = ids[0]
                predicted_tokens_list.append([token_id, token_score])

            # sort predictions by logprob_score (largest to smallest)
            predicted_tokens_list.sort(key=lambda row: row[1], reverse=True)
            target_list.append(predicted_tokens_list)

        return target_list, num_k_predictions

    def _get_source_tokens_and_masked_source_tokens_with_prediction_scores(self, source_tokens, num_k_predictions):
        """
        Reads the source_tokens input and creates a list of pairs for each token assigning masked prediction
        values to the source tokens and filling the other (k-1) positions with the BOS token. Prediction values
        for source are not used, but required for computation. Thus they can all be masked. It also creates a
        version of this list in which the source tokens are masked using the mask prediction token.
        """
        # Fill tensor with values to be used as source during distillation.
        source_tokens_list = list()
        masked_source_tokens_list = list()
        for source_id in source_tokens:
            cur_source_item = [[source_id, self.mask_prediction_value]
                               ] + ([[self.dictionary.bos_index, self.mask_prediction_value]] * (num_k_predictions - 1))
            source_tokens_list.append(cur_source_item)
            cur_masked_source_item = [[self.mask_token_id, self.mask_prediction_value]] * (num_k_predictions)
            masked_source_tokens_list.append(cur_masked_source_item)
        assert len(source_tokens_list) == len(source_tokens)
        assert len(masked_source_tokens_list) == len(source_tokens)

        return source_tokens_list, masked_source_tokens_list

    def _get_concatenated_tokens_with_prediction_scores(
        self, source_tokens_list, masked_source_tokens_list, source_tokens_len, prompt_eos_tokens, masked_prompt_eos_tokens,
        target_list, num_k_predictions
    ):
        """
        Concatenates the source_tokens_list and target_list and add an extra EOS token
        """
        last_item = [
            [[self.eod, self.likely_prediction_value]] +
            ([[self.dictionary.bos_index, self.unlikely_prediction_value]] * (num_k_predictions - 1))
        ]
        source_and_target_tokens = torch.FloatTensor(source_tokens_list + prompt_eos_tokens + target_list + last_item)
        target_tokens = torch.FloatTensor(masked_source_tokens_list + masked_prompt_eos_tokens + target_list + last_item)
        assert source_and_target_tokens.shape == target_tokens.shape
        assert torch.all(
            source_and_target_tokens[source_tokens_len + len(prompt_eos_tokens):] == target_tokens[source_tokens_len +
                                                                                                   len(prompt_eos_tokens):]
        )

        return source_and_target_tokens, target_tokens

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
        Each jsonl entry is a dict with "prompt_text" and "target_text" keys. Loss is computed
        only on the target tokens.
        Args:
            split (str): name of the split (e.g., train, valid, valid1, test)
        """
        # This function reads a bunch of jsonl files, concats them together,
        # shuffles them, then chunks them into blocks of tokens (e.g., 2048).

        # determine number of shards for this split shards = {}
        cur_shard_str = self.get_shard_str(epoch, split)

        # concatenate any jsonl files that are part of the shard
        datasets, corpora = [], []
        for file in sorted(os.listdir(os.path.join(self.args.data, split, cur_shard_str))):
            if not file.endswith(".jsonl"):
                continue
            datasets.append(
                JsonlDataset(
                    path=os.path.join(self.args.data, split, cur_shard_str, file),
                    tokenizer=self._tokenize_source_target_json,
                )
            )
            corpora.append(os.path.splitext(file)[0])
        assert len(datasets) > 0

        if (self.args.multicorpus_sampling_alpha != 1 or self.args.multicorpus_sampling_maximum > 0):
            datasets = self._alpha_sampling(datasets, corpora, epoch)

        dataset = torch.utils.data.ConcatDataset(datasets)

        break_mode = 'eos_pad_8' if self.is_soft_distillation else self.args.sample_break_mode
        if self.args.sample_break_mode != break_mode:
            logger.warning(
                f"Unsupported sample_break_mode for distillation: {self.args.sample_break_mode}. Updating it to {break_mode}."
            )

        self.datasets[split] = DocumentToSequenceDataset(
            dataset,
            # We generate blocks with one extra token, so that we have a target
            # for the final input token. This results in slight data loss.
            block_size=self.args.tokens_per_sample + 1,
            break_mode=break_mode,
            # we drop the remainder block during training
            drop_last=(split == "train"),
            padding_idx=self.source_dictionary.pad(),
            seed=self.args.seed,
            source_target=True,
            mask_token_id=self.mask_token_id,
            mask_prediction_value=self.mask_prediction_value,
        )

    def _collate_fn(self, items: List[Dict[str, Any]]):
        """
        The overall goal of the collate_tokens function is to take a list of tensors and
        transform them into a single padded tensor. This can be useful when working with
        batches of data where the individual data points may have different lengths. By
        padding the data to a consistent length, it can be more easily processed in parallel.
        """

        # StreamingTokenBlockDataset returns None as filler
        if len([x for x in items if x is not None]) == 0:
            return {}

        source_tokens = data_utils.collate_tokens(
            [x["src_block"] for x in items if x is not None],
            pad_idx=self.mask_token_id,
            pad_to_bsz=self.args.batch_size,
            mask_prediction_value=self.mask_prediction_value,
        )  # (batch_size, seq_len+1, num_top_k_predictions, 2)
        target_tokens = data_utils.collate_tokens(
            [x["tgt_block"] for x in items if x is not None],
            pad_idx=self.mask_token_id,
            pad_to_bsz=self.args.batch_size,
            mask_prediction_value=self.mask_prediction_value,
        )  # (batch_size, seq_len+1, num_top_k_predictions, 2)

        # Generate model input tokens
        if self.distillation_mode == 'hard_labels_distillation':
            input = source_tokens[:, :-1].contiguous()  # (batch_size, seq_len)
        elif self.is_soft_distillation:
            input = source_tokens[:, :-1, 0, 0].long().contiguous()  # (batch_size, seq_len)
        else:
            raise RuntimeError(f'Unexpected distillation mode {self.distillation_mode}')

        # Generate targets including tokens and logprobs or logits for distillation
        if self.distillation_mode == "hard_labels_distillation":
            target = target_tokens[:, 1:].clone().contiguous()  # (batch_size, seq_len)
        elif self.is_soft_distillation:
            # We need to target_tokens in two tensors because the sample
            # preparation step will transform the predictions tensor in float16, and if
            # the tokens are together in the same tensor, they will be rounded to 16 bits
            # with mismatches, leading to wrong results in the MSE loss calculation.
            target = {
                'target_tokens': target_tokens[:, 1:, :,
                                               0].clone().long().contiguous(),  # (batch_size, seq_len, num_top_k_predictions)
                'target_predictions': target_tokens[:, 1:, :,
                                                    1].clone().contiguous(),  # (batch_size, seq_len, num_top_k_predictions)
            }
            # Erase duplicated data from memory
            del target_tokens
            torch.cuda.empty_cache()

        ids = torch.cat([x["ids"] for x in items if x is not None])
        if ids.numel() != torch.unique(ids).numel():
            n_duplicate = ids.numel() - torch.unique(ids).numel()
            logger.error(f"found {n_duplicate}/{ids.numel()} duplicate document IDs in the same batch!")

        if self.is_soft_distillation:
            max_mask = target['target_predictions'] == math.inf
            min_mask = target['target_predictions'] == -math.inf
            # For MSE, update inf and -inf to non-infinite likely and unlikely prediction values to avoid
            # overflow during loss calculation
            if self.criterion_name == 'vocab_parallel_mse':
                self.likely_prediction_value = torch.max(target['target_predictions'][~max_mask])
                self.unlikely_prediction_value = torch.min(target['target_predictions'][~min_mask])
                assert self.likely_prediction_value != math.inf and self.unlikely_prediction_value != -math.inf
                target['target_predictions'][max_mask] = self.likely_prediction_value
                target['target_predictions'][min_mask] = self.unlikely_prediction_value
            # For Soft Cross entropy, update inf to 0 and use -inf as we use target.exp() in the loss calculation
            elif self.criterion_name == 'vocab_parallel_soft_cross_entropy':
                self.likely_prediction_value = 0
                target['target_predictions'][max_mask] = self.likely_prediction_value

            # The number of target tokens is the number of non-masked tokens. In hard distillation, we can consider the mask as just
            # the token ids because the input data would not show pad tokens in the output. However, for soft distillation, our target
            # is a combination of multiple tokens and predictions for each of them. Thus our mask is a combination of a mask id for tokens
            # and a mask value for predictions.
            # In other words, during soft distillation the assumption that a padding token is a masked token is not true because pad
            # tokens may be included in model predictions. We identify which pad tokens were normal outputs vs those which were
            # artifically added by checking if the associated prediction value is the masked prediction value
            masked_targets = target["target_tokens"].eq(self.mask_token_id
                                                        ) & target["target_predictions"].eq(self.mask_prediction_value)
            # this sum considers only a single prediction per token to be consistent with the num of tokens of input
            ntokens_target = (~masked_targets).sum(dim=1)  # (bs_size, num_top_k_predictions)
            ntokens_target = ntokens_target[:, 0].sum()
        elif self.distillation_mode == 'hard_labels_distillation':
            masked_targets = target.eq(self.mask_token_id)
            ntokens_target = target.ne(self.mask_token_id).sum()

        return {
            "id": ids,
            "net_input": {
                "src_tokens": input,
            },
            "target": target,
            "nsentences": input.size(0),
            "ntokens": input.ne(self.mask_token_id).sum(),
            "ntokens_target": ntokens_target,
            "distillation_mode": self.distillation_mode,
            "masked_targets": masked_targets,
        }
