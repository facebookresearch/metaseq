# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import ast
import copy
import logging
import os
import time
from argparse import Namespace
from typing import Any, Dict, Iterator, List, Optional

import numpy as np
import torch
from omegaconf import open_dict
from torch import nn

from metaseq import checkpoint_utils, tasks
from metaseq import utils
from metaseq.data import encoders
from metaseq.dataclass.configs import MetaseqConfig
from metaseq.dataclass.utils import convert_namespace_to_omegaconf
from metaseq.distributed import fsdp_enable_wrap, fsdp_wrap
from metaseq.distributed.utils import (
    get_data_parallel_rank,
    get_data_parallel_world_size,
)

logger = logging.getLogger(__name__)


def from_pretrained(
    model_name_or_path,
    checkpoint_file="model.pt",
    data_name_or_path=".",
    archive_map=None,
    **kwargs,
):
    post_build_model_hook = kwargs.get("post_build_model_hook", None)

    from metaseq import checkpoint_utils, file_utils

    if archive_map is not None:
        if model_name_or_path in archive_map:
            model_name_or_path = archive_map[model_name_or_path]
        if data_name_or_path is not None and data_name_or_path in archive_map:
            data_name_or_path = archive_map[data_name_or_path]

        # allow archive_map to set default arg_overrides (e.g., tokenizer, bpe)
        # for each model
        if isinstance(model_name_or_path, dict):
            for k, v in model_name_or_path.items():
                if k == "checkpoint_file":
                    checkpoint_file = v
                elif (
                    k != "path"
                    # only set kwargs that don't already have overrides
                    and k not in kwargs
                ):
                    kwargs[k] = v
            model_name_or_path = model_name_or_path["path"]

    model_path = file_utils.load_archive_file(model_name_or_path)

    # convenience hack for loading data and BPE codes from model archive
    if data_name_or_path.startswith("."):
        kwargs["data"] = os.path.abspath(os.path.join(model_path, data_name_or_path))
    else:
        kwargs["data"] = file_utils.load_archive_file(data_name_or_path)
    for file, arg in {
        "code": "bpe_codes",
        "bpecodes": "bpe_codes",
        "sentencepiece.bpe.model": "sentencepiece_model",
        "merges.txt": "bpe_merges",
        "vocab.json": "bpe_vocab",
    }.items():
        path = os.path.join(model_path, file)
        if os.path.exists(path):
            kwargs[arg] = path

    if "user_dir" in kwargs:
        utils.import_user_module(argparse.Namespace(user_dir=kwargs["user_dir"]))

    def _build_fn(train_cfg, task):
        if post_build_model_hook:
            return post_build_model_hook(task.build_model(train_cfg.model), task)
        else:
            return task.build_model(train_cfg.model)

    models, args, task = checkpoint_utils.load_model_ensemble_and_task(
        [os.path.join(model_path, cpt) for cpt in checkpoint_file.split(os.pathsep)],
        arg_overrides=kwargs,
        suffix=kwargs.get("suffix", ""),
        build_model_hook=lambda cfg, task: _build_fn(cfg, task),
    )

    return {
        "args": args,
        "task": task,
        "models": models,
    }


class BPEHubInterface(object):
    """PyTorch Hub interface for Byte-Pair Encoding (BPE)."""

    def __init__(self, bpe, **kwargs):
        super().__init__()
        args = argparse.Namespace(bpe=bpe, **kwargs)
        self.bpe = encoders.build_bpe(args)
        assert self.bpe is not None

    def encode(self, sentence: str) -> str:
        return self.bpe.encode(sentence)

    def decode(self, sentence: str) -> str:
        return self.bpe.decode(sentence)


class TokenizerHubInterface(object):
    """PyTorch Hub interface for tokenization."""

    def __init__(self, tokenizer, **kwargs):
        super().__init__()
        args = argparse.Namespace(tokenizer=tokenizer, **kwargs)
        self.tokenizer = encoders.build_tokenizer(args)
        assert self.tokenizer is not None

    def encode(self, sentence: str) -> str:
        return self.tokenizer.encode(sentence)

    def decode(self, sentence: str) -> str:
        return self.tokenizer.decode(sentence)


class GeneratorInterface:
    """
    PyTorch Hub interface for generating sequences from a pre-trained
    translation or language model.
    """

    def __init__(self, cfg: MetaseqConfig):
        self.cfg = cfg
        if isinstance(self.cfg, Namespace):
            self.cfg = convert_namespace_to_omegaconf(self.cfg)

    def load_model(self):
        utils.import_user_module(self.cfg.common)

        # Fix seed for stochastic decoding
        if (
            self.cfg.common.seed is not None
            and not self.cfg.generation.no_seed_provided
        ):
            np.random.seed(self.cfg.common.seed)
            utils.set_torch_seed(self.cfg.common.seed)

        # Setup task, e.g., translation
        task = tasks.setup_task(self.cfg.task)

        def _build_model(cfg, task):
            model = task.build_model(cfg.model).half().cuda()
            model.make_generation_fast_()
            return fsdp_wrap(model)

        # Load the model
        overrides = ast.literal_eval(self.cfg.common_eval.model_overrides)
        logger.info("loading model(s) from {}".format(self.cfg.common_eval.path))
        with fsdp_enable_wrap(
            self.cfg.distributed_training,
            use_sharded_state=self.cfg.distributed_training.use_sharded_state,
        ):
            models, _model_args, _task = checkpoint_utils.load_model_ensemble_and_task(
                utils.split_paths(self.cfg.common_eval.path),
                arg_overrides=overrides,
                task=task,
                suffix=self.cfg.checkpoint.checkpoint_suffix,
                strict=(self.cfg.checkpoint.checkpoint_shard_count == 1),
                num_shards=self.cfg.checkpoint.checkpoint_shard_count,
                build_model_hook=_build_model,
            )
        # Set dictionaries
        src_dict = task.source_dictionary
        tgt_dict = task.target_dictionary

        # Handle tokenization and BPE
        bpe = task.build_bpe(self.cfg.bpe)

        # Set state
        self.bpe = bpe
        self.task = task
        self.models = models
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict
        return models

    def generate(
        self,
        inputs: List[List[int]],
        min_tokens: List[int] = None,
        max_tokens: List[int] = None,
        temperature: float = 1.0,
        top_p: float = -1.0,
        logprobs: int = 0,
        n: int = 1,
        best_of: Optional[int] = None,
        echo: bool = False,
        stop: Optional[List[int]] = None,
        seed: Optional[int] = None,
        use_cuda: bool = True,
    ):
        """
        Generate from sequences.
        Parameters match those of the OpenAI API.
        https://beta.openai.com/docs/api-reference/completions/create
        inputs: a list of pre-tokenized prompts
        min_tokens: blocks EOS until at least this many tokens is provided
        max_tokens: forces EOS after this many tokens
        temperature: softmax temperature
        top_p: nucleus probability
        log_probs: return this cutoff of the probability distribution
        n: beam size
        best_of: number of beams to return. must be <= n
        echo: if true, returned text/tokens/scores includes the prompt.
            This is useful for getting PPL evaluations.
        stop: a list of terminating tokens
        seed: an integer if desired
        use_cuda: should we use GPUs.
        """
        if seed:
            utils.set_torch_seed(seed)
        start_time = time.time()
        total_generation_time = 0

        # Initialize generator
        if not best_of:
            best_of = n
        assert best_of >= n
        self.cfg.generation.sampling_topp = top_p if top_p > 0 else -1
        self.cfg.generation.sampling = top_p > 0.0
        self.cfg.generation.beam = best_of
        if temperature > 0:
            self.cfg.generation.temperature = temperature
        elif temperature == 0:
            self.cfg.generation.sampling = False
            self.cfg.generation.temperature = 1.0
            self.cfg.generation.sampling_topp = -1
        elif temperature < 0:
            raise ValueError("temperature must be >= 0 and <= 1")

        MAX_SEQ_LEN = utils.resolve_max_positions(
            self.task.max_positions(), *[model.max_positions() for model in self.models]
        )

        # TODO(roller): simplify
        retval = []
        tokens = [torch.LongTensor(t) for t in inputs]
        lengths = [len(t) for t in inputs]
        batches = self.task.get_batch_iterator(
            dataset=self.task.build_dataset_for_inference(tokens, lengths),
            max_tokens=None,
            max_sentences=None,
            max_positions=None,
            ignore_invalid_inputs=False,
        ).next_epoch_itr(shuffle=False)
        for batch in batches:
            src_tokens = batch["net_input"]["src_tokens"]
            src_lengths = batch["net_input"]["src_lengths"]
            batchsize = src_tokens.size(0)

            # set generation args
            # prevent us from ever generating past our max sequence length
            if max_tokens is None:
                max_tokens = [MAX_SEQ_LEN] * batchsize
            if min_tokens is None:
                min_tokens = [0] * batchsize
            total_max_tokens = min(
                MAX_SEQ_LEN, max(max_tokens) + src_lengths.max().item()
            )
            total_min_tokens = max(min_tokens) + src_lengths.max().item()
            self.cfg.generation.min_len = total_min_tokens
            self.cfg.generation.max_len_b = total_max_tokens
            self.cfg.generation.max_len_a = 0

            logger.info(f"Preparing generator with settings {self.cfg.generation}")
            need_logprobs = True if logprobs > 0 else False
            generator = self.task.build_generator(
                self.models,
                self.cfg.generation,
                extra_gen_cls_kwargs={"stop": stop, "need_logprobs": need_logprobs},
            )

            # okay actually generate
            logger.info(f"Executing generation on input tensor size {src_tokens.shape}")
            if use_cuda:
                batch = utils.move_to_cuda(batch)

            translate_start_time = time.time()
            translations = self.task.inference_step(generator, self.models, batch)
            translate_time = time.time() - translate_start_time
            total_generation_time += translate_time

            all_tokens = translations["tokens"].cpu()[: len(inputs)]
            all_scores = translations["scores"].cpu()[: len(inputs)]
            if logprobs > 0:
                all_distributions = translations["distributions"].cpu()[: len(inputs)]
            else:
                all_distributions = None

            # actually turn everything into strings
            for i in range(all_tokens.size(0)):
                beams = []
                for j in range(best_of):
                    # first beam is always the highest scoring
                    tokens = all_tokens[i, j].tolist()
                    scores = all_scores[i, j].tolist()
                    distributions = all_distributions[i, j] if logprobs > 0 else None
                    tokens, scores, distributions = GeneratorInterface._filter_special(
                        tokens, scores, distributions
                    )
                    prompt_len = lengths[i]
                    if echo:
                        # don't cut off prompt
                        tokens = tokens[: prompt_len + max_tokens[i]]
                        scores = scores[: prompt_len + max_tokens[i]]
                        if logprobs > 0:
                            distributions = distributions[: prompt_len + max_tokens[i]]
                    else:
                        # cut off prompt
                        tokens = tokens[prompt_len:][: max_tokens[i]]
                        scores = scores[prompt_len:][: max_tokens[i]]
                        if logprobs > 0:
                            distributions = distributions[prompt_len:][: max_tokens[i]]
                    # turn it into a string
                    text = self.bpe.bpe.decode(tokens)
                    # re-encode it so we get offsets
                    token_offsets = [s for s, e in self.bpe.bpe.encode(text).offsets]

                    result = {
                        "text": self.bpe.bpe.decode(tokens),
                        "tokens": [self.bpe.bpe.decode([t]) for t in tokens],
                        # text offset is useful for cutting off prompts or prefixes
                        # or evaluating PPL on just a subset of tokens
                        "text_offset": token_offsets,
                        "token_scores": scores,
                    }
                    if logprobs > 0:
                        # final result is a List[Dict[str, float]]
                        # where each item in the list corresponds to a token in the
                        # sequence, and the dict provides the probabilities of the
                        # top-k tokens at that timestep.
                        out_logprobs = []
                        all_top_toks, all_top_scores = distributions.topk(
                            k=logprobs, dim=-1
                        )
                        for top_scores, top_toks in zip(all_top_toks, all_top_scores):
                            lp = {
                                self.bpe.bpe.decode([t.item()]): s.item()
                                for t, s in zip(top_toks, top_scores)
                            }
                            out_logprobs.append(lp)
                        result["top_logprobs"] = out_logprobs
                    else:
                        result["top_logprobs"] = None

                    beams.append(result)
                retval.append(beams)

        logger.info(
            "Total time: {:.3f} seconds; generation time: {:.3f}".format(
                time.time() - start_time, total_generation_time
            )
        )
        return retval

    @staticmethod
    def _filter_special(
        tokens: List[int],
        scores: List[float],
        distributions,
        pad_token: int = 1,
    ):
        """
        Cut off tokens after finding a special tokens.
        """

        # tokens is a 1D list of token IDs of length seqlen
        # scores is a 1D list of log-probability scores for those tokens (length seqlen)
        # distributions (optional) is a seqlen x vocab_size tensor corresponding to
        # the full distribution of predictions at each timestep

        output = []
        mask = []
        for t, s in zip(tokens, scores):
            if t == pad_token:
                # simply skip pads
                mask.append(False)
                continue
            if t <= 3:
                # and other special tokens should end things
                break
            mask.append(True)
            output.append((t, s))
        new_tokens, new_scores = zip(*output)

        # cut off at stop and drop pads
        if distributions is not None:
            distributions = distributions[: len(mask)][mask]
            distributions = distributions[: len(output)]
        return new_tokens, new_scores, distributions
