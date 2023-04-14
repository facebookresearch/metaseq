# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Reference:
# https://github.com/facebookresearch/fairseq/blob/main/fairseq/sequence_generator.py

import logging
import math
from typing import Dict, List, Optional, Tuple
from metaseq.data.dictionary import Dictionary
import sys

import torch
import torch.nn as nn
from torch import Tensor

from metaseq import utils
from metaseq.data import data_utils
from metaseq.models import BaseDecoder

logger = logging.getLogger(__name__)


class SequenceGenerator(nn.Module):
    def __init__(
        self,
        models,
        tgt_dict: Dictionary,
        beam_size: int = 1,
        max_len_a: int = 0,
        max_len_b: int = 200,
        min_len: int = 1,
        temperature: float = 1.0,
        need_logprobs: bool = False,
        stop: Optional[List[int]] = None,
        topp: float = -1,
        profile=False,
        omega_bound: float = 0.3,
        lambda_decay: float = -1,
        alpha_presence: float = 0.0,
        alpha_frequency: float = 0.0,
        alpha_presence_src: float = 0.0,
        alpha_frequency_src: float = 0.0,
        alpha_src_penalty_end_idx: int = -1,
    ):
        """Generates translations of a given source sentence.

        For discussion of repetition penalties, see
        https://github.com/facebookresearch/metaseq/pull/306

        Args:
            models: ensemble of models
            beam_size (int, optional): beam width (default: 1)
            max_len_a/b (int, optional): generate sequences of maximum length
                ax + b, where x is the source length
            min_len (int, optional): the minimum length of the generated output
                (not including end-of-sentence)
            temperature (float, optional): temperature, where values
                >1.0 produce more uniform samples and values <1.0 produce
                sharper samples (default: 1.0)
            stop: An optional list of other tokens that can cause early termination.
            need_logprobs (bool): Return additional log-prob distributions for
                every timestep of the search.
            omega_bound (float, optional): lower p-bound when decaying nucleus
            lamda_decay (float, optional): decay factor for decaying p in nucleus sampling
                default -1 disables.
            alpha_presence (float, optional): repetition penalty for token presence in
                current generation
            alpha_frequency (float, optional): repetition penalty for token frequency in
                current generation
            alpha_presence_src (float, optional): repetition penalty for token presence in
                incoming context
            alpha_frequency_src (float, optional): repetition penalty for token frequency in
                incoming context
            alpha_src_penalty_end_idx (int, optional): index of the last token in incoming
                context to consider for repetition penalty
        """
        super().__init__()
        self.model = models[0]
        self.tgt_dict = tgt_dict
        self.pad = tgt_dict.pad()
        self.unk = tgt_dict.unk()
        self.eos = tgt_dict.eos()
        self.vocab_size = len(tgt_dict)
        self.beam_size = beam_size
        # the max beam size is the dictionary size - 1, since we never select pad
        self.beam_size = min(beam_size, self.vocab_size - 1)
        self.max_len_a = max_len_a
        self.max_len_b = max_len_b
        self.min_len = min_len
        self.need_logprobs = need_logprobs
        self.stop = stop if stop is not None else []
        if topp is None:
            topp = 0.0
        self.sampling_topp = max(0, topp)
        self.temperature = temperature
        assert temperature >= 0, "--temperature must be >=0"

        self.model.eval()
        self.profile = profile

        # factual nucleus
        buffer = torch.zeros(beam_size)
        self.sampling_topp = max(0, topp)
        self.sampling_topp_tensor = (
            buffer.clone().fill_(self.sampling_topp).unsqueeze(1)
        )
        self.init_p = self.sampling_topp_tensor.clone()
        self.lambda_decay = lambda_decay
        self.omega_bound = torch.tensor([omega_bound])
        self.toks_since_reset = buffer.clone()
        self.full_stop_list = torch.tensor([tgt_dict.index(w) for w in [".", "?", "!"]])

        # repetition reduction
        self.alpha_presence = alpha_presence
        self.alpha_frequency = alpha_frequency
        self.alpha_presence_src = alpha_presence_src
        self.alpha_frequency_src = alpha_frequency_src
        self.alpha_src_penalty_end_idx = alpha_src_penalty_end_idx

    def cuda(self):
        self.model.cuda()
        return self

    @torch.no_grad()
    def forward(
        self,
        sample: Dict[str, Dict[str, Tensor]],
        prefix_tokens: Optional[Tensor] = None,
        bos_token: Optional[int] = None,
    ):
        """Generate a batch of translations."""
        return self._generate(sample, prefix_tokens, bos_token=bos_token)

    @torch.no_grad()
    def generate(self, models, sample: Dict[str, Dict[str, Tensor]], **kwargs):
        """Generate translations. Match the api of other metaseq generators."""
        return self._generate(sample, **kwargs)

    def _generate(
        self,
        sample: Dict[str, Dict[str, Tensor]],
        prefix_tokens: Optional[Tensor] = None,
        bos_token: Optional[int] = None,
    ):
        """
        Args:
            sample (dict): batch
            prefix_tokens (torch.LongTensor, optional): force decoder to begin
                with these tokens
            bos_token (int, optional): beginning of sentence token
                (default: self.eos)
        """
        incremental_states = torch.jit.annotate(
            Dict[str, Dict[str, Optional[Tensor]]], {}
        )
        net_input = sample["net_input"]

        if "src_tokens" in net_input:
            src_tokens = net_input["src_tokens"]
            # length of the source text being the character length except EndOfSentence and pad
            src_lengths = (
                (src_tokens.ne(self.eos) & src_tokens.ne(self.pad)).long().sum(dim=1)
            )
        elif "source" in net_input:
            src_tokens = net_input["source"]
            src_lengths = (
                net_input["padding_mask"].size(-1) - net_input["padding_mask"].sum(-1)
                if net_input["padding_mask"] is not None
                else torch.tensor(src_tokens.size(-1)).to(src_tokens)
            )
        else:
            raise Exception("expected src_tokens or source in net input")

        # bsz: total number of sentences in beam
        # Note that src_tokens may have more than 2 dimensions (i.e. audio features)
        bsz, src_len = src_tokens.size()[:2]
        beam_size = self.beam_size

        max_len = min(self.model.max_decoder_positions(), self.max_len_b or 1e99)
        min_len = min(max_len, self.min_len or 0)

        assert (
            min_len <= max_len
        ), "min_len cannot be larger than max_len, please adjust these!"

        # placeholder of indices for bsz * beam_size to hold tokens and accumulative scores
        new_order = torch.arange(bsz).view(-1, 1).repeat(1, beam_size).view(-1)
        new_order = new_order.to(src_tokens.device).long()

        # initialize buffers
        (
            scores,
            tokens,
            count_tokens,
            count_tokens_src,
        ) = self._initialize_generation_buffers(
            bsz, beam_size, int(max_len), src_tokens
        )

        # notes:
        # - scores \in FloatTensor(bsz * beam_size, max_len)
        # - tokens \in LongTensor(bsz * beam_size, max_len)
        # - count_tokens \in LongTensor(bsz * beam_size, vocab_size)
        # - count_tokens_src \in LongTensor(bsz * beam_size, vocab_size)
        # - src_tokens \in LongTensor(bsz, prompt_len)
        # - all_lprobs \in FloatTensor(bsz * beam_size, max_len, vocab_size)
        #   is the next word distribution at every timestep

        if self.need_logprobs:
            # lprobs are costly for memory, so only compute them if we have to
            all_lprobs = (
                torch.zeros(bsz * beam_size, max_len, self.vocab_size)
                .to(src_tokens)
                .float()
            )

        # first forward through all the fixed tokens with forced decoding we'll
        # need to handle normalization and prep for bookkeeping of incremental
        # decoding
        start_step = src_tokens.shape[1]
        # set all the forced tokens
        tokens[:, :start_step] = src_tokens.repeat_interleave(beam_size, 0)
        # compute the model predictions
        model_out = self.model(
            tokens[:, :start_step],
            incremental_state=incremental_states,
        )
        # temperature and normalization
        # convert to float before the temparture divide to ensure good precision.
        # Avoid dividing by 1.0 to prevent unnecessary numerical instability
        # and always log in float
        model_predictions = model_out[0].float()
        if self.temperature > 0 and self.temperature != 1.0:
            model_predictions.div_(self.temperature)
        # lprobs is the log probability of each possible token in every position
        # lprobs \in FloatTensor(bsz * beam_size, prompt_len, vocab_size)
        lprobs = self.model.get_normalized_probs(model_predictions, log_probs=True)

        # don't allow generation of eos/pad
        model_predictions[:, :, self.eos] = -math.inf
        model_predictions[:, :, self.pad] = -math.inf
        for stop_token in self.stop:
            model_predictions[:, :, stop_token] = -math.inf

        if self.need_logprobs:
            all_lprobs[:, 1:start_step] = lprobs[:, :-1].type_as(all_lprobs)
        else:
            all_lprobs = None

        # find and store the logprobs of each prompt token, cutting out the
        # rest of the vocab. Note the shift of 1 here b/c autoregressive.
        prompt_tokens = tokens[:, 1:start_step].unsqueeze(-1)
        # look up a specific vocab logprob, and broadcast it into scores
        toscores = torch.gather(lprobs, -1, prompt_tokens).squeeze(-1)
        scores[:, 1:start_step] = toscores.type_as(scores)
        # reset scores after the last point of forced decoding and gather the
        # probabilities of the most recent token prediction, as search
        # decisions are only over the most recent token.
        lprobs_cut = []
        for i in range(src_tokens.shape[0]):
            prompt_len = src_lengths[i]
            scores[i * beam_size : (i + 1) * beam_size, prompt_len + 1 :] = 0.0  # reset
            lprobs_cut.append(lprobs[i * beam_size : (i + 1) * beam_size, prompt_len])
        lprobs = torch.cat(lprobs_cut, dim=0)

        eos_mask = torch.zeros(lprobs.size(0), dtype=torch.bool, device=lprobs.device)

        for step in range(start_step, max_len):
            if step < min_len:
                # minimum length constraint (does not apply if using prefix_tokens)
                lprobs[:, self.eos] = -math.inf
                for stop_token in self.stop:
                    lprobs[:, stop_token] = -math.inf

            lprobs[lprobs != lprobs] = torch.tensor(-math.inf).to(lprobs)
            lprobs[:, self.pad] = -math.inf  # never select pad

            # handle max length constraint
            if step >= max_len - 1:
                lprobs[:, : self.eos] = -math.inf
                lprobs[:, self.eos + 1 :] = -math.inf

            # apply repetition penalties
            lprobs = self._apply_repetition_penalties(
                lprobs, count_tokens, count_tokens_src
            )

            # already ended beams should only do eos
            lprobs[eos_mask, : self.eos] = -math.inf
            lprobs[eos_mask, self.eos + 1 :] = -math.inf

            # find our next tokens and record them
            # protect this step for the last token so we don't overflow
            next_scores, next_toks = self._sample_topp(lprobs)
            count_tokens = self._update_repetition_counts(next_toks, count_tokens)
            if step < max_len:
                tokens[:, step] = next_toks
                scores[:, step] = next_scores
                if self.need_logprobs:
                    all_lprobs[:, step] = lprobs

            eos_mask |= next_toks == self.eos
            for stop_token in self.stop:
                # if there are other early stopping tokens, allow those to trigger stop
                eos_mask |= next_toks == stop_token

            if torch.all(eos_mask):
                break

            # forward through the next pass
            model_out = self.model.decoder(
                tokens[:, : step + 1],
                incremental_state=incremental_states,
            )
            # see above for why this must remain float
            model_predictions = model_out[0].float()
            if self.temperature > 0 and self.temperature != 1.0:
                model_predictions.div_(self.temperature)
            lprobs = self.model.get_normalized_probs(model_predictions, log_probs=True)
            lprobs = lprobs[:, -1, :]

        # we want the highest scoring items to be top ranked
        beamscores = scores.view(bsz, beam_size, -1).cumsum(dim=-1)[:, :, -1]
        indices = beamscores.sort(dim=-1, descending=True).indices
        sorted_indices = (
            indices + beam_size * torch.arange(bsz, device=lprobs.device).unsqueeze(1)
        ).view(-1)
        tokens = tokens[sorted_indices]
        scores = scores[sorted_indices]

        # prepare the return value
        retval = {
            "tokens": tokens.view(bsz, beam_size, -1),
            "scores": scores.view(bsz, beam_size, -1),
        }
        if all_lprobs is not None:
            all_lprobs = all_lprobs[sorted_indices]
            retval["distributions"] = all_lprobs.view(
                bsz, beam_size, -1, self.vocab_size
            )
        return retval

    def _initialize_generation_buffers(
        self, bsz: int, beam_size: int, max_len: int, src_tokens: torch.Tensor
    ) -> Tuple[
        torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]
    ]:
        """
        Initialize buffers for use during generation.

        Args:
            bsz (int): batch size
            beam_size (int): beam size
            max_len (int): max gen length
            src_tokens (tensor): incoming context

        Returns:
            scores: token score buffer [bsz*beam_size, max_len]
            tokens: generation tokens [bsz*beam_size, max_len]
            count_tokens: token count buffer for in-generation repetition penalty
                [bsz*beam_size, vocab_size]
            count_tokens_src: token count buffer for in-context repetition penalty
                [bsz*beam_size, vocab_size]
        """
        scores = torch.zeros(bsz * beam_size, max_len).to(src_tokens).float()
        tokens = (
            torch.zeros(bsz * beam_size, max_len).to(src_tokens).long().fill_(self.pad)
        )
        count_tokens = None
        count_tokens_src = None
        if self.alpha_presence > 0 or self.alpha_frequency > 0:
            count_tokens = torch.zeros(bsz * beam_size, self.vocab_size).to(src_tokens)
        if self.alpha_presence_src > 0 or self.alpha_frequency_src > 0:
            # histc requires floats
            float_src_tokens = src_tokens.float()
            if self.alpha_src_penalty_end_idx > 0 and not (bsz > 1):
                # ignore this parameter if we're in a batch. sorry!
                float_src_tokens = float_src_tokens[:, : self.alpha_src_penalty_end_idx]
            count_tokens_src = torch.zeros(bsz * beam_size, self.vocab_size).to(
                src_tokens
            )
            for i in range(bsz * beam_size):
                # histc constructs a histogram counting frequencies of occuring values
                # and buckets into vocab_size bins
                count_tokens_src[i] = torch.histc(
                    float_src_tokens[i], self.vocab_size, min=0, max=self.vocab_size
                )
        if self.lambda_decay > 0:
            # need to reset the buffers
            self.toks_since_reset = self.toks_since_reset.unsqueeze(0).repeat(bsz, 1)
            self.sampling_topp_tensor = self.sampling_topp_tensor.repeat(bsz, 1)
            self.init_p = self.init_p.repeat(bsz, 1)

        return scores, tokens, count_tokens, count_tokens_src

    def _sample_topp(self, lprobs):
        """Sample among the smallest set of elements whose cumulative probability mass exceeds p.
        See `"The Curious Case of Neural Text Degeneration"
        (Holtzman et al., 2019) <https://arxiv.org/abs/1904.09751>`_.
        Args:
            lprobs: (bsz x input_beam_size x vocab_size)
                the model's log-probabilities over the vocabulary at the current step
        Return: A tuple of (trimed_probs, truncated_indices) where:
            trimed_probs: (bsz x input_beam_size x ?)
                the model's probabilities over the elements selected to sample from. The
                width of the third dimension is determined by top-P.
            truncated_indices: (bsz x input_beam_size x ?)
                the indices of the chosen elements.
        """
        if self.temperature == 0.0 or self.sampling_topp == 0.0:
            # greedy search
            return tuple(lprobs.max(dim=-1))

        probs = torch.softmax(lprobs, dim=-1)
        sprobs, sinds = probs.sort(dim=-1, descending=True)
        if self.lambda_decay <= 0:
            # normal nucleus sampling
            mask = (sprobs.cumsum(dim=-1) - sprobs) >= self.sampling_topp
        else:
            # factual/decayed nucleus sampling. each batch item may have
            # a different topp value, so the mask is different for each
            mask = (sprobs.cumsum(dim=-1) - sprobs) >= self.sampling_topp_tensor.expand(
                sprobs.size()
            ).to(sprobs.device)
        trunc_sprobs = sprobs.detach().clone()
        trunc_sprobs[mask] = 0
        trunc_sprobs.div_(trunc_sprobs.sum(dim=-1).unsqueeze(-1))
        choices = torch.multinomial(trunc_sprobs, 1)[:, 0]
        hyp_ids = torch.arange(lprobs.size(0)).to(lprobs.device)
        tok_ids = sinds[hyp_ids, choices]
        scores = sprobs[hyp_ids, choices].log()
        if self.lambda_decay > 0:
            self._update_sampling_topp(tok_ids)
        return scores, tok_ids

    def _update_sampling_topp(self, tokens: torch.Tensor):
        """
        Update the p for sampling according to Factual Nucleus generation
        (see https://arxiv.org/abs/2206.04624 for more details)

        Args:
            tokens (torch.Tensor): chosen generation tokens
        """
        for batch_i, toks in enumerate(tokens):
            if toks.dim() == 1:
                for beam_i, t in enumerate(toks):
                    if self.full_stop_list.to(tokens.device).eq(t).sum() > 0:
                        self.toks_since_reset[batch_i, beam_i] = 0
                    else:
                        self.toks_since_reset[batch_i, beam_i] += 1
                    decay_factor = max(0, self.toks_since_reset[batch_i, beam_i] - 1)
                    self.sampling_topp_tensor[batch_i, beam_i] = torch.max(
                        self.omega_bound,
                        self.init_p[batch_i, beam_i]
                        * (self.lambda_decay ** (decay_factor)),
                    )
            else:
                t = toks
                if self.full_stop_list.to(tokens.device).eq(t).sum() > 0:
                    self.toks_since_reset[batch_i] = 0
                else:
                    self.toks_since_reset[batch_i] += 1
                decay_factor = max(0, self.toks_since_reset[batch_i] - 1)
                self.sampling_topp_tensor[batch_i] = torch.max(
                    self.omega_bound,
                    self.init_p[batch_i] * (self.lambda_decay ** (decay_factor)),
                )

    def _apply_repetition_penalties(
        self,
        lprobs: torch.Tensor,
        count_tokens: Optional[torch.Tensor],
        count_tokens_src: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Apply repetition penalties.

        From https://beta.openai.com/docs/api-reference/engines/retrieve:

        mu[j] -> mu[j] - c[j] * alpha_frequency - float(c[j] > 0) * alpha_presence
        Where:

        mu[j] is the logit of the j-th token
        c[j] is how often that token was sampled prior to the current position
        float(c[j] > 0) is 1 if c[j] > 0 and 0 otherwise
        alpha_frequency is the frequency penalty coefficient
        alpha_presence is the presence penalty coefficient

        :param lprobs:
            (bsz*beam_size, vocab_size) tensor
        :param count_tokens:
            (bsz*beam_size, vocab_size) tensor
        :param count_tokens_src:
            (bsz*beam_size, vocab_size) tensor

        :return lprobs:
            return new lprobs!
        """
        if self.alpha_frequency > 0 or self.alpha_presence > 0:
            assert count_tokens is not None
            lprobs = (
                lprobs
                - (count_tokens * self.alpha_frequency)
                - (count_tokens.gt(0) * self.alpha_presence)
            )
        if self.alpha_frequency_src > 0 or self.alpha_presence_src > 0:
            assert count_tokens_src is not None
            lprobs = (
                lprobs
                - (count_tokens_src * self.alpha_frequency_src)
                - (count_tokens_src.gt(0) * self.alpha_presence_src)
            )

        return lprobs

    def _update_repetition_counts(
        self, tokens: torch.Tensor, count_tokens: Optional[torch.Tensor]
    ) -> Optional[torch.Tensor]:
        """
        Update the repetition counts

        :param tokens:
            [batchsize * beam, 1]
        :param count_tokens:
            [batchsize * beam, vocab_size]
        """
        if self.alpha_frequency > 0 or self.alpha_presence > 0:
            assert count_tokens is not None
            for i, t in enumerate(tokens):
                count_tokens[i, t] += 1
        return count_tokens


class EnsembleModel(nn.Module):
    """A wrapper around an ensemble of models."""

    def __init__(self, models):
        super().__init__()
        self.models_size = len(models)
        # method '__len__' is not supported in ModuleList for torch script
        self.single_model = models[0]
        self.models = nn.ModuleList(models)

        self.has_incremental: bool = False
        if all(
            hasattr(m, "decoder") and isinstance(m.decoder, BaseDecoder) for m in models
        ):
            self.has_incremental = True

    def forward(self):
        pass

    def has_encoder(self):
        return hasattr(self.single_model, "encoder")

    def has_incremental_states(self):
        return self.has_incremental

    def max_decoder_positions(self):
        return min(
            [
                m.max_decoder_positions()
                for m in self.models
                if hasattr(m, "max_decoder_positions")
            ]
            + [sys.maxsize]
        )

    def set_decoder_beam_size(self, beam_size):
        """Set beam size for efficient beamable enc-dec attention."""
        if beam_size > 1:
            for model in self.models:
                if hasattr(model, "set_beam_size"):
                    model.set_beam_size(beam_size)

    @torch.jit.export
    def forward_encoder(self, net_input: Dict[str, Tensor]):
        if not self.has_encoder():
            return None
        return [model.encoder.forward_torchscript(net_input) for model in self.models]

    @torch.jit.export
    def forward_decoder(
        self,
        tokens,
        encoder_outs: List[Dict[str, List[Tensor]]],
        incremental_states: List[Dict[str, Dict[str, Optional[Tensor]]]],
        temperature: float = 1.0,
    ):
        log_probs = []
        avg_attn: Optional[Tensor] = None
        encoder_out: Optional[Dict[str, List[Tensor]]] = None
        for i, model in enumerate(self.models):
            if self.has_encoder():
                encoder_out = encoder_outs[i]
            # decode each model
            if self.has_incremental_states():
                decoder_out = model.decoder.forward(
                    tokens,
                    encoder_out=encoder_out,
                    incremental_state=incremental_states[i],
                )
            else:
                if hasattr(model, "decoder"):
                    decoder_out = model.decoder.forward(tokens, encoder_out=encoder_out)
                else:
                    decoder_out = model.forward(tokens)

            attn: Optional[Tensor] = None
            decoder_len = len(decoder_out)
            if decoder_len > 1 and decoder_out[1] is not None:
                if isinstance(decoder_out[1], Tensor):
                    attn = decoder_out[1]
                else:
                    attn_holder = decoder_out[1]["attn"]
                    if isinstance(attn_holder, Tensor):
                        attn = attn_holder
                    elif attn_holder is not None:
                        attn = attn_holder[0]
                if attn is not None:
                    attn = attn[:, -1, :]

            decoder_out_tuple = (
                decoder_out[0][:, -1:, :].div_(temperature),
                None if decoder_len <= 1 else decoder_out[1],
            )
            probs = model.get_normalized_probs(
                decoder_out_tuple, log_probs=True, sample=None
            )
            probs = probs[:, -1, :]
            if self.models_size == 1:
                return probs, attn

            log_probs.append(probs)
            if attn is not None:
                if avg_attn is None:
                    avg_attn = attn
                else:
                    avg_attn.add_(attn)

        avg_probs = torch.logsumexp(torch.stack(log_probs, dim=0), dim=0) - math.log(
            self.models_size
        )

        if avg_attn is not None:
            avg_attn.div_(self.models_size)
        return avg_probs, avg_attn

    @torch.jit.export
    def reorder_encoder_out(
        self, encoder_outs: Optional[List[Dict[str, List[Tensor]]]], new_order
    ):
        """
        Reorder encoder output according to *new_order*.
        Args:
            encoder_out: output from the ``forward()`` method
            new_order (LongTensor): desired order
        Returns:
            *encoder_out* rearranged according to *new_order*
        """
        new_outs: List[Dict[str, List[Tensor]]] = []
        if not self.has_encoder():
            return new_outs
        for i, model in enumerate(self.models):
            assert encoder_outs is not None
            new_outs.append(
                model.encoder.reorder_encoder_out(encoder_outs[i], new_order)
            )
        return new_outs

    @torch.jit.export
    def reorder_incremental_state(
        self,
        incremental_states: List[Dict[str, Dict[str, Optional[Tensor]]]],
        new_order,
    ):
        if not self.has_incremental_states():
            return
        for i, model in enumerate(self.models):
            model.decoder.reorder_incremental_state_scripting(
                incremental_states[i], new_order
            )


class SequenceGeneratorWithAlignment(SequenceGenerator):
    def __init__(
        self, models, tgt_dict, left_pad_target=False, print_alignment="hard", **kwargs
    ):
        """Generates translations of a given source sentence.
        Produces alignments following "Jointly Learning to Align and
        Translate with Transformer Models" (Garg et al., EMNLP 2019).
        Args:
            left_pad_target (bool, optional): Whether or not the
                hypothesis should be left padded or not when they are
                teacher forced for generating alignments.
        """
        super().__init__(EnsembleModelWithAlignment(models), tgt_dict, **kwargs)
        self.left_pad_target = left_pad_target

        if print_alignment == "hard":
            self.extract_alignment = utils.extract_hard_alignment
        elif print_alignment == "soft":
            self.extract_alignment = utils.extract_soft_alignment

    @torch.no_grad()
    def generate(self, models, sample, **kwargs):
        finalized = super()._generate(sample, **kwargs)

        src_tokens = sample["net_input"]["src_tokens"]
        bsz = src_tokens.shape[0]
        beam_size = self.beam_size
        (
            src_tokens,
            src_lengths,
            prev_output_tokens,
            tgt_tokens,
        ) = self._prepare_batch_for_alignment(sample, finalized)
        if any(getattr(m, "full_context_alignment", False) for m in self.model.models):
            attn = self.model.forward_align(src_tokens, src_lengths, prev_output_tokens)
        else:
            attn = [
                finalized[i // beam_size][i % beam_size]["attention"].transpose(1, 0)
                for i in range(bsz * beam_size)
            ]

        if src_tokens.device != "cpu":
            src_tokens = src_tokens.to("cpu")
            tgt_tokens = tgt_tokens.to("cpu")
            attn = [i.to("cpu") for i in attn]

        # Process the attn matrix to extract hard alignments.
        for i in range(bsz * beam_size):
            alignment = self.extract_alignment(
                attn[i], src_tokens[i], tgt_tokens[i], self.pad, self.eos
            )
            finalized[i // beam_size][i % beam_size]["alignment"] = alignment
        return finalized

    def _prepare_batch_for_alignment(self, sample, hypothesis):
        src_tokens = sample["net_input"]["src_tokens"]
        bsz = src_tokens.shape[0]
        src_tokens = (
            src_tokens[:, None, :]
            .expand(-1, self.beam_size, -1)
            .contiguous()
            .view(bsz * self.beam_size, -1)
        )
        src_lengths = sample["net_input"]["src_lengths"]
        src_lengths = (
            src_lengths[:, None]
            .expand(-1, self.beam_size)
            .contiguous()
            .view(bsz * self.beam_size)
        )
        prev_output_tokens = data_utils.collate_tokens(
            [beam["tokens"] for example in hypothesis for beam in example],
            self.pad,
            self.eos,
            self.left_pad_target,
            move_eos_to_beginning=True,
        )
        tgt_tokens = data_utils.collate_tokens(
            [beam["tokens"] for example in hypothesis for beam in example],
            self.pad,
            self.eos,
            self.left_pad_target,
            move_eos_to_beginning=False,
        )
        return src_tokens, src_lengths, prev_output_tokens, tgt_tokens


class EnsembleModelWithAlignment(EnsembleModel):
    """A wrapper around an ensemble of models."""

    def __init__(self, models):
        super().__init__(models)

    def forward_align(self, src_tokens, src_lengths, prev_output_tokens):
        avg_attn = None
        for model in self.models:
            decoder_out = model(src_tokens, src_lengths, prev_output_tokens)
            attn = decoder_out[1]["attn"][0]
            if avg_attn is None:
                avg_attn = attn
            else:
                avg_attn.add_(attn)
        if len(self.models) > 1:
            avg_attn.div_(len(self.models))
        return avg_attn
