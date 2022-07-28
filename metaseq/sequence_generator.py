# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import math
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch import Tensor

logger = logging.getLogger(__name__)


class SequenceGenerator(nn.Module):
    @torch.no_grad()
    def __init__(
        self,
        models,
        tgt_dict,
        beam_size: int = 1,
        max_len_a: int = 0,
        max_len_b: int = 200,
        min_len: int = 1,
        temperature: float = 1.0,
        need_logprobs: bool = False,
        stop: Optional[List[int]] = None,
        topp: float = -1,
        profile=False,
    ):
        """Generates translations of a given source sentence.

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
        """
        super().__init__()
        self.model = models[0]
        self.tgt_dict = tgt_dict
        self.pad = tgt_dict.pad()
        self.unk = tgt_dict.unk()
        self.eos = tgt_dict.eos()
        self.vocab_size = len(tgt_dict)
        # the max beam size is the dictionary size - 1, since we never select pad
        self.default_beam_size = min(beam_size, self.vocab_size - 1)
        self.default_max_len_a = max_len_a
        self.default_max_len_b = max_len_b
        self.default_min_len = min_len
        self.need_logprobs = need_logprobs
        self.stop = stop if stop is not None else []
        if topp is None:
            topp = 0.0
        self.default_sampling_topp = max(0, topp)
        self.default_temperature = temperature
        assert temperature > 0, "--temperature must be greater than 0"

        self.model.eval()
        self.profile = profile

        # [Naman] This is hardcoded for now for nbest=1 and seq-len=2048.
        # With this setting, I am seeing 95 ms -> 66 ms per token latency improvement.
        # We can make it better by creating multiple cuda graphs for various bucketed seq_len.
        max_bsz = 1
        max_len = 2048

        decoder = self.model.decoder
        dtype = next(decoder.parameters()).dtype
        self.static_single_input_embedding = torch.zeros(
            (max_bsz, 1, decoder.embed_dim),
            device='cuda',
            dtype=dtype,
        )
        self.static_masked_tokens = torch.zeros((
            max_bsz * decoder.layers[0].self_attn.num_heads_partition,
            max_len,
            decoder.layers[0].head_dim
        ), dtype=torch.bool, device='cuda')
        self.static_self_attn_padding_mask = torch.ones(
            (max_bsz, max_len),
            dtype=torch.bool, device='cuda'
        )
        self.static_incremental_states = []
        for _ in range(len(decoder.layers)):
            prev_key = torch.zeros((max_bsz * decoder.layers[0].self_attn.num_heads_partition, max_len, decoder.layers[0].head_dim), dtype=dtype, device='cuda')
            prev_value = torch.zeros((max_bsz * decoder.layers[0].self_attn.num_heads_partition, max_len, decoder.layers[0].head_dim), dtype=dtype, device='cuda')
            self.static_incremental_states.append({
                'prev_key': prev_key,
                'prev_value': prev_value,
            })
        self._warmup()
        self.recorded_graph = self._record_graph()
        self._run_recorded_graph()

    def _warmup(self):
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for i in range(10):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                _ = self.model.decoder.forward_transformer_layers(
                    self.static_single_input_embedding,
                    incremental_state=self.static_incremental_states,
                    self_attn_mask=None,
                    encoder_out=None,
                    self_attn_padding_mask=self.static_self_attn_padding_mask,
                    static_masked_tokens=self.static_masked_tokens
                )
                end.record()
                torch.cuda.synchronize()

                time = start.elapsed_time(end)
                # Params have been updated. static_y_pred, static_loss, and .grad
                # attributes hold values from computing on this iteration's data.
                logger.info(f"warmup time for one iteration: {time}")
        torch.cuda.current_stream().wait_stream(s)

    def _record_graph(self):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()

        recorded_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(recorded_graph):
            self.static_output = self.model.decoder.forward_transformer_layers(
                self.static_single_input_embedding,
                incremental_state=self.static_incremental_states,
                self_attn_mask=None,
                encoder_out=None,
                self_attn_padding_mask=self.static_self_attn_padding_mask,
                static_masked_tokens=self.static_masked_tokens,
            )
        end.record()
        torch.cuda.synchronize()

        time = start.elapsed_time(end)
        # Params have been updated. static_y_pred, static_loss, and .grad
        # attributes hold values from computing on this iteration's data.
        logger.info(f"time taken in recoding graph: {time}")
        return recorded_graph


    def _run_recorded_graph(self):
        for i in range(10):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            self.recorded_graph.replay()
            end.record()
            torch.cuda.synchronize()

            time = start.elapsed_time(end)
        # Params have been updated. static_y_pred, static_loss, and .grad
        # attributes hold values from computing on this iteration's data.
            logger.info(f"time for replaying recorded graph: {time}")


    def _copy_incremental_state_to_static_ones(self, new_incremental_states, offset=0):
        for new_incremental_state, current_incremental_state in zip(new_incremental_states, self.static_incremental_states):
            current_incremental_state['prev_key'][:, offset:offset+new_incremental_state['prev_key'].size(1),:].copy_(new_incremental_state['prev_key'])
            current_incremental_state['prev_value'][:, offset:offset+new_incremental_state['prev_value'].size(1),:].copy_(new_incremental_state['prev_value'])


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
        beam_size: Optional[int] = None,
        max_len_a: Optional[int] = None,
        max_len_b: Optional[int] = None,
        min_len: Optional[int] = None,
        temperature: Optional[float] = None,
        sampling_topp: Optional[float] = None,
    ):
        """
        Args:
            sample (dict): batch
            prefix_tokens (torch.LongTensor, optional): force decoder to begin
                with these tokens
            bos_token (int, optional): beginning of sentence token
                (default: self.eos)
        """
        # incremental_states = torch.jit.annotate(
        #     Dict[str, Dict[str, Optional[Tensor]]], {}
        # )
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
        beam_size = beam_size or self.default_beam_size
        max_len_a = max_len_a or self.default_max_len_a
        max_len_b = max_len_b or self.default_max_len_b
        min_len = min_len or self.default_min_len
        temperature = temperature or self.default_temperature
        sampling_topp = sampling_topp or self.default_sampling_topp

        max_len = min(self.model.max_decoder_positions() - 1, max_len_b or 1e99)
        min_len = min(max_len - 1, min_len or 0)

        assert (
            min_len <= max_len
        ), "min_len cannot be larger than max_len, please adjust these!"

        # placeholder of indices for bsz * beam_size to hold tokens and accumulative scores
        new_order = torch.arange(bsz).view(-1, 1).repeat(1, beam_size).view(-1)
        new_order = new_order.to(src_tokens.device).long()

        # initialize buffers
        scores = (
            torch.zeros(bsz * beam_size, max_len + 1).to(src_tokens).float()
        )  # +1 for eos; pad is never chosen for scoring
        tokens = (
            torch.zeros(bsz * beam_size, max_len + 2)
            .to(src_tokens)
            .long()
            .fill_(self.pad)
        )  # +2 for eos and pad
        tokens[:, 0] = self.eos if bos_token is None else bos_token

        # notes:
        # - scores \in FloatTensor(bsz * beam_size, max_len + 1)
        # - tokens \in LongTensor(bsz * beam_size, max_len + 2)
        # - src_tokens \in LongTensor(bsz, prompt_len)
        # - all_lprobs \in FloatTensor(bsz * beam_size, max_len + 1, vocab_size)
        #   is the next word distribution at every timestep

        if self.need_logprobs:
            # lprobs are costly for memory, so only compute them if we have to
            all_lprobs = (
                torch.zeros(bsz * beam_size, max_len + 1, self.vocab_size)
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

        model_out = self.model.decoder(
            tokens[:, :start_step]
        )
        self._copy_incremental_state_to_static_ones(model_out[2], offset=0)


        # normalize
        model_out[0].div_(temperature, rounding_mode="trunc")
        # lprobs is the log probability of each possible token in every position
        # lprobs \in FloatTensor(bsz * beam_size, prompt_len, vocab_size)
        lprobs = self.model.get_normalized_probs(model_out, log_probs=True, sample=None)

        # don't allow generation of eos/pad
        model_out[0][:, :, self.eos] = -math.inf
        model_out[0][:, :, self.pad] = -math.inf
        for stop_token in self.stop:
            model_out[0][:, :, stop_token] = -math.inf

        if self.need_logprobs:
            all_lprobs[:, :start_step] = lprobs.type_as(all_lprobs)
        else:
            all_lprobs = None

        # find and store the logprobs of each prompt token, cutting out the
        # rest of the vocab. Note the shift of 1 here b/c autoregressive.
        prompt_tokens = tokens[:, 1:start_step].unsqueeze(-1)
        # look up a specific vocab logprob, and broadcast it into scores
        toscores = torch.gather(lprobs, -1, prompt_tokens).squeeze(-1)
        scores[:, : start_step - 1] = toscores.type_as(scores)
        # reset scores after the last point of forced decoding and gather the
        # probabilities of the most recent token prediction, as search
        # decisions are only over the most recent token.
        lprobs_cut = []
        for i in range(src_tokens.shape[0]):
            prompt_len = src_lengths[i]
            scores[i * beam_size : (i + 1) * beam_size, prompt_len:] = 0.0  # reset
            lprobs_cut.append(lprobs[i * beam_size : (i + 1) * beam_size, prompt_len])
        lprobs = torch.cat(lprobs_cut, dim=0)

        eos_mask = torch.zeros(lprobs.size(0), dtype=torch.bool, device=lprobs.device)

        for step in range(start_step, max_len + 1):
            if step < min_len:
                # minimum length constraint (does not apply if using prefix_tokens)
                lprobs[:, self.eos] = -math.inf
                for stop_token in self.stop:
                    lprobs[:, stop_token] = -math.inf

            lprobs[lprobs != lprobs] = torch.tensor(-math.inf).to(lprobs)
            lprobs[:, self.pad] = -math.inf  # never select pad

            # handle max length constraint
            if step >= max_len:
                lprobs[:, : self.eos] = -math.inf
                lprobs[:, self.eos + 1 :] = -math.inf

            # already ended beams should only do eos
            lprobs[eos_mask, : self.eos] = -math.inf
            lprobs[eos_mask, self.eos + 1 :] = -math.inf

            # find our next tokens and record them
            next_scores, next_toks = self._sample_topp(lprobs, sampling_topp, temperature)
            tokens[:, step] = next_toks
            scores[:, step] = next_scores

            eos_mask |= next_toks == self.eos
            for stop_token in self.stop:
                # if there are other early stopping tokens, allow those to trigger stop
                eos_mask |= next_toks == stop_token

            if torch.all(eos_mask):
                break

            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()

            x, _, _ = self.model.decoder.forward_embedding(tokens[:, : step + 1], None)
            # forward through the next pass
            self.static_masked_tokens.fill_(False)
            self.static_masked_tokens[:, step: step+1, :] = True
            self.static_self_attn_padding_mask.fill_(True)
            self.static_self_attn_padding_mask[:, :step+1].fill_(False)
            self.static_single_input_embedding.copy_(x[:, -1:, :])
            self.recorded_graph.replay()
            # model_out = (self.static_output[0], self.static_output[1])
            torch.cuda.synchronize()
            model_out = (self.model.decoder.output_layer(self.static_output[0]), self.static_output[1])


            end.record()
            torch.cuda.synchronize()

            time = start.elapsed_time(end)
            logger.info(f"time for replaying recorded graph: {time}")

            model_out[0].div_(temperature)
            lprobs = self.model.get_normalized_probs(
                model_out, log_probs=True, sample=None
            )
            lprobs = lprobs[:, -1, :]
            if self.need_logprobs:
                all_lprobs[:, step] = lprobs

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
            "tokens": tokens.view(bsz, beam_size, -1)[:, :, 1:],
            "scores": scores.view(bsz, beam_size, -1)[:, :, 1:],
        }
        if all_lprobs is not None:
            all_lprobs = all_lprobs[sorted_indices]
            retval["distributions"] = all_lprobs.view(
                bsz, beam_size, -1, self.vocab_size
            )
        return retval

    def _sample_topp(self, lprobs,  sampling_topp, temperature):
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
        if temperature == 0.0 or sampling_topp == 0.0:
            # greedy search
            return tuple(lprobs.max(dim=-1))

        probs = torch.softmax(lprobs, dim=-1)
        sprobs, sinds = probs.sort(dim=-1, descending=True)
        mask = (sprobs.cumsum(dim=-1) - sprobs) >= sampling_topp
        trunc_sprobs = sprobs.detach().clone()
        trunc_sprobs[mask] = 0
        trunc_sprobs.div_(trunc_sprobs.sum(dim=-1).unsqueeze(-1))
        choices = torch.multinomial(trunc_sprobs, 1)[:, 0]
        hyp_ids = torch.arange(lprobs.size(0)).to(lprobs.device)
        tok_ids = sinds[hyp_ids, choices]
        scores = sprobs[hyp_ids, choices].log()
        return scores, tok_ids
