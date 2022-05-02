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

from metaseq import search

logger = logging.getLogger(__name__)


class SequenceGenerator(nn.Module):
    def __init__(
        self,
        models,
        tgt_dict,
        beam_size: int = 1,
        max_len_a: int = 0,
        max_len_b: int = 200,
        min_len: int = 1,
        temperature: float = 1.0,
        search_strategy=None,
        need_logprobs: bool = False,
        stop: Optional[List[int]] = None,
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
        self.beam_size = beam_size
        # the max beam size is the dictionary size - 1, since we never select pad
        self.beam_size = min(beam_size, self.vocab_size - 1)
        self.max_len_a = max_len_a
        self.max_len_b = max_len_b
        self.min_len = min_len
        self.need_logprobs = need_logprobs
        self.stop = stop if stop is not None else []

        self.temperature = temperature
        assert temperature > 0, "--temperature must be greater than 0"

        self.search = (
            search.BeamSearch(tgt_dict) if search_strategy is None else search_strategy
        )

        self.model.eval()
        self.profile = profile

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

        max_len = min(self.model.max_decoder_positions() - 1, self.max_len_b or 1e99)
        min_len = min(max_len - 1, self.min_len or 0)

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

        # A list that indicates candidates that should be ignored.
        # For example, suppose we're sampling and have already finalized 2/5
        # samples. Then cands_to_ignore would mark 2 positions as being ignored,
        # so that we only finalize the remaining 3 samples.
        cands_to_ignore = (
            torch.zeros(bsz, beam_size).to(src_tokens).eq(-1)
        )  # forward and backward-compatible False mask

        # list of completed sentences
        finalized = torch.jit.annotate(
            List[List[Dict[str, Tensor]]],
            [torch.jit.annotate(List[Dict[str, Tensor]], []) for i in range(bsz)],
        )  # contains lists of dictionaries of infomation about the hypothesis being finalized at each step

        finished = [
            False for i in range(bsz)
        ]  # a boolean array indicating if the sentence at the index is finished or not
        num_remaining_sent = bsz  # number of sentences remaining

        # number of candidate hypos per step
        cand_size = 2 * beam_size  # 2 x beam size in case half are EOS

        # offset arrays for converting between different indexing schemes
        bbsz_offsets = (
            (torch.arange(0, bsz) * beam_size)
            .unsqueeze(1)
            .type_as(tokens)
            .to(src_tokens.device)
        )
        cand_offsets = torch.arange(0, cand_size).type_as(tokens).to(src_tokens.device)

        reorder_state: Optional[Tensor] = None
        batch_idxs: Optional[Tensor] = None

        original_batch_idxs: Optional[Tensor] = None
        if "id" in sample and isinstance(sample["id"], Tensor):
            original_batch_idxs = sample["id"]
        else:
            original_batch_idxs = torch.arange(0, bsz).type_as(tokens)

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
        tokens[:, :start_step] = src_tokens
        # compute the model predictions
        model_out = self.model.decoder(
            tokens[:, :start_step],
            incremental_state=incremental_states,
        )
        # normalize
        model_out[0].div_(self.temperature, rounding_mode="trunc")
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
        # finally, scores is actually stored as the cumulative NLL, but we have
        # individual NLL scores right now
        scores = scores.cumsum(dim=1)

        # start from previous timestep because we still have to do beam search
        # bookkeeping (i.e. finalize the hypothesis if it's the final token)
        for step in range(start_step - 1, max_len + 1):
            # reorder decoder internal states based on the prev choice of beams
            if reorder_state is not None:
                if batch_idxs is not None:
                    # update beam indices to take into account removed sentences
                    corr = batch_idxs - torch.arange(batch_idxs.numel()).type_as(
                        batch_idxs
                    )
                    reorder_state.view(-1, beam_size).add_(
                        corr.unsqueeze(-1) * beam_size
                    )
                    original_batch_idxs = original_batch_idxs[batch_idxs]
                # note we may be decreasing the size of the incremental state or
                # all_lprobs here, as beams hit EOS.
                self.model.decoder.reorder_incremental_state_scripting(
                    incremental_states, reorder_state
                )
                if self.need_logprobs:
                    all_lprobs = all_lprobs.index_select(0, reorder_state)

            # the first step is already computed via forced decoding above, but
            # we still need to do the beam search aspects, so only compute the
            # forward pass if we're past the prompt tokens
            if step != start_step - 1:
                model_out = self.model.decoder(
                    tokens[:, : step + 1],
                    incremental_state=incremental_states,
                )
                model_out[0].div_(self.temperature)
                lprobs = self.model.get_normalized_probs(
                    model_out, log_probs=True, sample=None
                )
                # search decisions are only over the most recent token
                lprobs = lprobs[:, -1, :]
                if self.need_logprobs:
                    all_lprobs[:, step] = lprobs

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

            # Shape: (batch, cand_size)
            cand_scores, cand_indices, cand_beams = self.search.step(
                # underlying search indexes from first token being generated,
                # so we need to account for the size of the prompt.
                step - start_step + 1,
                lprobs.view(bsz, -1, self.vocab_size),
                scores[:, start_step - 1 : step].view(bsz, beam_size, -1),
                tokens[:, start_step - 1 : step + 1],
                original_batch_idxs,
            )

            # cand_bbsz_idx contains beam indices for the top candidate
            # hypotheses, with a range of values: [0, bsz*beam_size),
            # and dimensions: [bsz, cand_size]
            cand_bbsz_idx = cand_beams.add(bbsz_offsets)

            # finalize hypotheses that end in eos
            # Shape of eos_mask: (batch size, beam size)
            eos_mask = cand_indices.eq(self.eos)
            for stop_token in self.stop:
                # if there are other early stopping tokens, allow those to trigger stop
                eos_mask = eos_mask | cand_indices.eq(stop_token)
            eos_mask = eos_mask & cand_scores.ne(-math.inf)
            eos_mask[:, :beam_size][cands_to_ignore] = torch.tensor(0).to(eos_mask)

            # only consider eos when it's among the top beam_size indices
            # Now we know what beam item(s) to finish
            # Shape: 1d list of absolute-numbered
            eos_bbsz_idx = torch.masked_select(
                cand_bbsz_idx[:, :beam_size], mask=eos_mask[:, :beam_size]
            )

            finalized_sents: List[int] = []
            if eos_bbsz_idx.numel() > 0:
                eos_scores = torch.masked_select(
                    cand_scores[:, :beam_size], mask=eos_mask[:, :beam_size]
                )

                finalized_sents = self.finalize_hypos(
                    step,
                    eos_bbsz_idx,
                    eos_scores,
                    tokens,
                    scores,
                    finalized,
                    finished,
                    beam_size,
                    max_len,
                    all_lprobs,
                )
                num_remaining_sent -= len(finalized_sents)

            assert num_remaining_sent >= 0
            if num_remaining_sent == 0:
                break
            if self.search.stop_on_max_len and step >= max_len:
                break
            assert step < max_len, f"{step} < {max_len}"

            # Remove finalized sentences (ones for which {beam_size}
            # finished hypotheses have been generated) from the batch.
            if len(finalized_sents) > 0:
                new_bsz = bsz - len(finalized_sents)

                # construct batch_idxs which holds indices of batches to keep for the next pass
                batch_mask = torch.ones(
                    bsz, dtype=torch.bool, device=cand_indices.device
                )
                batch_mask[finalized_sents] = False
                # TODO replace `nonzero(as_tuple=False)` after TorchScript supports it
                batch_idxs = torch.arange(
                    bsz, device=cand_indices.device
                ).masked_select(batch_mask)

                # Choose the subset of the hypothesized constraints that will continue
                self.search.prune_sentences(batch_idxs)

                eos_mask = eos_mask[batch_idxs]
                cand_beams = cand_beams[batch_idxs]
                bbsz_offsets.resize_(new_bsz, 1)
                cand_bbsz_idx = cand_beams.add(bbsz_offsets)
                cand_scores = cand_scores[batch_idxs]
                cand_indices = cand_indices[batch_idxs]

                if prefix_tokens is not None:
                    prefix_tokens = prefix_tokens[batch_idxs]
                src_lengths = src_lengths[batch_idxs]
                cands_to_ignore = cands_to_ignore[batch_idxs]

                scores = scores.view(bsz, -1)[batch_idxs].view(new_bsz * beam_size, -1)
                tokens = tokens.view(bsz, -1)[batch_idxs].view(new_bsz * beam_size, -1)
                bsz = new_bsz
            else:
                batch_idxs = None

            # Set active_mask so that values > cand_size indicate eos hypos
            # and values < cand_size indicate candidate active hypos.
            # After, the min values per row are the top candidate active hypos

            # Rewrite the operator since the element wise or is not supported in torchscript.

            eos_mask[:, :beam_size] = ~((~cands_to_ignore) & (~eos_mask[:, :beam_size]))
            active_mask = torch.add(
                eos_mask.type_as(cand_offsets) * cand_size,
                cand_offsets[: eos_mask.size(1)],
            )

            # get the top beam_size active hypotheses, which are just
            # the hypos with the smallest values in active_mask.
            # {active_hypos} indicates which {beam_size} hypotheses
            # from the list of {2 * beam_size} candidates were
            # selected. Shapes: (batch size, beam size)
            new_cands_to_ignore, active_hypos = torch.topk(
                active_mask, k=beam_size, dim=1, largest=False
            )

            # update cands_to_ignore to ignore any finalized hypos.
            cands_to_ignore = new_cands_to_ignore.ge(cand_size)[:, :beam_size]
            # Make sure there is at least one active item for each sentence in the batch.
            assert (~cands_to_ignore).any(dim=1).all()

            # update cands_to_ignore to ignore any finalized hypos

            # {active_bbsz_idx} denotes which beam number is continued for each new hypothesis (a beam
            # can be selected more than once).
            active_bbsz_idx = torch.gather(cand_bbsz_idx, dim=1, index=active_hypos)
            active_scores = torch.gather(cand_scores, dim=1, index=active_hypos)

            active_bbsz_idx = active_bbsz_idx.view(-1)
            active_scores = active_scores.view(-1)

            # copy tokens and scores for active hypotheses. note that we cut off
            # the prompt tokens here for ease of bookkeeping.

            # Set the tokens for each beam (can select the same row more than once)
            tokens[:, start_step : step + 1] = torch.index_select(
                tokens[:, start_step : step + 1], dim=0, index=active_bbsz_idx
            )
            # Select the next token for each of them
            tokens.view(bsz, beam_size, -1)[:, :, step + 1] = torch.gather(
                cand_indices, dim=1, index=active_hypos
            )
            if step > start_step:
                scores[:, start_step:step] = torch.index_select(
                    scores[:, start_step:step], dim=0, index=active_bbsz_idx
                )
            scores.view(bsz, beam_size, -1)[:, :, step] = torch.gather(
                cand_scores, dim=1, index=active_hypos
            )

            # reorder incremental state in decoder
            reorder_state = active_bbsz_idx

        # sort by score descending
        for sent in range(len(finalized)):
            scores = torch.tensor(
                [float(elem["score"].item()) for elem in finalized[sent]]
            )
            _, sorted_scores_indices = torch.sort(scores, descending=True)
            finalized[sent] = [finalized[sent][ssi] for ssi in sorted_scores_indices]
            finalized[sent] = torch.jit.annotate(
                List[Dict[str, Tensor]], finalized[sent]
            )
        return finalized

    def finalize_hypos(
        self,
        step: int,
        bbsz_idx,
        eos_scores,
        tokens,
        scores,
        finalized: List[List[Dict[str, Tensor]]],
        finished: List[bool],
        beam_size: int,
        max_len: int,
        all_lprobs: Optional[Tensor],
    ):
        """
        Finalize hypothesis, store finalized information in `finalized`, and
        change `finished` accordingly. A sentence is finalized when {beam_size}
        finished items have been collected for it.

        Returns number of sentences (not beam items) being finalized. These
        will be removed from the batch and not processed further.
        """
        assert bbsz_idx.numel() == eos_scores.numel()

        # clone relevant token and attention tensors.
        # tokens is (batch * beam, max_len). So the index_select
        # gets the newly EOS rows, then selects cols 1..{step + 2}
        tokens_clone = tokens.index_select(0, bbsz_idx)[
            :, 1 : step + 2
        ]  # skip the first index, which is EOS

        tokens_clone[:, step] = self.eos

        # compute scores per token position
        pos_scores = scores.index_select(0, bbsz_idx)[:, : step + 1]
        pos_scores[:, step] = eos_scores
        # convert from cumulative to per-position scores
        pos_scores[:, 1:] = pos_scores[:, 1:] - pos_scores[:, :-1]

        # cum_unfin records which sentences in the batch are finished.
        # It helps match indexing between (a) the original sentences
        # in the batch and (b) the current, possibly-reduced set of
        # sentences.
        cum_unfin: List[int] = []
        prev = 0
        for f in finished:
            if f:
                prev += 1
            else:
                cum_unfin.append(prev)

        # The keys here are of the form "{sent}_{unfin_idx}", where
        # "unfin_idx" is the index in the current (possibly reduced)
        # list of sentences, and "sent" is the index in the original,
        # unreduced batch
        # set() is not supported in script export
        sents_seen: Dict[str, Optional[Tensor]] = {}

        # For every finished beam item
        for i in range(bbsz_idx.size()[0]):
            idx = bbsz_idx[i]
            score = eos_scores[i]
            # sentence index in the current (possibly reduced) batch
            unfin_idx = torch.div(idx, beam_size, rounding_mode="trunc")
            # sentence index in the original (unreduced) batch
            sent = unfin_idx + cum_unfin[unfin_idx]
            # Cannot create dict for key type '(int, int)' in torchscript.
            # The workaround is to cast int to string
            seen = str(sent.item()) + "_" + str(unfin_idx.item())
            if seen not in sents_seen:
                sents_seen[seen] = None

            # An input sentence (among those in a batch) is finished when
            # beam_size hypotheses have been collected for it
            if len(finalized[sent]) < beam_size:
                fin = {
                    "tokens": tokens_clone[i],
                    "score": score,
                    "alignment": torch.empty(0),
                    "positional_scores": pos_scores[i],
                }
                if all_lprobs is not None:
                    fin["distributions"] = all_lprobs[i]
                finalized[sent].append(fin)

        newly_finished: List[int] = []

        for seen in sents_seen.keys():
            # check termination conditions for this sentence
            sent: int = int(float(seen.split("_")[0]))
            unfin_idx: int = int(float(seen.split("_")[1]))

            if not finished[sent] and self.is_finished(
                step, max_len, len(finalized[sent]), beam_size
            ):
                finished[sent] = True
                newly_finished.append(unfin_idx)

        return newly_finished

    def is_finished(
        self,
        step: int,
        max_len: int,
        finalized_sent_len: int,
        beam_size: int,
    ):
        """
        Check whether decoding for a sentence is finished, which
        occurs when the list of finalized sentences has reached the
        beam size, or when we reach the maximum length.
        """
        assert finalized_sent_len <= beam_size
        if finalized_sent_len == beam_size or step == max_len:
            return True
        return False
