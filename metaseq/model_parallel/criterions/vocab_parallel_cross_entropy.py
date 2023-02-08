# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import torch

from metaseq import metrics, utils
from metaseq.criterions import BaseCriterion, register_criterion


try:
    from megatron.mpu.cross_entropy import (
        vocab_parallel_cross_entropy,
    )

    has_megatron_submodule = True
except (ImportError, ModuleNotFoundError):
    has_megatron_submodule = False


@register_criterion("vocab_parallel_cross_entropy")
class VocabParallelCrossEntropyCriterion(BaseCriterion):
    def __init__(self, task):
        super().__init__(task)
        if not has_megatron_submodule:
            raise ImportError(
                "\n\nPlease install megatron using the setup instructions!"
            )
        self.split_loss = getattr(task, "split_loss", None)

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        target = sample["target"]
        has_pad = target.eq(self.padding_idx).any().item()

        net_output = model(**sample["net_input"])
        loss = vocab_parallel_cross_entropy(net_output[0].float(), target)
        if has_pad:
            loss = loss * (target != self.padding_idx)
        if self.split_loss is not None:
            loss_start = loss.detach()[:, :self.split_loss].sum()
            loss_end = loss.detach()[:, self.split_loss:].sum()
            if has_pad:
                ntokens_start = (target.detach()[:, :self.split_loss] != self.padding_idx).sum()
                ntokens_end = (target.detach()[:, self.split_loss:] != self.padding_idx).sum()
            else:
                ntokens_start = loss.shape[0] * min(loss.shape[1], self.split_loss)
                ntokens_end = loss.shape[0] * max(loss.shape[1] - self.split_loss, 0)

        loss = loss.sum()
        # When using target loss only, use num tokens in target only as the sample_size
        # See StreamingSrcTgtDataset
        sample_size = (
            sample["ntokens_target"]
            if "ntokens_target" in sample
            else sample["ntokens"]
        )
        logging_output = {
            "loss": loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }
        if self.split_loss is not None:
            logging_output["loss_start"] = loss_start.item()
            logging_output["loss_end"] = loss_end.item()
            logging_output["ntokens_start"] = ntokens_start
            logging_output["ntokens_end"] = ntokens_end
        if "src_tokens" in sample["net_input"] and hasattr(self.task, "eod"):
            logging_output["ndocseps"] = (sample["target"] == self.task.eod).sum()
        if (
            len(net_output) >= 2
            and isinstance(net_output[1], dict)
            and "inner_states" in net_output[1]
        ):
            with torch.no_grad():
                # yank out the inner states we wish to instrument
                # see transformer_decoder.py TransformerDecoder.extract_features
                emb, *_, actv = net_output[1]["inner_states"]
                assert isinstance(
                    emb, dict
                ), "Expecting the first inner state to be a dict of embedding representations"
                emb["actv"] = actv  # throw on final for code brevity
                for key, value in emb.items():
                    if value is None:
                        # maybe future proofing relative positional embeddings
                        continue
                    value = emb[key]
                    logging_output[f"{key}_norm"] = value.norm(p=2, dim=-1).sum(
                        dtype=torch.float32
                    )

        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        for type_ in ("actv", "pos", "tok", "emb"):
            key = f"{type_}_norm"
            if any(key in log for log in logging_outputs):
                actv_norm = sum(log.get(key, 0) for log in logging_outputs)
                metrics.log_scalar(key, actv_norm / ntokens, round=3)

        if any("ndocseps" in log for log in logging_outputs):
            # nsentences = batch size
            nsentences = sum(log.get("nsentences", 0) for log in logging_outputs)
            # ndocseps = number of document separators we found
            ndocseps = sum(log.get("ndocseps", 0) for log in logging_outputs)
            # so docs/example = (1 + ndocseps) / example = (ndocseps + nsents) / nsents
            metrics.log_scalar("docsperex", (ndocseps + nsentences) / nsentences)

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_derived(
            "ppl", lambda meters: utils.get_perplexity(meters["loss"].avg)
        )

        if any("loss_start" in log for log in logging_outputs):
            loss_start_sum = sum(log.get("loss_start", 0) for log in logging_outputs)
            loss_end_sum = sum(log.get("loss_end", 0) for log in logging_outputs)
            ntokens_start = sum(log.get("ntokens_start", 0) for log in logging_outputs)
            ntokens_end = sum(log.get("ntokens_end", 0) for log in logging_outputs)
            metrics.log_scalar(
                "loss_start", loss_start_sum / ntokens_start / math.log(2), ntokens_start, round=3
            )
            metrics.log_scalar(
                "loss_end", loss_end_sum / ntokens_end / math.log(2), ntokens_end, round=3
            )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improve distributed training speed.
        """
        return True
