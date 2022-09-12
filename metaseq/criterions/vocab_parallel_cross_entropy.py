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
        batch_loss = loss.sum(-1)  # TODO: length normalize?
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
        if not model.training and "is_positive" in sample:
            # ignore dummy batches added to make the batch size consistent
            batch_loss = batch_loss[: sample["is_positive"].size(0)]
            logging_output.update({"batch_loss": batch_loss})
            logging_output.update({"id": sample["id"].tolist()})
            logging_output.update({"is_positive": sample["is_positive"].tolist()})
            logging_output.update({"num_cands": sample["num_cands"].tolist()})
            # ignore negative examples for logging loss
            logging_output.update(
                {"loss": (batch_loss * sample["is_positive"]).sum().data}
            )

        if "src_tokens" in sample["net_input"] and hasattr(self.task, "eod"):
            logging_output["ndocseps"] = (sample["target"] == self.task.eod).sum()
        if (
            len(net_output) >= 2
            and isinstance(net_output[1], dict)
            and "inner_states" in net_output[1]
        ):
            with torch.no_grad():
                # yank out the inner states we wish to instrument
                # see transformer_decoder.py ModelParallelTransformerDecoder.extract_features
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
        if logging_outputs[0].get("batch_loss", None) is not None:
            batch_loss = []
            ids = []
            is_positive = []
            num_cands = []
            for log in logging_outputs:
                batch_loss += log.get("batch_loss", [])
                ids += log.get("id", [])
                is_positive += log.get("is_positive", [])
                num_cands += log.get("num_cands", [])
        else:
            batch_loss = None

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
        if batch_loss is not None:
            # history accepts a single value or a dictionary which contains "value" key
            metrics.log_history(
                "batch_loss",
                [
                    {"id": id, "is_positive": is_p, "num_cands": nc, "value": l}
                    for id, is_p, nc, l in zip(ids, is_positive, num_cands, batch_loss)
                ],
                1,
                round=3,
            )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improve distributed training speed.
        """
        return False  # Since not all metrics are scalar, setting this to False to make it work.
