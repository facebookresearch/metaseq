# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch

from metaseq import metrics, utils
from metaseq.criterions import BaseCriterion, register_criterion
from metaseq.modules.megatron.mpu import vocab_parallel_cross_entropy


@register_criterion("vocab_parallel_cross_entropy")
class VocabParallelCrossEntropyCriterion(BaseCriterion):
    def __init__(self, task):
        super().__init__(task)
        self.start_image_token_index = task.start_image_token_index
        self.end_image_token_index = task.end_image_token_index

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

        loss_flattened = loss.view(-1)
        flat_target = sample["target"].view(-1)

        # Get Image Specific Loss
        image_tokens = torch.logical_and(
            flat_target >= self.start_image_token_index,
            flat_target <= self.end_image_token_index,
        )
        image_loss_unreduced = loss_flattened * image_tokens

        # Get Text Specific Loss
        text_tokens = torch.logical_and(
            torch.logical_not(image_tokens), flat_target != self.padding_idx
        )
        text_loss_unreduced = loss_flattened * text_tokens

        if has_pad:
            loss = loss * (target != self.padding_idx)
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
            # Image Modality
            "image_loss": image_loss_unreduced.sum().data,
            "image_ntokens": image_tokens.sum().data,
            # Text Modality
            "text_loss": text_loss_unreduced.sum().data,
            "text_ntokens": text_tokens.sum().data,
        }
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

        image_sum = sum(log.get("image_loss", 0) for log in logging_outputs)
        ntokens_image = sum(log.get("image_ntokens", 0) for log in logging_outputs)

        text_sum = sum(log.get("text_loss", 0) for log in logging_outputs)
        ntokens_text = sum(log.get("text_ntokens", 0) for log in logging_outputs)

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

        # IMAGE MODALITY
        metrics.log_scalar(
            "loss_image",
            image_sum / ntokens_image / math.log(2),
            ntokens_image,
            round=3,
        )
        metrics.log_derived(
            "ppl_image", lambda meters: utils.get_perplexity(meters["loss_image"].avg)
        )

        # TEXT MODALITY
        metrics.log_scalar(
            "loss_text",
            text_sum / ntokens_text / math.log(2),
            ntokens_text,
            round=3,
        )
        metrics.log_derived(
            "ppl_text", lambda meters: utils.get_perplexity(meters["loss_text"].avg)
        )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improve distributed training speed.
        """
        return True
