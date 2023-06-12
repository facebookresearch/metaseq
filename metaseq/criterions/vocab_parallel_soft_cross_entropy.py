import math
import torch

from metaseq import metrics, utils
from metaseq.criterions import BaseCriterion, register_criterion
from metaseq.criterions.vocab_parallel_cross_entropy import VocabParallelCrossEntropyCriterion
from metaseq.utils import print_with_rank, print_tensor_with_rank

try:
    from megatron.mpu.initialize import get_tensor_model_parallel_group
    from megatron.mpu.initialize import get_tensor_model_parallel_rank
    from megatron.mpu.initialize import get_tensor_model_parallel_world_size
    from megatron.mpu.utils import VocabUtility
    has_megatron_submodule = True
except (ImportError, ModuleNotFoundError):
    has_megatron_submodule = False


class _VocabParallelSoftCrossEntropy(torch.autograd.Function):

    @staticmethod
    def forward(ctx, vocab_parallel_logits, target_tokens, target_predictions):
        # Maximum value along vocab dimension across all GPUs.
        logits_max = torch.max(vocab_parallel_logits, dim=-1)[0]
        torch.distributed.all_reduce(logits_max, op=torch.distributed.ReduceOp.MAX, group=get_tensor_model_parallel_group())
        # Subtract the maximum value.
        vocab_parallel_logits.sub_(logits_max.unsqueeze(dim=-1))

        # Get the partition's vocab indecies
        get_vocab_range = VocabUtility.vocab_range_from_per_partition_vocab_size
        partition_vocab_size = vocab_parallel_logits.size()[-1]
        rank = get_tensor_model_parallel_rank()
        world_size = get_tensor_model_parallel_world_size()
        vocab_start_index, vocab_end_index = get_vocab_range(partition_vocab_size, rank, world_size)

        if target_tokens.dtype != torch.int64:
            target_tokens = target_tokens.long()

        # Create a mask of valid vocab ids (1 means it needs to be masked).
        target_mask = (target_tokens < vocab_start_index) | (target_tokens >= vocab_end_index)
        masked_target = target_tokens.clone() - vocab_start_index
        masked_target[target_mask] = 0

        # Get predicted-logits = logits[top_logprobs].
        predicted_logits = vocab_parallel_logits.gather(dim=-1, index=masked_target)
        predicted_logits[target_mask] = 0.0
        # All reduce is needed to get the chunks from other GPUs.
        torch.distributed.all_reduce(
            predicted_logits, op=torch.distributed.ReduceOp.SUM, group=get_tensor_model_parallel_group()
        )

        # Sum of exponential of logits along vocab dimension across all GPUs.
        exp_logits = vocab_parallel_logits
        torch.exp(vocab_parallel_logits, out=exp_logits)
        sum_exp_logits = exp_logits.sum(dim=-1)
        torch.distributed.all_reduce(
            sum_exp_logits, op=torch.distributed.ReduceOp.SUM, group=get_tensor_model_parallel_group()
        )

        # Loss = log(sum(exp(logits))) - predicted-logit.
        target_weights = target_predictions.exp()
        loss = ((torch.log(sum_exp_logits).unsqueeze(dim=-1) - predicted_logits) * target_weights).sum(-1)

        # Store softmax, top_logprobs-mask and masked-top_logprobs for backward pass.
        softmax = exp_logits.div(sum_exp_logits.unsqueeze(dim=-1))
        ctx.save_for_backward(softmax, target_mask, masked_target, target_weights)

        return loss

    @staticmethod
    def backward(ctx, grad_output):

        # Retreive tensors from the forward path.
        softmax, target_mask, masked_target, target_weights = ctx.saved_tensors

        # All the inputs have softmax as thier gradient.
        grad_input = softmax

        # Add the gradient from matching classes.
        grad_input = grad_input.scatter_add(-1, masked_target, (-1.0 + target_mask.float()) * target_weights)

        # Finally elementwise multiplication with the output gradients.
        grad_input = grad_input.mul(grad_output.unsqueeze(dim=-1))

        return grad_input, None, None


def vocab_parallel_soft_cross_entropy(vocab_parallel_predictions, target_tokens, target_predictions):
    """Helper function for the Soft Cross entropy w/ model parallel."""
    return _VocabParallelSoftCrossEntropy.apply(vocab_parallel_predictions, target_tokens, target_predictions)


@register_criterion("vocab_parallel_soft_cross_entropy")
class VocabParallelSoftCrossEntropyCriterion(VocabParallelCrossEntropyCriterion):

    def __init__(self, task):
        super().__init__(task)
        if not has_megatron_submodule:
            raise ImportError("\n\nPlease install megatron using the setup instructions!")

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample["net_input"])
        logits = net_output[0].float()
        target = sample["target"]

        masked_targets = sample["masked_targets"]
        has_mask = masked_targets.any().item()

        assert sample["distillation_mode"] == 'logprobs_distillation'
        loss = vocab_parallel_soft_cross_entropy(logits.clone(), target["target_tokens"], target["target_predictions"])
        if has_mask:
            masked_targets = masked_targets.max(dim=-1)[0]
            loss = loss * (~masked_targets).float()
        loss = loss.sum()

        # Take the first (highest predicted) token of each.
        # This couples code with production of target tokens
        # See: https://github.com/microsoft/grindstone/blob/9f9a1e49d63954fa9c272821e085f6103939c3e0/metaseq/tasks/streaming_distillation_language_modeling.py#L261-L263
        target_token_ids = target["target_tokens"][:, :, 0]
        # Pad token id is already ingored by mask tensor
        ignored_token_ids = [self.eos_token_id] if has_mask else []
        token_accuracy = utils.vocab_parallel_token_accuracy(
            vocab_parallel_logits=logits,
            target=target_token_ids,
            ignored_token_ids=ignored_token_ids,
            ignored_tokens_mask=masked_targets
        )

        # When using target loss only, use num tokens in target as the sample_size
        # See StreamingSrcTgtDataset
        sample_size = (sample["ntokens_target"] if "ntokens_target" in sample else sample["ntokens"])
        logging_output = {
            "loss": loss.data,
            "token_accuracy": token_accuracy.data,
            "ntokens": sample["ntokens"],
            "nsentences": target["target_tokens"].size(0),
            "sample_size": sample_size,
        }
        if "src_tokens" in sample["net_input"] and hasattr(self.task, "eod"):
            logging_output["ndocseps"] = (target["target_tokens"] == self.task.eod).sum()

        if (len(net_output) >= 2 and isinstance(net_output[1], dict) and "inner_states" in net_output[1]):
            with torch.no_grad():
                # yank out the inner states we wish to instrument
                # see transformer_decoder.py TransformerDecoder.extract_features
                emb, *_, actv = net_output[1]["inner_states"]
                assert isinstance(emb, dict), "Expecting the first inner state to be a dict of embedding representations"
                emb["actv"] = actv  # throw on final for code brevity
                for key, value in emb.items():
                    if value is None:
                        # maybe future proofing relative positional embeddings
                        continue
                    value = emb[key]
                    logging_output[f"{key}_norm"] = value.norm(p=2, dim=-1).sum(dtype=torch.float32)

        return loss, sample_size, logging_output
