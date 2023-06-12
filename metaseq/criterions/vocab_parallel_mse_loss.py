import math
import torch

from metaseq import metrics, utils
from metaseq.criterions import BaseCriterion, register_criterion
from metaseq.criterions.vocab_parallel_cross_entropy import VocabParallelCrossEntropyCriterion

try:
    from megatron.mpu.initialize import get_tensor_model_parallel_group
    from megatron.mpu.initialize import get_tensor_model_parallel_rank
    from megatron.mpu.initialize import get_tensor_model_parallel_world_size
    from megatron.mpu.utils import VocabUtility
    has_megatron_submodule = True
except (ImportError, ModuleNotFoundError):
    has_megatron_submodule = False


class _VocabParallelMSELoss(torch.autograd.Function):

    @staticmethod
    def forward(ctx, vocab_parallel_predictions, target, is_logprobs, debug_mode):

        # Transform logits in logprobs if in logprobs distillation mode.
        if is_logprobs:
            # Sum of exponential of logits along vocab dimension across all GPUs.
            exp_logits = vocab_parallel_predictions  # (bs_size, seq_len, partition_vocab_size)
            torch.exp(vocab_parallel_predictions, out=exp_logits)
            sum_exp_logits = exp_logits.sum(dim=-1)  # (bs_size, seq_len)
            torch.distributed.all_reduce(
                sum_exp_logits, op=torch.distributed.ReduceOp.SUM, group=get_tensor_model_parallel_group()
            )
            if debug_mode:
                torch.distributed.barrier()
            # Calculate softmax
            exp_logits.div_(sum_exp_logits.unsqueeze(dim=-1))
            # Now calculate the logprob as the log of the softmax
            vocab_parallel_predictions = torch.log(exp_logits)

        # -- Get the partition's vocab indexes.
        partition_vocab_size = vocab_parallel_predictions.size()[-1]
        rank = get_tensor_model_parallel_rank()
        world_size = get_tensor_model_parallel_world_size()
        get_vocab_range = VocabUtility.vocab_range_from_per_partition_vocab_size
        vocab_start_index, vocab_end_index = get_vocab_range(partition_vocab_size, rank, world_size)  # range on this GPU

        # Capture target logits/logprobs and tokens separately
        target_predictions = target['target_predictions']  # (batch_size, seq_len, num_k_predictions)
        target_tokens = target['target_tokens']  # (batch_size, seq_len, num_k_predictions)

        # Get top K target tokens for every sequence target
        num_k_predictions = target_tokens.size(2)

        # Create a mask of valid vocab ids (1 means it needs to be masked).
        target_mask = (target_tokens <
                       vocab_start_index) | (target_tokens >= vocab_end_index)  # (batch_size, seq_len, num_k_predictions)
        # Create masked target tensors.
        # Adjust the index of each token to current vocab partition
        masked_target_tokens = target_tokens - vocab_start_index  # (batch_size, seq_len, num_k_predictions)
        # Apply mask to target tokens (fill masked tokens with 0)
        masked_target_tokens.masked_fill_(target_mask, 0)
        masked_target_predictions = target_predictions.masked_fill(target_mask, 0)
        # Get selected_predictions = predictions[target_tokens].
        # For Simplicity, we convert logits/logprobs to a 2-D tensor with size
        # [(batch_size * seq_len), partition_vocab_size] and target to a 2-D tensor of size
        # [(batch_size * seq_len), num_k_predictions].
        predictions_2d = vocab_parallel_predictions.view(
            -1, partition_vocab_size
        )  # ((batch_size * seq_len), partition_vocab_size)
        masked_target_tokens_2d = masked_target_tokens.view(
            -1, num_k_predictions
        )  # ((batch_size * seq_len), num_k_predictions)
        # Select the logits/logprobs for each token of masked_target_2d
        selected_predictions_2d = predictions_2d.gather(
            -1, masked_target_tokens_2d
        )  # ((batch_size * seq_len), num_k_predictions)
        # Adjust to the same shape of target_predictions
        selected_predictions = selected_predictions_2d.view_as(target_predictions)  # (batch_size, seq_len, num_k_predictions)
        # Apply mask to predicted logits/logprobs
        selected_predictions.masked_fill_(target_mask, 0.0)  # (batch_size, seq_len, num_k_predictions)

        # All reduce is needed to get the chunks from other GPUs.
        torch.distributed.all_reduce(
            selected_predictions, op=torch.distributed.ReduceOp.SUM, group=get_tensor_model_parallel_group()
        )
        if debug_mode:
            torch.distributed.barrier()
        # At this point we have the MSE loss sum from all GPUs.
        # There is no need to calculate the mean here as this operation is performed
        # in the trainer script.
        mse_loss = (selected_predictions - target_predictions)**2
        return mse_loss

    @staticmethod
    def backward(ctx, grad_output):
        # TODO
        pass


def parallel_mse_loss(vocab_parallel_predictions, target, is_logprobs, debug_mode=False):
    """Helper function for the MSE w/ model parallel."""
    return _VocabParallelMSELoss.apply(vocab_parallel_predictions, target, is_logprobs, debug_mode)


@register_criterion("vocab_parallel_mse")
class VocabParallelMSELoss(VocabParallelCrossEntropyCriterion):

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
        input = net_output[0].float()  # logits
        target = sample["target"]
        mask_token_id = sample["mask_token_id"]
        has_mask = target["target_tokens"].eq(mask_token_id).any().item()
        is_logprobs = sample["distillation_mode"] == 'logprobs_distillation'
        loss = parallel_mse_loss(input, target, is_logprobs, debug_mode=False)
        if has_mask:
            loss = loss * (target["target_tokens"] != mask_token_id)
        loss = loss.sum()

        # When using target loss only, use num tokens in target as the sample_size
        # See StreamingSrcTgtDataset
        sample_size = (sample["ntokens_target"] if "ntokens_target" in sample else sample["ntokens"])
        logging_output = {
            "loss": loss.data,
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
