(section-grindstone-losses)=
# Losses on Metaseq Pipeline

The original Metaseq code relies on a parallel implementation of Cross Entropy Loss to train and finetune a model. The parallel implementation comes from [Megatron's code](https://github.com/ngoyal2707/Megatron-LM/blob/fa6c0860b62e4ed2ac13a513e7d950d72f576a44/megatron/mpu/cross_entropy.py). Recall that Megatron is one of the dependencies of this project. Megatron's parallel cross entropy is called on the forward method implemented by the [VocabParallelCrossEntropyCriterion](https://github.com/microsoft/grindstone/blob/main/metaseq/criterions/vocab_parallel_cross_entropy.py) criterion class.

The cross entropy loss function is adequate for pre-training and finetuning, in which the target is composed only of tokens. However, for Distillation, this loss function is not appropriate.

During Distillation, the target can of composed of one of the following:

* Top K **logprobs** predicted by the teacher model for every token in the output sequence. Logprobs are what OpenAI  teacher models return in their API.

* Top K **logits** predicted by the teacher model for every token in the output sequence. Logits are more common and can be returned by any other teacher model.

For each of these, there is an appropriate loss function that should be used. When we handle a dataset that comes with teacher model **logprobs**, we must use **Soft Cross Entropy loss**, while when handling datasets with **logits**, we must use **MSE loss**.

Below we describe the details for each loss function in our code.

## Cross Entropy Losses

### Megatron's *_VocabParallelCrossEntropy* class

**Forward**

[Megatron's _VocabParallelCrossEntropy](https://github.com/ngoyal2707/Megatron-LM/blob/fa6c0860b62e4ed2ac13a513e7d950d72f576a44/megatron/mpu/cross_entropy.py) receives as input:

- `vocab_parallel_logits`: logits that belong to current model partition with shape `(batch_size, seq_len, vocab_partition_size)`. For instance, considering a vocabulary size of 50272 tokens, the last layer of our model would have 50272 logits. When MP=2, each model partition on each GPU would receive 25136 logits, i.e., `vocab_partition_size`=25136. If MP=1, a single GPU will receive all the 50272 logits at once, i.e, `vocab_partition_size`=50272.

- `target`: target tokens that we want to predict with shape `(batch_size, seq_len)`

Cross entropy is defined as:

$cross\_ent\_loss = - \sum_i^n{t_i * log (p_i)} $ for n classes,

where $t_i$ is the truth label and $p_i$ is the Softmax probability for the $i^{th}$ class.

The target tokens have $p_i = 1$ as they represent the desired predictions, while other tokens in the vocab have their $p_i = 0$ .

Considering softmax defined as:

$softmax = \frac{e^{logits}}{\sum_i^n{e^{logits}}}$

then we have

$cross\_ent\_loss = - t_i * log \hat{p_i} =  - 1 * log \frac{e^{logits}}{\sum_i^n{e^{logits}}}  = - (logits - log(\sum_i^n{e^{logits}})) $



**Backward Formulas**

Let's introduce some notation first.
- For any PyTorch model, assume the final loss of this model is called `final_loss`.
- This model uses some PyTorch modules, given one module, the `forward` function transform `input` to `output` as `output = forward(input)`.

Then when we implement `backward` function, we have

$$grad\_output = \frac{\partial final\_loss}{\partial output}$$


$$grad\_input = \frac{\partial final\_loss}{\partial input}$$

so $$grad\_input = grad\_output * \frac{\partial grad\_output}{\partial grad\_input}$$

so the key to implement `backward` function is to find
$\frac{\partial grad\_output}{\partial grad\_input}$

For cross entropy loss, assume a simple scenario where we only predict one token (instead of tokens with shape `(batch_size, seq_len)`), the output is a scalar indicates the loss, the input is logits with shape `(vocab_size,)`

$$ \frac{\partial output}{\partial input} = \frac{\partial loss}{\partial logit} * \hat{y} - y = SOFTMAX - y$$

where $\hat{y}$ is $softmax(logits)$ or we simply call it $SOFTMAX$, $y$ is one hot encoding of real token.


**Backward Implementation**

Consider Megatron's [Megatron's _VocabParallelCrossEntropy](https://github.com/ngoyal2707/Megatron-LM/blob/fa6c0860b62e4ed2ac13a513e7d950d72f576a44/megatron/mpu/cross_entropy.py) backward implementation.

The inputs to the function are:

- `grad_output`: defined as $grad\_output = \frac{\partial final\_loss}{\partial output} $. Comes from C++ internal calls.
- `softmax`: calculated during forward step for all tokens of the vocab.
- `masked_target_1d`: flattened view of target tokens that belong to the current vocab partitions with shape `(bs * seq_len)`
    - if the token belongs to the worker, the token idx is adjusted by `idx - vocab_start_index`, otherwise, the value is just 0.
    - indicates which elements in grad_2d need to be updated, update by `target_mask.view(-1, 1).float() - 1.0`. The value -1 indicates tokens that belongs to the vocab range of current model parallel worker, and 0 otherwise.
- `target_mask`: 0 if token belogs to partition, 1 otherwise. shape `(bs, seq_len)`,

In following code, we implement the logic of $SOFTMAX - y$. Consider that $y = 1$, therefore `target_mask.view(-1, 1).float() - 1.0` will contain -1 value for tokens that belongs to the vocab range of current model parallel worker, and 0 otherwise.

```python
    def backward(ctx, grad_output):

        # Retreive tensors from the forward path.
        softmax, target_mask, masked_target_1d = ctx.saved_tensors

        # All the inputs have softmax as thier gradient.
        grad_input = softmax

        # For simplicity, work with the 2D gradient.
        partition_vocab_size = softmax.size()[-1]
        grad_2d = grad_input.view(-1, partition_vocab_size)

        grad_2d.scatter_add_(
            dim=-1,
            index=masked_target_1d.unsqueeze(-1),
            src=target_mask.view(-1, 1).float() - 1.0
        )

        # Finally elementwise multiplication with the output gradients.
        grad_input.mul_(grad_output.unsqueeze(dim=-1))

        return grad_input, None
```

### Grindstone's *_VocabParallelSoftCrossEntropy* class

In maths, cross entropy is just H(P, Q), and measures the relation between two probability distributions P and Q. Soft Cross Entropy is standard cross entropy function. We use the word "soft" because in Deep Learning area, the default implementation of cross entropy often assume P (the true distribution) is a one hot encoding, i.e. only one class is the true label, has probability 1, other classes has probability 0.

Megatron's `_VocabParallelCrossEntropy` only works with hard labels (one-hot encodings) since the target is provided as a dense representation (with a single class label per instance).

This class implements a new version of cross entropy loss that considers multiple possible predictions for each token of the target sequence.

This should be used when the source dataset with teacher comes with **logprobs** (not logits).

**Forward**

[Grinstrone's _VocabParallelSoftCrossEntropy](https://github.com/ngoyal2707/Megatron-LM/blob/fa6c0860b62e4ed2ac13a513e7d950d72f576a44/megatron/mpu/cross_entropy.py) receives as input:

- `vocab_parallel_logits`: logits that belong to current model partition with shape `(batch_size, seq_len, vocab_partition_size)`. See Cross Entropy above for more details.

- `target_tokens`: top K target tokens that we want to predict with shape `(batch_size, seq_len, K)`

- `target_predictions`: top K target logprobs that we want our model to mimic `(batch_size, seq_len, K)`

Consider the same cross entropy equation described before:

$cross\_ent\_loss = - \sum_i^n{t_i * log (p_i)} $

for n classes, where

$log (p_i) = log \frac{e^{logits}}{\sum_i^n{e^{logits}}} = (logits - log(\sum_i^n{e^{logits}})) $

Note that now $t_i$ is **not** a one-hot encoding of probabilities anymore, but rather probabilities between 0 and 1.

If we ignore the model parallel details of SoftCrossEntropy implementation, the loss calculation on  forward function is given by:

```python
log_q =  predicted_logits - torch.log(sum_exp_logits).unsqueeze(dim=-1)
probs = target_predictions.exp()
loss = (-log_q * probs).sum(dim=-1)
```

Note that :
- `predicted_logits` represents the correspondent logits for the top K tokens that we receive from the teacher model.
- `target_predictions` are the **logprobs** so we need to perform the `exp()` operation to revert logarithms back to probabilities.

**Backward Formulas**

We use the same gradient formula that is used for Cross Entropy Loss.

$$ \frac{\partial output}{\partial input} = \frac{\partial loss}{\partial logit} = \hat{y} - y = SOFTMAX - y$$

No matter $y$ is a one hot encoding comes from true label, or is a distribution comes from teacher model, the above formula of gradient holds.

**Backward Implementation**

Inputs to the backward function:

- `grad_output`: defined as $grad\_output = \frac{\partial final\_loss}{\partial output} $. Comes from C++ internal calls.
- `softmax`: calculated during forward step for all tokens of the vocab.
- `target_weights`: shape `(bs, seq_len, K)`
    - the value is the probability predicted by the teacher.
- `masked_target`: target tokens that belong to the current vocab partitions with shape `(bs, seq_len, K)`
    - if the token belongs to the worker, adjusted by `idx - vocab_start_index`, otherwise, the value is just 0.
- `target_mask`: shape `(bs, seq_len, K)`
    - Value is `True` or `False`. `True` means the token doesn't belong to the worker.

In following code, we again implement `SOFTMAX - y$`, use `masked_target` to select elements to update, `(-1.0 + target_mask.float()) * target_weight` to decide how much to update.

```python
    def backward(ctx, grad_output):

        # Retreive tensors from the forward path.
        softmax, target_mask, masked_target, target_weights = ctx.saved_tensors

        grad_input = grad_input.scatter_add(
            dim=-1,
            index=masked_target,
            src=(-1.0 + target_mask.float()) * target_weights
        )

        # Finally elementwise multiplication with the output gradients.
        grad_input = grad_input.mul(grad_output.unsqueeze(dim=-1))

        return grad_input, None, None
```

## Grindstone's MSE Loss

### Forward using *_VocabParallelMSELoss* class

Consider the formula for MSE Loss:

$MSE\_loss=  \frac{1}{n} * \sum_1^n{(logits-y)²}$

where target_predictions is `y`. We implement model parallel for $\sum_1^n{(logits-y)²}$. The division by $\frac{1}{n}$ happens outside of the class, in the trainer script.

### Backward Formulas

We want to calculte grad_input, which represents the gradient of the loss w.r.t. the input of the forward function (vocab_parallel_predictions). It is calculated using the chain rule. The gradient of the loss with respect to the input is equal to the dot product of grad_output (the gradient of the loss with respect to the output) and the derivative of loss w.r.t. input.

$$ \frac{\partial output}{\partial logits} = \frac{\partial loss}{\partial logits} = \frac{\partial (logits - y)^T (logits - y)}{\partial logits} = 2 (logits - y)$$

where target_pred is `y'`.

### Backward Implementation in *_VocabParallelMSELoss* class

In `vocab_parallel_mse_loss`, we want to implement a truncated version of MSE loss, which means we only care  difference in some position of $logits-y$. This calculation should only take to account the partition of the vocabulary that belongs to the current GPU device.

The pseudo code is like:
```python
loss = mse(predictions.gather(target_tokens),
            target_pred.gather(target_tokens), reduction="sum")
```

So the $\frac{\partial loss}{\partial input}$ is of the same shape of `predictions`, at positions specified by target_tokens, the values are filled from `2 (predictions - target_pred).gather(target_token)`, other positions are filled with 0.

As we are implementing a model parallel version of truncated MSE, please refer to `vocab_parallel_mse_loss.py` for final implementation.
