# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import functools
import threading
import weakref
from contextlib import contextmanager
from typing import Any, Dict, Generator, Optional, Tuple

import torch
import torch.nn as nn
import torch.utils.checkpoint as torch_checkpoint
from fairscale.nn.checkpoint.checkpoint_utils import patch_batchnorm
from fairscale.utils.containers import (
    pack_kwargs,
    split_non_tensors,
    unpack_kwargs,
    unpack_non_tensors,
)
from megatron import mpu
from torch import Tensor

from metaseq.distributed.utils import get_model_parallel_group


# https://docs.python.org/3/library/threading.html#thread-local-data
# Manage the checkpoint context with thread-local data.
class ThreadLocal(threading.local):
    def __init__(self) -> None:
        self.is_checkpointing = False
        self.is_recomputing = False
        self.is_checkpointing_disabled = False


thread_local = ThreadLocal()


@contextmanager
def disable_checkpointing() -> Generator[None, None, None]:
    """Makes :func:`is_checkpointing_disabled` return :data:`True` within a context."""
    orig = thread_local.is_checkpointing_disabled
    thread_local.is_checkpointing_disabled = True
    try:
        yield
    finally:
        thread_local.is_checkpointing_disabled = orig


@contextmanager
def enable_checkpointing() -> Generator[None, None, None]:
    """Makes :func:`is_checkpointing` return :data:`True` within a context."""
    orig = thread_local.is_checkpointing
    thread_local.is_checkpointing = True
    try:
        yield
    finally:
        thread_local.is_checkpointing = orig


@contextmanager
def enable_recomputing() -> Generator[None, None, None]:
    """Makes :func:`is_recomputing` return :data:`True` within a context."""
    orig = thread_local.is_recomputing
    thread_local.is_recomputing = True
    try:
        yield
    finally:
        thread_local.is_recomputing = orig


def is_checkpointing() -> bool:
    """Whether the current forward propagation is under checkpointing.

    Returns:
        bool: :data:`True` if it's under checkpointing.

    """
    return thread_local.is_checkpointing


def is_recomputing() -> bool:
    """Whether the current forward propagation is under checkpoint
    recomputation. Use this to prevent duplicated side-effects at forward
    propagation::

        class Counter(nn.Module):
            def __init__(self):
                super().__init__()
                self.counter = 0

            def forward(self, input):
                if not is_recomputing():
                    self.counter += 1
                return input

    Returns:
        bool: :data:`True` if it's under checkpoint recomputation.
    """
    return thread_local.is_recomputing


def checkpoint_wrapper(
    module: nn.Module,
    offload_to_cpu: bool = False,
    distribute_checkpointed_activations: bool = False,
) -> nn.Module:
    """
    A friendlier wrapper for performing activation checkpointing.

    Compared to the PyTorch version, this version:

        - wraps an nn.Module, so that all subsequent calls will use checkpointing
        - handles keyword arguments in the forward
        - handles non-Tensor outputs from the forward
        - supports offloading activations to CPU

    Usage::

        checkpointed_module = checkpoint_wrapper(my_module, offload_to_cpu=True)
        a, b = checkpointed_module(x, y=3, z=torch.Tensor([1]))

    To understand the benefits of checkpointing and the `offload_to_cpu` flag,
    let's divide activations into 2 types: inner activations and outer
    activations w.r.t. the checkpointed modules. The inner ones are saved
    by activation checkpointing, the outer ones are saved by offload_to_cpu.

    In terms of GPU memory savings:

        - When inner ones are large in size and outer ones are small,
          checkpointing helps a lot, offload_to_cpu may help a little.
        - When inner ones are small and outer ones are large,
          checkpointing helps little, offload_to_cpu helps a lot.
        - When both inner and outer are large, both help and the
          benefit is additive.

    ..Note::

        The first and last layers are not likely to benefit from the `offload_to_cpu` flag
        because (1) there are typically other references to the first layer's input, so
        the GPU memory won't be freed; (2) the input to the last layer is immediately
        used by the backward pass and won't result in memory savings.

    Args:
        module (nn.Module):
            The module to be wrapped
        offload_to_cpu (bool):
            Whether to offload activations to CPU.

    Returns:
        (nn.Module):
            Wrapped module
    """
    # Patch the batchnorm layers in case there are any in this module.
    patch_batchnorm(module)

    # The use of weakref here is to prevent creating a ref cycle: m -> m.forward -> m.
    # When such cycle exists, gc won't collect the module when the module is freed.
    # That causes GPU memory to be leaked. See the unit test for how we catch that.
    #
    # We prefer this over a class wrapper since the class wrapper would have to
    # proxy a lot of fields and methods.
    module.forward = functools.partial(  # type: ignore
        _checkpointed_forward,
        type(module).forward,
        weakref.ref(module),
        offload_to_cpu,
        distribute_checkpointed_activations,
    )
    return module


def _checkpointed_forward(
    original_forward: Any,
    weak_self: Any,
    offload_to_cpu: bool,
    distribute_checkpointed_activations: bool,
    *args: Any,
    **kwargs: Any
) -> Any:
    module = weak_self()

    # If gradients are disabled, just use original `.forward()` method directly.
    if not torch.is_grad_enabled() or thread_local.is_checkpointing_disabled:
        return original_forward(module, *args, **kwargs)

    # Autograd Functions in PyTorch work best with positional args, since
    # the backward must return gradients (or None) for every input argument.
    # We can flatten keyword arguments to make this easier.
    args = (module,) + args
    kwarg_keys, flat_args = pack_kwargs(*args, **kwargs)
    parent_ctx_dict: Dict[str, Any] = {
        "offload": offload_to_cpu,
        "distribute_checkpointed_activations": distribute_checkpointed_activations,
    }
    # Dummy tensor with grad is used to ensure the backward pass is called. This is needed
    # when original_forward's input are non-tensor (i.e. a tuple). Using this dummy tensor
    # avoids requiring users to set their input tensors's requires_grad flag. In the case
    # of tuple type inputs, setting the flag won't even trigger the backward pass.
    #
    # One implication of this is that since we always feed in a dummy tensor
    # needing grad, then the output will always require grad, even if it originally
    # wouldn't, such as if the module and original input both do not require grad.
    # We get around this by saving the desired requires_grad value in output and
    # detaching the output if needed.
    output = CheckpointFunction.apply(
        torch.tensor([], requires_grad=True),
        original_forward,
        parent_ctx_dict,
        kwarg_keys,
        *flat_args
    )
    output_requires_grad = parent_ctx_dict["output_requires_grad"]
    if not isinstance(output, torch.Tensor):
        # If output should not require grad, then detach it, since otherwise it will
        # always have requires_grad = True due to our dummy tensor input above that
        # requires_grad
        output = [x.detach() if not output_requires_grad else x for x in output]

        packed_non_tensor_outputs = parent_ctx_dict["packed_non_tensor_outputs"]
        if packed_non_tensor_outputs:
            output = unpack_non_tensors(output, packed_non_tensor_outputs)

    else:
        # If output should not require grad, then detach it, since otherwise it will
        # always have requires_grad = True due to our dummy tensor input above that
        # requires_grad
        if not output_requires_grad:
            output = output.detach()

    return output


def get_rng_state() -> Dict[str, Any]:
    state = {"torch_rng_state": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda_rng_state"] = torch.cuda.get_rng_state()
    return state


def set_rng_state(state: Dict[str, Any]) -> None:
    torch.set_rng_state(state["torch_rng_state"])
    if torch.cuda.is_available():
        torch.cuda.set_rng_state(state["cuda_rng_state"])


def is_autocast_enabled() -> bool:
    """Similar to torch.is_autocast_enabled, but compatible with torch 1.5.1"""
    if hasattr(torch, "is_autocast_enabled"):
        return torch.is_autocast_enabled()
    return False


@contextmanager
def autocast(enabled: bool) -> Generator:
    """Similar to torch.cuda.amp.autocast, but compatible with torch 1.5.1"""
    if enabled:
        with torch.cuda.amp.autocast(enabled):
            yield
    else:
        yield


class CheckpointFunction(torch.autograd.Function):
    """Similar to the torch version, but support non-Tensor outputs.

    The caller is expected to provide a dict (*parent_ctx_dict*) that will hold
    the non-Tensor outputs. These should be combined with the Tensor *outputs*
    by calling :func:`unpack_non_tensors`.
    """

    @staticmethod
    def forward(  # type: ignore
        ctx: Any,
        dummy_tensor_requires_grad: torch.Tensor,
        run_function: Any,
        parent_ctx_dict: Dict[str, Any],
        kwarg_keys: Tuple[str, ...],
        *args: Any,
        **kwargs: Any
    ) -> Any:
        torch_checkpoint.check_backward_validity(args)

        ctx.run_function = run_function
        ctx.kwarg_keys = kwarg_keys
        ctx.fwd_rng_state = get_rng_state()

        ctx.is_model_parallel = get_model_parallel_group() is not None
        # Megatron's dropout random state
        if ctx.is_model_parallel:
            ctx.fwd_cuda_rng_state_tracker = (
                mpu.random.get_cuda_rng_tracker().get_states()
            )
        ctx.had_autocast_in_fwd = is_autocast_enabled()

        tensor_inputs, packed_non_tensor_inputs = split_non_tensors(args)
        if parent_ctx_dict["offload"]:
            ctx.fwd_device = tuple(x.device for x in tensor_inputs)
            ctx.grad_requirements = tuple(x.requires_grad for x in tensor_inputs)
            tensor_inputs = tuple(x.to("cpu", non_blocking=True) for x in tensor_inputs)
        else:
            ctx.fwd_device, ctx.grad_requirements = None, None

        with torch.no_grad(), enable_checkpointing():
            unpacked_args, unpacked_kwargs = unpack_kwargs(kwarg_keys, args)
            outputs = run_function(*unpacked_args, **unpacked_kwargs)
            the_module = unpacked_args[0]
        ctx.distribute_checkpointed_activations = (
            parent_ctx_dict["distribute_checkpointed_activations"]
            and ctx.is_model_parallel
        )
        if ctx.distribute_checkpointed_activations:
            # HACK [TODO: naman] currently only distributing the first tensor.
            # second tensor for usual decoder models is just attention mask, which is seq_len * seq_len
            # and not big enough to distribute over.
            ctx.tensor_input_0_shape = tensor_inputs[0].data.shape
            tensor_inputs[0].data = tensor_inputs[0].data.contiguous()
            tensor_inputs[0].data = mpu.random.split_tensor_into_1d_equal_chunks(
                tensor_inputs[0].data, new_buffer=True
            )
        ctx.save_for_backward(*tensor_inputs)

        ctx.packed_non_tensor_inputs = packed_non_tensor_inputs
        # Because we run with torch.no_grad(), we can't actually access
        # outputs.requires_grad. Instead, we manually compute it by
        # checking if either the input or the module needs grads
        parameters = list(the_module.parameters())

        # If the module is wrapped by FlattenParamsWrapper, then the
        # parameters would have been deleted. If so, we need to access
        # the views into the flattened parameters.
        if hasattr(the_module, "_unflattened_param_views"):
            parameters += the_module._unflattened_param_views

        output_requires_grad = any(param.requires_grad for param in parameters) or any(
            x.requires_grad for x in tensor_inputs
        )
        parent_ctx_dict["output_requires_grad"] = output_requires_grad

        if not isinstance(outputs, torch.Tensor):
            # Autograd Functions don't like non-Tensor outputs. We can split the
            # non-Tensor and Tensor outputs, returning the former by reference
            # through *parent_ctx_dict* and returning the latter directly.
            outputs, packed_non_tensor_outputs = split_non_tensors(outputs)
            parent_ctx_dict["packed_non_tensor_outputs"] = packed_non_tensor_outputs

        return outputs

    @staticmethod
    def backward(ctx: Any, *args: Any) -> Tuple[Optional[Tensor], ...]:
        if not torch.autograd._is_checkpoint_valid():
            raise RuntimeError(
                "Checkpointing is not compatible with .grad(), please use .backward() if possible"
            )

        tensor_inputs: Tuple = ctx.saved_tensors
        if ctx.distribute_checkpointed_activations:
            tensor_inputs[0].data = mpu.random.gather_split_1d_tensor(
                tensor_inputs[0].data
            )
            tensor_inputs[0].data = tensor_inputs[0].data.view(ctx.tensor_input_0_shape)

        tensor_inputs = torch_checkpoint.detach_variable(tensor_inputs)
        if ctx.fwd_device is not None:
            tensor_inputs = tuple(
                t.to(ctx.fwd_device[i], non_blocking=True)
                for i, t in enumerate(tensor_inputs)
            )
            for i, need_grad in enumerate(ctx.grad_requirements):
                tensor_inputs[i].requires_grad = need_grad
        inputs = unpack_non_tensors(tensor_inputs, ctx.packed_non_tensor_inputs)

        # Store the current states.
        bwd_rng_state = get_rng_state()
        if ctx.is_model_parallel:
            bwd_cuda_rng_state_tracker = mpu.random.get_cuda_rng_tracker().get_states()
            mpu.random.get_cuda_rng_tracker().set_states(ctx.fwd_cuda_rng_state_tracker)

        # Set the states to what it used to be before the forward pass.
        set_rng_state(ctx.fwd_rng_state)

        with torch.enable_grad(), enable_recomputing(), autocast(
            ctx.had_autocast_in_fwd
        ):
            unpacked_args, unpacked_kwargs = unpack_kwargs(ctx.kwarg_keys, inputs)
            outputs = ctx.run_function(*unpacked_args, **unpacked_kwargs)
            tensor_outputs, _ = split_non_tensors(outputs)

        # Set the states back to what it was at the start of this function.
        set_rng_state(bwd_rng_state)
        if ctx.is_model_parallel:
            mpu.random.get_cuda_rng_tracker().set_states(bwd_cuda_rng_state_tracker)

        # Run backward() with only Tensors that require grad
        outputs_with_grad = []
        args_with_grad = []
        for i in range(len(tensor_outputs)):
            if tensor_outputs[i].requires_grad:
                outputs_with_grad.append(tensor_outputs[i])
                args_with_grad.append(args[i])

        if len(outputs_with_grad) == 0:
            raise RuntimeError(
                "None of the outputs have requires_grad=True, "
                "this checkpoint() is not necessary"
            )

        torch.autograd.backward(outputs_with_grad, args_with_grad)

        grads = tuple(
            inp.grad if isinstance(inp, torch.Tensor) else None for inp in inputs
        )

        return (None, None, None, None) + grads
