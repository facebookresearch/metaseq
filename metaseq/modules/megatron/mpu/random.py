# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Taken from:
# https://github.com/ngoyal2707/Megatron-LM/blob/fa6c0860b62e4ed2ac13a513e7d950d72f576a44/megatron/mpu/random.py

import contextlib

import torch
from torch import _C
from torch.cuda import _lazy_call, device as device_ctx_manager

from .initialize import (
    get_data_parallel_rank,
    get_tensor_model_parallel_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)

# Default name for the model parallel rng tracker.
_MODEL_PARALLEL_RNG_TRACKER_NAME = "model-parallel-rng"


def _set_cuda_rng_state(new_state, device=-1):
    """Sets the random number generator state of the current GPU.

    Argumentss:
        new_state (torch.ByteTensor): The desired state
    This function is adapted from PyTorch repo (torch.cuda.set_rng_state)
    with a single change: the input state is not cloned. Cloning caused
    major performance issues for +4 GPU cases.
    """
    if hasattr(_C, "_cuda_setRNGState") and callable(_C._cuda_setRNGState):
        # older PyTorch
        def cb():
            with device_ctx_manager(device):
                _C._cuda_setRNGState(new_state)

    else:
        # newer PyTorch
        if device == -1:
            device = torch.device("cuda")
        elif isinstance(device, str):
            device = torch.device(device)
        elif isinstance(device, int):
            device = torch.device("cuda", device)

        def cb():
            idx = device.index
            if idx is None:
                idx = torch.cuda.current_device()
            default_generator = torch.cuda.default_generators[idx]
            default_generator.set_state(new_state)

    _lazy_call(cb)


def split_tensor_into_1d_equal_chunks(tensor, new_buffer=False):
    """Break a tensor into equal 1D chunks."""
    partition_size = torch.numel(tensor) // get_tensor_model_parallel_world_size()
    start_index = partition_size * get_tensor_model_parallel_rank()
    end_index = start_index + partition_size
    if new_buffer:
        data = torch.empty(
            partition_size,
            dtype=tensor.dtype,
            device=torch.cuda.current_device(),
            requires_grad=False,
        )
        data.copy_(tensor.view(-1)[start_index:end_index])
    else:
        data = tensor.view(-1)[start_index:end_index]
    return data


def gather_split_1d_tensor(tensor):
    """Opposite of above function, gather values from model parallel ranks."""
    numel_gathered = torch.numel(tensor) * get_tensor_model_parallel_world_size()
    gathered = torch.empty(
        numel_gathered,
        dtype=tensor.dtype,
        device=torch.cuda.current_device(),
        requires_grad=False,
    )
    # TODO: This API is experimental in pytorch (as of Feb 2022) and
    # this might break in future pytorch releases. We chose this API
    # as opposed to torch.distributed.all_gather for efficiency reasons.
    # This API calls directly NCCL all-gather versus the former does
    # internal copies and can potentially cause slow down.
    torch.distributed._all_gather_base(
        gathered, tensor, group=get_tensor_model_parallel_group()
    )
    return gathered


class CudaRNGStatesTracker:
    """Tracker for the cuda RNG states.

    Using the `add` method, a cuda rng state is initialized based on
    the input `seed` and is assigned to `name`. Later, by forking the
    rng state, we can perform operations and return to our starting
    cuda state.
    """

    def __init__(self):
        # Map from a string name to the cuda rng state.
        self.states_ = {}
        # Seeds are just for book keeping and ensure no seed is set twice.
        self.seeds_ = set()

    def reset(self):
        """Set to the initial state (no tracker)."""
        self.states_ = {}
        self.seeds_ = set()

    def get_states(self):
        """Get rng states. Copy the dictionary so we have direct
        pointers to the states, not just a pointer to the dictionary."""
        states = {}
        for name in self.states_:
            states[name] = self.states_[name]
        return states

    def set_states(self, states):
        """Set the rng states. For efficiency purposes, we do not check
        the size of seed for compatibility."""
        self.states_ = states

    def add(self, name, seed):
        """Track the rng state."""
        # Check seed is not already used.
        if seed in self.seeds_:
            raise Exception("seed {} already exists".format(seed))
        self.seeds_.add(seed)
        # Check that state is not already defined.
        if name in self.states_:
            raise Exception("cuda rng state {} already exists".format(name))
        # Get the current rng state.
        orig_rng_state = torch.cuda.get_rng_state()
        # Set the new state and store it.
        torch.cuda.manual_seed(seed)
        self.states_[name] = torch.cuda.get_rng_state()
        # Reset rng state to what it was.
        _set_cuda_rng_state(orig_rng_state)

    @contextlib.contextmanager
    def fork(self, name=_MODEL_PARALLEL_RNG_TRACKER_NAME):
        """Fork the cuda rng state, perform operations, and exit with
        the original state."""
        # Check if we have added the state
        if name not in self.states_:
            raise Exception("cuda rng state {} is not added".format(name))
        # Store current rng state.
        orig_cuda_rng_state = torch.cuda.get_rng_state()
        # Set rng state to the desired one
        _set_cuda_rng_state(self.states_[name])
        # Do the stuff we wanted to do.
        try:
            yield
        finally:
            # Update the current rng state for later use.
            self.states_[name] = torch.cuda.get_rng_state()
            # And set the state to the original state we started with.
            _set_cuda_rng_state(orig_cuda_rng_state)


# RNG tracker object.
_CUDA_RNG_STATE_TRACKER = CudaRNGStatesTracker()


def get_cuda_rng_tracker():
    """Get cuda rng tracker."""
    return _CUDA_RNG_STATE_TRACKER


def model_parallel_cuda_manual_seed(seed):
    """Initialize model parallel cuda seed.

    This function should be called after the model parallel is
    initialized. Also, no torch.cuda.manual_seed should be called
    after this function. Basically, this is replacement for that
    function.
    Two set of RNG states are tracked:
        default state: This is for data parallelism and is the same among a
                       set of model parallel GPUs but different across
                       different model paralle groups. This is used for
                       example for dropout in the non-tensor-model-parallel regions.
        tensor-model-parallel state: This state is different among a set of model
                              parallel GPUs, but the same across data parallel
                              groups. This is used for example for dropout in
                              model parallel regions.
    """
    # 2718 is just for fun and any POSITIVE value will work.
    offset = seed + 2718
    tensor_model_parallel_seed = offset + get_tensor_model_parallel_rank()
    # Data parallel gets the original seed.
    data_parallel_seed = seed

    if torch.distributed.get_rank() == 0:
        print(
            "> initializing model parallel cuda seeds on global rank {}, "
            "model parallel rank {}, and data parallel rank {} with "
            "model parallel seed: {} and data parallel seed: {}".format(
                torch.distributed.get_rank(),
                get_tensor_model_parallel_rank(),
                get_data_parallel_rank(),
                tensor_model_parallel_seed,
                data_parallel_seed,
            ),
            flush=True,
        )
    _CUDA_RNG_STATE_TRACKER.reset()
    # Set the default state.
    torch.cuda.manual_seed(data_parallel_seed)
    # and model parallel state.
    _CUDA_RNG_STATE_TRACKER.add(
        _MODEL_PARALLEL_RNG_TRACKER_NAME, tensor_model_parallel_seed
    )
