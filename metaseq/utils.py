# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
import importlib
import logging
import os
import random
import re
import sys
import warnings
import math
from typing import List, Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F

from metaseq.distributed import utils as distributed_utils

try:
    from amp_C import multi_tensor_l2norm

    multi_tensor_l2norm_available = True
except ImportError:
    multi_tensor_l2norm_available = False


logger = logging.getLogger(__name__)


def split_paths(paths: str) -> List[str]:
    return paths.split(os.pathsep) if "://" not in paths else paths.split("|")


def apply_to_sample(f, sample):
    if hasattr(sample, "__len__") and len(sample) == 0:
        return {}

    def _apply(x):
        if torch.is_tensor(x):
            return f(x)
        elif isinstance(x, dict):
            return {key: _apply(value) for key, value in x.items()}
        elif isinstance(x, list):
            return [_apply(x) for x in x]
        elif isinstance(x, tuple):
            return tuple(_apply(x) for x in x)
        elif isinstance(x, set):
            return {_apply(x) for x in x}
        else:
            return x

    return _apply(sample)


def move_to_cuda(sample, device=None):
    device = device or torch.cuda.current_device()

    def _move_to_cuda(tensor):
        # non_blocking is ignored if tensor is not pinned, so we can always set
        # to True (see github.com/PyTorchLightning/pytorch-lightning/issues/620)
        return tensor.to(device=device, non_blocking=True)

    return apply_to_sample(_move_to_cuda, sample)


def move_to_cpu(sample, cast_to_fp32=True):
    def _move_to_cpu(tensor):
        # PyTorch has poor support for half tensors (float16) on CPU.
        # Move any such tensors to float32.
        if cast_to_fp32 and tensor.dtype in {torch.bfloat16, torch.float16}:
            tensor = tensor.to(dtype=torch.float32)
        return tensor.cpu()

    return apply_to_sample(_move_to_cpu, sample)


def load_align_dict(replace_unk):
    if replace_unk is None:
        align_dict = None
    elif isinstance(replace_unk, str) and len(replace_unk) > 0:
        # Load alignment dictionary for unknown word replacement if it was passed as an argument.
        align_dict = {}
        with open(replace_unk, "r") as f:
            for line in f:
                cols = line.split()
                align_dict[cols[0]] = cols[1]
    else:
        # No alignment dictionary provided but we still want to perform unknown word replacement by copying the
        # original source word.
        align_dict = {}
    return align_dict


def make_positions(tensor, padding_idx: int):
    """Replace non-padding symbols with their position numbers.

    Position numbers begin at padding_idx+1. Padding symbols are ignored.
    """
    # The series of casts and type-conversions here are carefully
    # balanced to both work with ONNX export and XLA. In particular XLA
    # prefers ints, cumsum defaults to output longs, and ONNX doesn't know
    # how to handle the dtype kwarg in cumsum.
    mask = tensor.ne(padding_idx).int()
    return (torch.cumsum(mask, dim=1).type_as(mask) * mask).long() + padding_idx


def strip_pad(tensor, pad):
    return tensor[tensor.ne(pad)]


def item(tensor):
    if hasattr(tensor, "item"):
        return tensor.item()
    if hasattr(tensor, "__getitem__"):
        return tensor[0]
    return tensor


def multi_tensor_l2_total_norm(grads, chunk_size=2048 * 32) -> torch.Tensor:
    per_device_grads = {}
    norms = []
    for grad in grads:
        device = grad.device
        cur_device_grads = per_device_grads.get(device)
        if cur_device_grads is None:
            cur_device_grads = []
            per_device_grads[device] = cur_device_grads
        cur_device_grads.append(grad)
    for device in per_device_grads.keys():
        cur_device_grads = per_device_grads[device]
        if device.type == "cuda":
            # TODO(msb) return has_inf
            has_inf = torch.zeros((1, 1), dtype=torch.int, device=device)
            with torch.cuda.device(device):
                norm = multi_tensor_l2norm(
                    chunk_size, has_inf, [cur_device_grads], False
                )
            norms.append(norm[0].to(torch.cuda.current_device()))
        else:
            norms += [torch.norm(g, p=2, dtype=torch.float32) for g in cur_device_grads]
    total_norm = torch.norm(torch.stack(norms))
    return total_norm


norm_type2_reduce_op = {"l2": dist.ReduceOp.SUM, "inf": dist.ReduceOp.MAX}


@torch.no_grad()
def clip_grad_norm_(
    params, max_norm, norm_type="l2", aggregate_norm_fn=None, device=None
) -> torch.Tensor:
    def grad_exists(p):
        return p is not None and getattr(p, "grad", None) is not None

    if isinstance(params, torch.Tensor):
        params = [params]
    params = list(params)
    params = list(filter(grad_exists, params))
    grads, sharded_grads = [], []

    if device is None:
        if torch.cuda.is_available():
            # param/grads could be on CPU if using CPU offloading, but we want
            # everything on GPU if possible
            device = torch.device("cuda:{}".format(torch.cuda.current_device()))
        elif len(params) > 0:
            device = params[0].device  # could be "xla"
        else:
            device = torch.device("cpu")

    def norm(t, n_type):
        if n_type == "l2":
            return torch.norm(t, p=2, dtype=torch.float32)
        elif n_type == "inf":
            return torch.norm(t, p=float("inf"), dtype=torch.float32)
        else:
            raise ValueError(
                f"Invalid clip_norm_type: {n_type}! Please pass either 'l2' or 'inf'!"
            )

    for p in params:
        if hasattr(p, "_is_sharded"):
            sharded_grads.append(p.grad.detach())
        else:
            grads.append(p.grad.detach())

    if len(grads) == 0:
        total_norm = torch.tensor(0.0, dtype=torch.float32, device=device)
    elif len(grads) == 1:
        total_norm = norm(grads[0], norm_type)
    else:
        if (
            multi_tensor_l2norm_available
            and norm_type == "l2"
            and grads[0].dtype != torch.bfloat16
        ):
            total_norm = multi_tensor_l2_total_norm(grads)
        else:
            if (
                torch.cuda.is_available()
                and norm_type == "l2"
                and grads[0].dtype != torch.bfloat16
            ):
                warnings.warn(
                    "amp_C fused kernels unavailable, disabling multi_tensor_l2norm; "
                    "you may get better performance by installing NVIDIA's apex library"
                )
            total_norm = norm(
                torch.stack([norm(g, norm_type) for g in grads]), norm_type
            )

    # calculate split_norm and all_reduce with other workers
    norms = [total_norm]
    for split_grads in [sharded_grads]:
        if len(split_grads) == 0:
            continue
        split_norm = norm(
            torch.stack([norm(g, norm_type) for g in split_grads]), norm_type
        )
        if dist.is_initialized():
            reduce_op = norm_type2_reduce_op[norm_type]
            if norm_type == "l2":
                split_norm.pow_(2)
            dist.all_reduce(
                split_norm,
                group=distributed_utils.get_data_parallel_group(),
                op=reduce_op,
            )
            if norm_type == "l2":
                split_norm.sqrt_()
        norms.append(split_norm)

    if len(norms) > 1:
        total_norm = norm(torch.stack(norms), norm_type)

    if aggregate_norm_fn is not None:
        total_norm = aggregate_norm_fn(total_norm)

    if max_norm > 0:
        max_norm = float(max_norm)
        clip_coef = (max_norm / (total_norm + 1e-6)).clamp_(max=1)
        for g in grads + sharded_grads:
            g.mul_(clip_coef)
    return total_norm


def fill_with_neg_inf(t):
    """FP16-compatible function that fills a tensor with -inf."""
    return t.float().fill_(float("-inf")).type_as(t)


def _match_types(arg1, arg2):
    """Convert the numerical argument to the same type as the other argument"""

    def upgrade(arg_number, arg_structure):
        if isinstance(arg_structure, tuple):
            return tuple([arg_number] * len(arg_structure))
        elif isinstance(arg_structure, dict):
            arg = copy.deepcopy(arg_structure)
            for k in arg:
                arg[k] = upgrade(arg_number, arg_structure[k])
            return arg
        else:
            return arg_number

    if isinstance(arg1, float) or isinstance(arg1, int):
        return upgrade(arg1, arg2), arg2
    elif isinstance(arg2, float) or isinstance(arg2, int):
        return arg1, upgrade(arg2, arg1)

    return arg1, arg2


def resolve_max_positions(*args):
    """Resolve max position constraints from multiple sources."""

    def map_value_update(d1, d2):
        updated_value = copy.deepcopy(d1)
        for key in d2:
            if key not in updated_value:
                updated_value[key] = d2[key]
            else:
                updated_value[key] = min(d1[key], d2[key])
        return updated_value

    def nullsafe_min(l):
        minim = None
        for item in l:
            if minim is None:
                minim = item
            elif item is not None and item < minim:
                minim = item
        return minim

    max_positions = None
    for arg in args:
        if max_positions is None:
            max_positions = arg
        elif arg is not None:
            max_positions, arg = _match_types(max_positions, arg)
            if isinstance(arg, float) or isinstance(arg, int):
                max_positions = min(max_positions, arg)
            elif isinstance(arg, dict):
                max_positions = map_value_update(max_positions, arg)
            else:
                max_positions = tuple(map(nullsafe_min, zip(max_positions, arg)))

    return max_positions


def import_user_module(args):
    module_path = getattr(args, "user_dir", None)
    if module_path is not None:
        module_path = os.path.abspath(args.user_dir)
        if not os.path.exists(module_path) and not os.path.isfile(
            os.path.dirname(module_path)
        ):
            metaseq_rel_path = os.path.join(os.path.dirname(__file__), args.user_dir)
            if os.path.exists(metaseq_rel_path):
                module_path = metaseq_rel_path
            else:
                metaseq_rel_path = os.path.join(
                    os.path.dirname(__file__), "..", args.user_dir
                )
                if os.path.exists(metaseq_rel_path):
                    module_path = metaseq_rel_path
                else:
                    raise FileNotFoundError(module_path)

        # ensure that user modules are only imported once
        import_user_module.memo = getattr(import_user_module, "memo", set())
        if module_path not in import_user_module.memo:
            import_user_module.memo.add(module_path)

            module_parent, module_name = os.path.split(module_path)
            if module_name not in sys.modules:
                sys.path.insert(0, module_parent)
                importlib.import_module(module_name)
            else:
                raise ImportError(
                    "Failed to import --user-dir={} because the corresponding module name "
                    "({}) is not globally unique. Please rename the directory to "
                    "something unique and try again.".format(module_path, module_name)
                )


def softmax(x, dim: int):
    return F.softmax(x, dim=dim, dtype=torch.float32)


def log_softmax(x, dim: int):
    return F.log_softmax(x, dim=dim, dtype=torch.float32)


def get_perplexity(loss, round=2, base=2):
    from metaseq.logging.meters import safe_round

    if loss is None:
        return 0.0
    try:
        return safe_round(base**loss, round)
    except OverflowError:
        return float("inf")


def has_parameters(module):
    try:
        next(module.parameters())
        return True
    except StopIteration:
        return False


def get_rng_state():
    state = {"torch_rng_state": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda_rng_state"] = torch.cuda.get_rng_state()
    return state


def set_rng_state(state):
    torch.set_rng_state(state["torch_rng_state"])
    if torch.cuda.is_available():
        torch.cuda.set_rng_state(state["cuda_rng_state"])


class set_torch_seed(object):
    def __init__(self, seed):
        assert isinstance(seed, int)
        self.rng_state = get_rng_state()

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        set_rng_state(self.rng_state)


class CudaEnvironment(object):
    def __init__(self):
        cur_device = torch.cuda.current_device()
        prop = torch.cuda.get_device_properties("cuda:{}".format(cur_device))
        self.name = prop.name
        self.major = prop.major
        self.minor = prop.minor
        self.total_memory_in_GB = prop.total_memory / 1024 / 1024 / 1024

    @staticmethod
    def pretty_print_cuda_env_list(cuda_env_list):
        """
        Given a list of CudaEnviorments, pretty print them
        """
        num_workers = len(cuda_env_list)
        center = "CUDA enviroments for all {} workers".format(num_workers)
        banner_len = 40 - len(center) // 2
        first_line = "*" * banner_len + center + "*" * banner_len
        logger.info(first_line)
        for r, env in enumerate(cuda_env_list):
            logger.info(
                "rank {:3d}: ".format(r)
                + "capabilities = {:2d}.{:<2d} ; ".format(env.major, env.minor)
                + "total memory = {:.3f} GB ; ".format(env.total_memory_in_GB)
                + "name = {:40s}".format(env.name)
            )
        logger.info(first_line)


# TODO[susan]: Move this to metaseq-internal where it is currently used
def remove_prefix(text: str, prefix: str):
    if text.startswith(prefix):
        return text[len(prefix) :]
    return text


# TODO[susan]: Move this to metaseq-internal where it is currently used
def print_r0(*x, file=None):
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        print(*x, file=file, flush=True)


def get_random_port():
    old_state = random.getstate()
    random.seed()
    port = random.randint(10000, 20000)
    random.setstate(old_state)
    return port


def floating_point_precision_convertor(
    x, fp16: bool, memory_efficient_fp16: bool, bf16: bool
):
    """
    Convert a tensor x into the desired dtype.

    Also sanity checks combinations of options.
    """
    if memory_efficient_fp16:
        assert not bf16, "Do not combined bf16 with memory_efficient_fp16."
    if bf16:
        assert fp16, "Setting --bf16 requires also setting --fp16 for legacy reasons."
    if not fp16 and not bf16:
        return x
    if not memory_efficient_fp16:
        # original parameters stay in fp32 and are converted by fairscale
        return x
    elif bf16:
        return x.bfloat16()
    else:
        return x.half()


def get_model_init_dtype(args):
    if getattr(args, "memory_efficient_fp16", False) or getattr(
        args, "inference", False
    ):
        return torch.bfloat16 if getattr(args, "bf16", False) else torch.half
    return torch.float32


def get_precise_epoch(epoch: Optional[int], count: int, iterator_size: int) -> float:
    return (
        epoch - 1 + (count + 1) / float(iterator_size)
        if epoch is not None and iterator_size > 0
        else None
    )


def tokenize_line(line):
    line = re.compile(r"\s+").sub(" ", line)
    line = line.strip()
    return line.split()


def init_method_normal(sigma, truncate_init=False):
    """Init method based on N(0, sigma)."""

    def init_(tensor):
        if sigma <= 1e-8:  # effectively 0
            return torch.nn.init.zeros_(tensor)
        if truncate_init:
            return torch.nn.init.trunc_normal_(
                tensor, mean=0.0, std=sigma, a=-3 * sigma, b=3 * sigma
            )
        else:
            return torch.nn.init.normal_(tensor, mean=0.0, std=sigma)

    return init_


def scaled_init_method_normal(sigma, num_layers, truncate_init=False):
    """Init method based on N(0, sigma/sqrt(2*num_layers)."""
    std = sigma / math.sqrt(2.0 * num_layers)

    def init_(tensor):
        if sigma <= 1e-8:  # effectively 0
            return torch.nn.init.zeros_(tensor)
        if truncate_init:
            return torch.nn.init.trunc_normal_(
                tensor, mean=0.0, std=std, a=-3 * std, b=3 * std
            )
        else:
            return torch.nn.init.normal_(tensor, mean=0.0, std=std)

    return init_
