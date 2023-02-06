# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import io
import logging
import os
import pickle
import random
import signal
import socket
import struct
import subprocess
from argparse import Namespace
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional

import torch
import torch.distributed as dist
from omegaconf import open_dict

from metaseq.dataclass.configs import DistributedTrainingConfig, MetaseqConfig

# Flag to indicate if we're using Megatron
# NOTE: this is a temporary hack until we move away from Megatron's model parallel init
_USE_MEGATRON = False


logger = logging.getLogger(__name__)


def is_master(cfg: DistributedTrainingConfig):
    return cfg.distributed_rank == 0


def infer_init_method(cfg: DistributedTrainingConfig, force_distributed=False):
    if cfg.distributed_init_method is not None:
        return

    if cfg.distributed_port > 0:
        # we can determine the init method automatically for Slurm
        _infer_slurm_init(cfg)
    elif all(
        key in os.environ
        for key in ["MASTER_ADDR", "MASTER_PORT", "WORLD_SIZE", "RANK", "LOCAL_RANK"]
    ):
        # support torch.distributed.launch
        _infer_torch_distributed_launch_init(cfg)
    elif cfg.distributed_world_size > 1 or force_distributed:
        # fallback for single node with multiple GPUs
        _infer_single_node_init(cfg)

    if not cfg.distributed_no_spawn:
        with open_dict(cfg):
            cfg.distributed_num_procs = min(
                torch.cuda.device_count(), cfg.distributed_world_size
            )


def _infer_torch_distributed_launch_init(cfg: DistributedTrainingConfig):
    cfg.distributed_init_method = "env://"
    cfg.distributed_world_size = int(os.environ["WORLD_SIZE"])
    cfg.distributed_rank = int(os.environ["RANK"])
    cfg.device_id = int(os.environ["LOCAL_RANK"])
    # processes are created by torch.distributed.launch
    cfg.distributed_no_spawn = True


def _infer_slurm_init(cfg: DistributedTrainingConfig):
    node_list = os.environ.get("SLURM_STEP_NODELIST")
    if node_list is None:
        node_list = os.environ.get("SLURM_JOB_NODELIST")
    if node_list is not None:
        try:
            hostnames = subprocess.check_output(
                ["scontrol", "show", "hostnames", node_list]
            )
            host = hostnames.split()[0].decode("utf-8")
        except subprocess.CalledProcessError as e:  # scontrol failed
            raise e
        except FileNotFoundError:  # Slurm is not installed
            # if we're in a container, then maybe MASTER_ADDR is set
            host = os.environ.get("MASTER_ADDR", None)
        if host is None:
            return
        cfg.distributed_init_method = "barrierlesstcpr://{host}:{port}".format(
            host=host, port=cfg.distributed_port
        )
        nnodes = int(os.environ.get("SLURM_NNODES"))
        ntasks_per_node = os.environ.get("SLURM_NTASKS_PER_NODE")
        if ntasks_per_node is not None:
            ntasks_per_node = int(ntasks_per_node)
        else:
            ntasks = int(os.environ.get("SLURM_NTASKS"))
            nnodes = int(os.environ.get("SLURM_NNODES"))
            assert ntasks % nnodes == 0
            ntasks_per_node = int(ntasks / nnodes)
        if ntasks_per_node == 1:
            gpus_per_node = torch.cuda.device_count()
            node_id = int(os.environ.get("SLURM_NODEID"))
            cfg.distributed_rank = node_id * gpus_per_node
            cfg.distributed_world_size = nnodes * gpus_per_node
        else:
            assert ntasks_per_node == torch.cuda.device_count()
            cfg.distributed_world_size = ntasks_per_node * nnodes
            cfg.distributed_no_spawn = True
            cfg.distributed_rank = int(os.environ.get("SLURM_PROCID"))
            cfg.device_id = int(os.environ.get("SLURM_LOCALID"))


def _infer_single_node_init(cfg: DistributedTrainingConfig):
    assert (
        cfg.distributed_world_size <= torch.cuda.device_count()
    ), f"world size is {cfg.distributed_world_size} but have {torch.cuda.device_count()} available devices"
    port = random.randint(10000, 20000)
    cfg.distributed_init_method = "tcp://localhost:{port}".format(port=port)


def distributed_init(cfg: MetaseqConfig):
    if isinstance(cfg, Namespace):
        from metaseq.dataclass.utils import convert_namespace_to_omegaconf

        cfg = convert_namespace_to_omegaconf(cfg)

    # silence torch's distributed initialization info
    logging.getLogger("torch.distributed.distributed_c10d").setLevel(logging.WARNING)

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        logger.warning("Distributed is already initialized, cannot initialize twice!")
    else:
        logger.debug(
            "distributed init (rank {}): {}".format(
                cfg.distributed_training.distributed_rank,
                cfg.distributed_training.distributed_init_method,
            )
        )
        dist.init_process_group(
            backend=cfg.distributed_training.distributed_backend,
            init_method=cfg.distributed_training.distributed_init_method,
            world_size=cfg.distributed_training.distributed_world_size,
            rank=cfg.distributed_training.distributed_rank,
        )
        logger.info(
            "initialized host {} as rank {}".format(
                socket.gethostname(),
                cfg.distributed_training.distributed_rank,
            )
        )

        # perform a dummy all-reduce to initialize the NCCL communicator
        if torch.cuda.is_available():
            dist.all_reduce(torch.zeros(1).cuda())

    cfg.distributed_training.distributed_rank = torch.distributed.get_rank()

    # set global log level
    if is_master(cfg.distributed_training):
        logging.getLogger().setLevel(logging.INFO)
    else:
        logging.getLogger().setLevel(logging.WARNING)

    nodelist = os.environ.get("SLURM_STEP_NODELIST")
    if nodelist:
        logger.info(f"SLURM nodelist: {nodelist}")

    if (
        getattr(cfg.model, "arch", None) == "transformer_lm_megatron"
        or cfg.common.model_parallel_size > 1
    ):
        try:
            from megatron.mpu import (
                initialize_model_parallel,
                model_parallel_cuda_manual_seed,
            )

            # Following initializes memory buffer in Megatron code which uses
            # buffered memory for tensor parallel GPU comms protocols
            from megatron.global_vars import (
                _GLOBAL_MEMORY_BUFFER,
                _set_global_memory_buffer,
            )
        except ImportError:
            raise ImportError(
                "\n\nPlease install megatron using the setup instructions!"
            )
        global _USE_MEGATRON
        _USE_MEGATRON = True
        initialize_model_parallel(cfg.common.model_parallel_size)
        if torch.cuda.is_available():
            dist.all_reduce(torch.zeros(1).cuda(), group=get_model_parallel_group())
        model_parallel_cuda_manual_seed(cfg.common.seed)
        # This check should not be usually needed as we call init only once
        # but seems like tests are calling it multiple times.
        if _GLOBAL_MEMORY_BUFFER is None:
            _set_global_memory_buffer()
        model_part_number = get_model_parallel_rank()
        cfg.checkpoint.checkpoint_suffix += "-model_part-{0}".format(model_part_number)

    return cfg.distributed_training.distributed_rank


def distributed_main(i, main, cfg: MetaseqConfig, kwargs):
    # don't use MKL/OMP to avoid conflicting processes
    torch.set_num_threads(1)
    if not cfg.distributed_training.distributed_no_spawn:
        # if in local spawning, i is offset by -1 since torch.multiprocessing.spawn
        # always starts at rank 0
        i = i + 1
    cfg.distributed_training.device_id = i
    if torch.cuda.is_available() and not cfg.common.cpu:
        torch.cuda.set_device(cfg.distributed_training.device_id)
        # This is temporary way of making microsoft Tutel happy, as it reads the local rank from
        # the env. To make it work in cleaner way, we might need to change their interfaces to be
        # able to pass local rank.
        os.environ["LOCAL_RANK"] = str(cfg.distributed_training.device_id)
    if cfg.distributed_training.distributed_rank is None:
        # start_rank is the rank of gpu 0 on this machine.
        cfg.distributed_training.distributed_rank = kwargs.pop("start_rank", 0) + i

    cfg.distributed_training.distributed_rank = distributed_init(cfg)

    after_distributed_init_fn = kwargs.pop("after_distributed_init_fn", None)
    if after_distributed_init_fn:
        cfg = after_distributed_init_fn(cfg)
    retval = main(cfg, **kwargs)
    global_barrier()
    return retval


def global_barrier():
    """
    A global barrier that all workers in all process groups must wait for.
    """
    if torch.distributed.is_initialized():
        torch.distributed.barrier(get_global_group())


def _spawn_helper(main, cfg, kwargs):
    """
    Perform a fork() to many processes.

    Intentionally runs the rank0 in the main process so that signals
    can be more easily caught and we can cleanup processes.
    """
    # Launch multiple subprocesses
    spawncontext = torch.multiprocessing.start_processes(
        distributed_main,
        # need to give rank offset as 1 to cover the fact that the main
        # process is rank 0, but that spawn() doesn't let you control rank:
        # it always starts at 0
        (main, cfg, kwargs),
        nprocs=min(
            torch.cuda.device_count(),
            cfg.distributed_training.distributed_world_size - 1,
        ),
        join=False,
        start_method="spawn",
    )

    try:
        # -1 because we offset by +1 inside distributed_main when using
        # spawn_helper
        retval = distributed_main(-1, main, cfg, kwargs)
        spawncontext.join()
        return retval
    except (KeyboardInterrupt, Exception):
        # weirdly KeyboardInterrupt is not an Exception
        # propagate exceptions on the main node by killing workers
        for p in spawncontext.processes:
            if p.is_alive():
                os.kill(p.pid, signal.SIGTERM)
        raise


def call_main(cfg: MetaseqConfig, main, **kwargs):
    if cfg.distributed_training.distributed_init_method is None:
        infer_init_method(cfg.distributed_training)

    if cfg.distributed_training.distributed_init_method is not None:
        # distributed training
        if not cfg.distributed_training.distributed_no_spawn:
            start_rank = cfg.distributed_training.distributed_rank
            cfg.distributed_training.distributed_rank = None  # assign automatically
            kwargs["start_rank"] = start_rank
            return _spawn_helper(main, cfg, kwargs)
        else:
            return distributed_main(
                cfg.distributed_training.device_id, main, cfg, kwargs
            )
    else:
        # single GPU main
        return main(cfg, **kwargs)


def new_groups(grouped_ranks: List[List[int]]):
    groups = [dist.new_group(g) for g in grouped_ranks]
    my_group_idx = _find_my_group_index(grouped_ranks)
    return groups[my_group_idx]


def _find_my_group_index(grouped_ranks):
    my_rank = get_global_rank()
    for i, group in enumerate(grouped_ranks):
        if my_rank in group:
            return i
    raise RuntimeError


def _find_my_group(grouped_ranks):
    index = _find_my_group_index(grouped_ranks)
    return grouped_ranks[index]


def get_rank(group):
    return dist.get_rank(group=group)


def get_world_size(group):
    if torch.distributed.is_initialized():
        return dist.get_world_size(group=group)
    else:
        return 1


def get_global_group():
    if torch.distributed.is_initialized():
        if not hasattr(get_global_group, "_global_group"):
            # ideally we could use torch.distributed.group.WORLD, but it seems
            # to cause random NCCL hangs in some cases
            get_global_group._global_group = dist.new_group()
        return get_global_group._global_group
    else:
        return None


def get_global_rank():
    if torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    else:
        return 0


def get_global_world_size():
    if torch.distributed.is_initialized():
        return torch.distributed.get_world_size()
    else:
        return 1


def get_data_parallel_group():
    """Get the data parallel group the caller rank belongs to."""
    global _USE_MEGATRON
    if _USE_MEGATRON:
        from megatron import mpu

        return mpu.get_data_parallel_group()
    else:
        return get_global_group()


def get_data_parallel_rank():
    """Return my rank for the data parallel group."""
    dp_group = get_data_parallel_group()
    if dp_group is not None:
        return get_rank(dp_group)
    else:
        return get_global_rank()


def get_data_parallel_world_size():
    """Return world size for the data parallel group."""
    dp_group = get_data_parallel_group()
    if dp_group is not None:
        return get_world_size(dp_group)
    else:
        return get_global_world_size()


def get_model_parallel_group():
    global _USE_MEGATRON
    if _USE_MEGATRON:
        from megatron import mpu

        return mpu.get_tensor_model_parallel_group()
    else:
        return None


def get_model_parallel_rank():
    """Return my rank for the model parallel group."""
    return get_rank(get_model_parallel_group())


def get_model_parallel_world_size():
    """Return world size for the model parallel group."""
    return get_world_size(get_model_parallel_group())


def all_reduce(tensor, group, op="sum"):
    if op == "sum":
        op = dist.ReduceOp.SUM
    elif op == "max":
        op = dist.ReduceOp.MAX
    else:
        raise NotImplementedError
    dist.all_reduce(tensor, op=op, group=group)
    return tensor


def broadcast(tensor, src, group):
    dist.broadcast(tensor, src=src, group=group)


def all_to_all(tensor, group):
    """Perform an all-to-all operation on a 1D Tensor."""
    assert tensor.dim() == 1
    split_count = get_world_size(group=group)
    assert tensor.numel() % split_count == 0
    output = torch.zeros_like(tensor)
    dist.all_to_all_single(output, tensor, group=group)
    return output


def all_gather(tensor, group, return_tensor=False):
    """Perform an all-gather operation."""
    world_size = get_world_size(group=group)
    rank = get_rank(group=group)
    tensor_list = [
        tensor if i == rank else torch.empty_like(tensor) for i in range(world_size)
    ]
    dist.all_gather(tensor_list, tensor, group=group)
    if return_tensor:
        return torch.stack(tensor_list, dim=0)
    else:
        return tensor_list


def all_gather_list(data, group=None, max_size=16384):
    """Gathers arbitrary data from all nodes into a list.

    Similar to :func:`~torch.distributed.all_gather` but for arbitrary Python
    data. Note that *data* must be picklable and any CUDA tensors will be moved
    to CPU and returned on CPU as well.

    Args:
        data (Any): data from the local worker to be gathered on other workers
        group: group of the collective
        max_size (int, optional): maximum size of the data to be gathered
            across workers
    """
    from metaseq import utils

    if group is None:
        group = get_global_group()
    rank = get_rank(group=group)
    world_size = get_world_size(group=group)

    buffer_size = max_size * world_size
    if (
        not hasattr(all_gather_list, "_buffer")
        or all_gather_list._buffer.numel() < buffer_size
    ):
        all_gather_list._buffer = torch.cuda.ByteTensor(buffer_size)
        all_gather_list._cpu_buffer = torch.ByteTensor(max_size).pin_memory()
    buffer = all_gather_list._buffer
    buffer.zero_()
    cpu_buffer = all_gather_list._cpu_buffer

    data = utils.move_to_cpu(data)
    enc = pickle.dumps(data)
    enc_size = len(enc)
    header_size = 4  # size of header that contains the length of the encoded data
    size = header_size + enc_size
    if size > max_size:
        raise ValueError(
            "encoded data size ({}) exceeds max_size ({})".format(size, max_size)
        )

    header = struct.pack(">I", enc_size)
    cpu_buffer[:size] = torch.ByteTensor(list(header + enc))
    start = rank * max_size
    buffer[start : start + size].copy_(cpu_buffer[:size])

    all_reduce(buffer, group=group)

    buffer = buffer.cpu()
    try:
        result = []
        for i in range(world_size):
            out_buffer = buffer[i * max_size : (i + 1) * max_size]
            (enc_size,) = struct.unpack(">I", bytes(out_buffer[:header_size].tolist()))
            if enc_size > 0:
                result.append(
                    pickle.loads(
                        bytes(out_buffer[header_size : header_size + enc_size].tolist())
                    )
                )
        return result
    except pickle.UnpicklingError:
        raise Exception(
            "Unable to unpickle data from other workers. all_gather_list requires all "
            "workers to enter the function together, so this error usually indicates "
            "that the workers have fallen out of sync somehow. Workers can fall out of "
            "sync if one of them runs out of memory, or if there are other conditions "
            "in your training script that can cause one worker to finish an epoch "
            "while other workers are still iterating over their portions of the data. "
            "Try rerunning with --ddp-backend=legacy_ddp and see if that helps."
        )


def all_reduce_dict(data: Mapping[str, Any], device, group) -> Dict[str, Any]:
    """
    AllReduce a dictionary of values across workers. We separately
    reduce items that are already on the device and items on CPU for
    better performance.

    Args:
        data (Mapping[str, Any]): dictionary of data to all-reduce, but
            cannot be a nested dictionary
        device (torch.device): device for the reduction
        group: group of the collective
    """
    data_keys = list(data.keys())

    # We want to separately reduce items that are already on the
    # device and items on CPU for performance reasons.
    cpu_data = OrderedDict()
    device_data = OrderedDict()
    for k in data_keys:
        t = data[k]
        if not torch.is_tensor(t):
            cpu_data[k] = torch.tensor(t, dtype=torch.double)
        elif t.device.type != device.type:
            cpu_data[k] = t.to(dtype=torch.double)
        else:
            device_data[k] = t.to(dtype=torch.double)

    def _all_reduce_dict(data: OrderedDict):
        if len(data) == 0:
            return data
        buf = torch.cat([t.view(-1) for t in data.values()]).to(device=device)
        all_reduce(buf, group=group)
        split_buf = torch.split(buf, [t.numel() for t in data.values()])
        reduced_data = [t.view_as(orig) for t, orig in zip(split_buf, data.values())]
        return OrderedDict(zip(data.keys(), reduced_data))

    cpu_data = _all_reduce_dict(cpu_data)
    device_data = _all_reduce_dict(device_data)

    def get_from_stack(key):
        if key in cpu_data:
            return cpu_data[key]
        elif key in device_data:
            return device_data[key]
        raise KeyError

    return OrderedDict([(key, get_from_stack(key)) for key in data_keys])


def broadcast_tensors(
    tensors: Optional[List[torch.Tensor]],
    src_rank: int,
    group: object,
    dist_device: Optional[torch.device] = None,
) -> List[torch.Tensor]:
    """
    Broadcasts a list of tensors without other (non-src) ranks needing to know
    the dtypes/shapes of the tensors.
    """
    if dist_device is None:
        if torch.distributed.get_backend(group) == "nccl":
            dist_device = torch.device("cuda")
        else:
            dist_device = torch.device("cpu")

    # share metadata first to simplify transfer
    is_src_rank = get_rank(group) == src_rank
    if is_src_rank:
        # We only want to communicate device type ie (cpu vs cuda) and not the index of cuda.
        metadata = [
            {"size": t.size(), "dtype": t.dtype, "device": torch.device(t.device.type)}
            for t in tensors
        ]
        metadata = _broadcast_object_slow(metadata, src_rank, group, dist_device)
    else:
        metadata = _broadcast_object_slow(None, src_rank, group, dist_device)

    out_tensors = []
    for i, meta in enumerate(metadata):
        if is_src_rank:
            tensor = tensors[i]
            broadcast(tensors[i].to(dist_device), src=src_rank, group=group)
        else:
            tensor = torch.zeros(
                [meta["size"].numel()], dtype=meta["dtype"], device=dist_device
            )
            broadcast(tensor, src=src_rank, group=group)

        tensor = tensor.view(meta["size"]).to(meta["device"])
        out_tensors.append(tensor)
    return out_tensors


def broadcast_object(
    obj: Any,
    src_rank: int,
    group: object,
    dist_device: Optional[torch.device] = None,
) -> Any:
    """Broadcast an arbitrary Python object to other workers."""
    if dist_device is None:
        if torch.distributed.get_backend(group) == "nccl":
            dist_device = torch.device("cuda")
        else:
            dist_device = torch.device("cpu")

    if get_rank(group) == src_rank:
        # split the tensors from the non-tensors so we can broadcast them
        # directly, avoiding unnecessary serialization/deserialization
        tensors = []
        obj = _split_tensors_from_obj(obj, tensors)
        obj = _broadcast_object_slow(obj, src_rank, group, dist_device)
        tensors = broadcast_tensors(tensors, src_rank, group, dist_device)
    else:
        obj = _broadcast_object_slow(None, src_rank, group, dist_device)
        tensors = broadcast_tensors(None, src_rank, group, dist_device)
    return _put_tensors_in_obj(obj, tensors)


def _broadcast_object_slow(
    obj: Any,
    src_rank: int,
    group: object,
    dist_device: torch.device,
) -> Any:
    if get_rank(group) == src_rank:
        # Emit data
        buffer = io.BytesIO()
        torch.save(obj, buffer)
        buffer = torch.ByteTensor(buffer.getbuffer()).to(dist_device)
        length = torch.LongTensor([len(buffer)]).to(dist_device)
        broadcast(length, src=src_rank, group=group)
        broadcast(buffer, src=src_rank, group=group)
    else:
        # Fetch from the source
        length = torch.LongTensor([0]).to(dist_device)
        broadcast(length, src=src_rank, group=group)
        buffer = torch.ByteTensor(int(length.item())).to(dist_device)
        broadcast(buffer, src=src_rank, group=group)
        buffer = io.BytesIO(buffer.cpu().numpy())
        obj = torch.load(buffer, map_location="cpu")
    return obj


@dataclass(frozen=True)
class _TensorPlaceholder:
    index: int


def _split_tensors_from_obj(obj: Any, tensors: List[torch.Tensor]) -> Any:
    if torch.is_tensor(obj):
        placeholder = _TensorPlaceholder(index=len(tensors))
        tensors.append(obj)
        return placeholder
    elif isinstance(obj, dict):
        return {k: _split_tensors_from_obj(v, tensors) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_split_tensors_from_obj(v, tensors) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(_split_tensors_from_obj(v, tensors) for v in obj)
    elif isinstance(obj, set):
        return {_split_tensors_from_obj(v, tensors) for v in obj}
    else:
        return obj


def _put_tensors_in_obj(obj: Any, tensors: List[torch.Tensor]) -> Any:
    if isinstance(obj, _TensorPlaceholder):
        return tensors[obj.index]
    elif isinstance(obj, dict):
        return {k: _put_tensors_in_obj(v, tensors) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_put_tensors_in_obj(v, tensors) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(_put_tensors_in_obj(v, tensors) for v in obj)
    elif isinstance(obj, set):
        return {_put_tensors_in_obj(v, tensors) for v in obj}
    else:
        return obj
