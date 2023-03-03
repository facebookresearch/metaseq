# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Train a network across multiple GPUs.
"""

import contextlib
import functools
import logging
import math
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from itertools import chain
from typing import Any, Dict, List
import os
import torch
import torch.distributed as dist
from omegaconf import OmegaConf

from metaseq import checkpoint_utils, models, optim, utils
from metaseq.distributed import utils as distributed_utils, fsdp_enable_wrap, fsdp_wrap
from metaseq.file_io import PathManager
from metaseq.logging import meters, metrics
from metaseq.models.ema import build_ema
from metaseq.nan_detector import NanDetector
from metaseq.optim import lr_scheduler
from metaseq.utils import set_rank_seed

try:
    from megatron.mpu import (
        get_cuda_rng_tracker,
    )

    has_megatron_submodule = True
except (ImportError, ModuleNotFoundError):
    has_megatron_submodule = False

logger = logging.getLogger(__name__)


class Trainer(object):
    """Main class for data parallel training.

    This class supports synchronous distributed data parallel training,
    where multiple workers each have a full model replica and gradients
    are accumulated across workers before each update. We use
    :class:`~torch.nn.parallel.DistributedDataParallel` to handle
    communication of the gradients across workers.
    """

    def __init__(self, cfg, task, model, criterion):
        self.cfg = cfg
        self.task = task
        self.model_parallel_size = cfg.common.model_parallel_size

        if self.model_parallel_size > 1:
            if not has_megatron_submodule:
                raise ImportError(
                    "\n\nPlease install megatron using the setup instructions!"
                )

        # catalog shared parameters
        shared_params = _catalog_shared_params(model)
        self.cuda = torch.cuda.is_available() and not cfg.common.cpu

        self.quiet_logs = getattr(cfg.common, "quiet_logs", False)
        if self.cuda:
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        if self.is_fsdp:
            import fairscale

            if (
                max(self.cfg.optimization.update_freq) > 1
                and fairscale.__version__ < "0.4.0"
            ):
                raise RuntimeError(
                    "Please update to fairscale 0.4.0 or newer when combining "
                    "--update-freq with FullyShardedDataParallel"
                )
            if self.use_sharded_state:
                assert (
                    fairscale.__version__ >= "0.3.9"
                ), "--use-sharded-state requires newer fairscale. pip install -U fairscale"
        else:
            if self.cfg.distributed_training.cpu_offload:
                raise ValueError("--cpu-offload requires --ddp-backend=fully_sharded")

        # copy model and criterion to current device/dtype
        self._criterion = criterion
        self._model = model
        if not self.is_fsdp:
            if cfg.common.bf16:
                self._criterion = self._criterion.bfloat16()
                self._model = self._model.bfloat16()
            elif cfg.common.fp16:
                self._criterion = self._criterion.half()
                self._model = self._model.half()
        if (
            # the DistributedModel wrapper will handle moving to device,
            # so only handle cases which don't use the wrapper
            not self.use_distributed_wrapper
        ):
            self._criterion = self._criterion.to(device=self.device)
            self._model = self._model.to(device=self.device)

        # check that shared parameters are preserved after device transfer
        for shared_param in shared_params:
            ref = _get_module_by_path(self._model, shared_param[0])
            for path in shared_param[1:]:
                logger.info(
                    "detected shared parameter: {} <- {}".format(shared_param[0], path)
                )
                _set_module_by_path(self._model, path, ref)
        logger.info(metrics.get_nvidia_smi_gpu_memory_stats_str())

        self._dummy_batch = None  # indicates we don't have a dummy batch at first
        self._lr_scheduler = None
        self._num_updates = 0
        self._optim_history = None
        self._optimizer = None
        self._warn_once = set()
        self._wrapped_criterion = None
        self._wrapped_model = None
        self._ewm_loss = None
        self._skipped_loss_spikes = 0
        self._ema = None

        # TODO(myleott): support tpu
        if self.cuda and self.data_parallel_world_size > 1:
            self._grad_norm_buf = torch.cuda.DoubleTensor(self.data_parallel_world_size)
        else:
            self._grad_norm_buf = None

        # get detailed cuda environment
        if self.cuda:
            self.cuda_env = utils.CudaEnvironment()
            if self.data_parallel_world_size > 1:
                self.cuda_env_arr = distributed_utils.all_gather_list(
                    self.cuda_env, group=distributed_utils.get_global_group()
                )
            else:
                self.cuda_env_arr = [self.cuda_env]
            if self.data_parallel_rank == 0:
                utils.CudaEnvironment.pretty_print_cuda_env_list(self.cuda_env_arr)
        else:
            self.cuda_env = None
            self.cuda_env_arr = None

        metrics.log_start_time("wall", priority=790, round=0)

        self._start_time = time.time()
        self._previous_training_time = 0
        self._cumulative_training_time = None

    def reinitialize(self):
        """Reinitialize the Trainer, typically after model params change."""
        self._lr_scheduler = None
        self._optimizer = None
        self._wrapped_criterion = None
        self._wrapped_model = None

    @property
    def data_parallel_world_size(self):
        if self.cfg.distributed_training.distributed_world_size == 1:
            return 1
        return distributed_utils.get_data_parallel_world_size()

    @property
    def data_parallel_process_group(self):
        return distributed_utils.get_data_parallel_group()

    @property
    def data_parallel_rank(self):
        if self.cfg.distributed_training.distributed_world_size == 1:
            return 0
        return distributed_utils.get_data_parallel_rank()

    @property
    def is_data_parallel_master(self):
        # NOTE: this returns true for all model parallel replicas with data
        # parallel rank 0
        return self.data_parallel_rank == 0

    @property
    def use_distributed_wrapper(self) -> bool:
        return (self.data_parallel_world_size > 1) or (
            self.is_fsdp and self.cfg.distributed_training.cpu_offload
        )

    @property
    def should_save_checkpoint_on_current_rank(self) -> bool:
        """Indicates whether to save checkpoints on the current DDP rank."""
        if self.is_fsdp:
            return True
        else:
            return self.is_data_parallel_master

    @property
    def checkpoint_suffix(self) -> str:
        """Suffix to add to the checkpoint file name."""
        if not self.use_sharded_state:
            return self.cfg.checkpoint.checkpoint_suffix
        elif self.is_fsdp:
            return self.cfg.checkpoint.checkpoint_suffix + "-shard{0}".format(
                self.data_parallel_rank
            )
        else:
            return self.cfg.checkpoint.checkpoint_suffix or ""

    @property
    def criterion(self):
        if self._wrapped_criterion is None:
            if utils.has_parameters(self._criterion) and self.use_distributed_wrapper:
                self._wrapped_criterion = models.DistributedModel(
                    self.cfg.distributed_training,
                    self._criterion,
                    process_group=self.data_parallel_process_group,
                    device=self.device,
                )
            else:
                self._wrapped_criterion = self._criterion
        return self._wrapped_criterion

    @property
    def model(self):
        if self._wrapped_model is None:
            if self.use_distributed_wrapper or self.is_fsdp:
                self._wrapped_model = models.DistributedModel(
                    self.cfg.distributed_training,
                    self._model,
                    process_group=self.data_parallel_process_group,
                    device=self.device,
                )
            else:
                self._wrapped_model = self._model
        return self._wrapped_model

    @property
    def ema(self):
        if self._ema is None:
            self._build_ema()
        return self._ema

    def _build_ema(self):
        if self.cfg.ema.store_ema:
            if self.is_fsdp:
                # Build FSDP model
                extra = {
                    "is_moe": getattr(self.cfg.model, "moe_freq", 0) > 0,
                    "use_sharded_state": self.use_sharded_state,
                }
                with fsdp_enable_wrap(self.cfg.distributed_training, **extra):
                    model = fsdp_wrap(self.task.build_model(self.cfg.model))

                if self.cfg.common.memory_efficient_fp16:
                    if self.cfg.common.bf16:
                        model = model.bfloat16()
                    else:
                        model = model.half()

                # Copy FSDP model state (since copy.deepcopy doesn't work)
                state_dict = self.model.state_dict()
                if not self.use_sharded_state:
                    state_dict = distributed_utils.broadcast_object(
                        state_dict, src_rank=0, group=self.model.process_group
                    )
                model.load_state_dict(state_dict)
                self._ema = build_ema(model, self.cfg.ema, self.device)
            else:
                self._ema = build_ema(self._model, self.cfg.ema, self.device)
            logger.info("Exponential Moving Average Shadow Model is initialized.")

    @property
    def optimizer(self):
        if self._optimizer is None:
            self._build_optimizer()
        return self._optimizer

    @property
    def lr_scheduler(self):
        if self._lr_scheduler is None:
            self._build_optimizer()  # this will initialize self._lr_scheduler
        return self._lr_scheduler

    def _build_optimizer(self):
        params = list(
            filter(
                lambda p: p.requires_grad,
                chain(self.model.parameters(), self.criterion.parameters()),
            )
        )

        if self.is_fsdp and self.cfg.common.fp16:
            # FullyShardedDataParallel always uses MemoryEfficientFP16 wrapper,
            # mostly for the grad scaling. But if we don't have the
            # --memory-efficient-fp16 flag set, then we're effectively doing
            # regular --fp16 and can allow the use of optimizers that would
            # otherwise be unsupported by MemoryEfficientFP16Optimizer.
            allow_unsupported = not self.cfg.common.memory_efficient_fp16
            self._optimizer = optim.MemoryEfficientFP16Optimizer.build_optimizer(
                self.cfg, params, allow_unsupported=allow_unsupported
            )
        elif self.cfg.common.fp16:
            if self.cuda and torch.cuda.get_device_capability(0)[0] < 7:
                logger.info(
                    "NOTE: your device does NOT support faster training with --fp16, "
                    "please switch to FP32 which is likely to be faster"
                )
            if self.cfg.common.memory_efficient_fp16:
                self._optimizer = optim.MemoryEfficientFP16Optimizer.build_optimizer(
                    self.cfg, params
                )
            else:
                self._optimizer = optim.FP16Optimizer.build_optimizer(self.cfg, params)
        else:
            if self.cuda and torch.cuda.get_device_capability(0)[0] >= 7:
                logger.info("NOTE: your device may support faster training with --fp16")
            self._optimizer = optim.build_optimizer(self.cfg.optimizer, params)

        if self.is_fsdp:
            assert self._optimizer.supports_flat_params, (
                "--ddp-backend=fully_sharded is only compatible with pointwise "
                "optimizers (e.g., Adam, AdamW, Adadelta, Adamax, SGD, etc.). "
                "However, the sharding will result in slightly different results when "
                "using non-pointwise optimizers (e.g., Adagrad, Adafactor, LAMB)"
            )

        # We should initialize the learning rate scheduler immediately after
        # building the optimizer, so that the initial learning rate is set.
        self._lr_scheduler = lr_scheduler.build_lr_scheduler(
            self.cfg.lr_scheduler,
            self.optimizer,
        )
        self._lr_scheduler.step_update(0)

    @property
    def is_fsdp(self):
        return self.cfg.distributed_training.ddp_backend == "fully_sharded"

    @property
    def use_sharded_state(self):
        return self.cfg.distributed_training.use_sharded_state

    def consolidate_optimizer(self):
        """For OSS, we need to consolidate the state dict."""
        self._gathered_optim_state = None
        if hasattr(self.optimizer.optimizer, "consolidate_state_dict"):
            self.optimizer.optimizer.consolidate_state_dict()
        elif self.is_fsdp and not self.use_sharded_state:
            st = self.model.gather_full_optim_state_dict(
                self.optimizer
            )  # only returns on rank 0
            if st is None:
                st = -1  # sentinel so that workers do not save optimizer.state_dict()
            self._gathered_optim_state = st
            assert self._gathered_optim_state is not None

    def state_dict(self, filename, training_finished=False) -> Dict[str, Dict]:
        model_state_dict = self.model.state_dict()
        optim_state = self._gathered_optim_state or self.optimizer.state_dict()
        model_save_list = [
            (
                filename,
                model_state_dict,
                optim_state,
            )
        ]
        state_dicts = {}
        # This is what gets saved to checkpoints.
        for filename, model_state_dict, optimizer_state_dict in model_save_list:
            state_dict = {
                "cfg": OmegaConf.to_container(self.cfg)
                if OmegaConf.is_config(self.cfg)
                else self.cfg,
                "model": model_state_dict,
                "criterion": (
                    self.criterion.state_dict()
                    if utils.has_parameters(self.criterion)
                    else None
                ),
                "optimizer_history": (self._optim_history or [])
                + [
                    {
                        "criterion_name": self.get_criterion().__class__.__name__,
                        "optimizer_name": self.optimizer.__class__.__name__,
                        "lr_scheduler_state": self.lr_scheduler.state_dict(),
                        "num_updates": self.get_num_updates(),
                    }
                ],
                "task_state": self.task.state_dict() if self.task is not None else {},
                "extra_state": {
                    "metrics": metrics.state_dict(),
                    "previous_training_time": self.cumulative_training_time(),
                    "ewm_loss": self._ewm_loss,
                },
            }

            state_dict["last_optimizer_state"] = optimizer_state_dict

            if self.cfg.ema.store_ema:
                # Save EMA model state as extra state
                state_dict["extra_state"]["ema"] = self.ema.get_model().state_dict()
                if self.cfg.ema.ema_fp32:
                    # Save EMA params in fp32
                    state_dict["extra_state"]["ema_fp32_params"] = self.ema.fp32_params

            if self.is_fsdp and self.use_sharded_state:
                state_dict[
                    "shard_metadata"
                ] = (
                    self.model.local_metadata_dict()
                )  # save FSDP flattening and padding info
            state_dicts[filename] = state_dict
        return state_dicts

    def save_checkpoint(
        self, filename, extra_state, training_finished=False, async_callback_fn=None
    ):
        """Save all training state in a checkpoint file."""

        if self.model_parallel_size > 1:
            extra_state["rng_tracker_states"] = get_cuda_rng_tracker().get_states()

        # call state_dict on all ranks in case it needs internal communication
        state_dicts = self.state_dict(filename, training_finished)
        for filename, state_dict in state_dicts.items():
            logger.info(f"Saving checkpoint to {filename}")
            state_dict = utils.move_to_cpu(
                state_dict,
                # keep params in FP16 when training with --memory-efficient-fp16
                cast_to_fp32=not self.cfg.common.memory_efficient_fp16,
            )
            state_dict["extra_state"].update(extra_state)
            if self.should_save_checkpoint_on_current_rank:
                if not hasattr(self, "async_checkpoint"):
                    self.async_checkpoint = ThreadPoolExecutor(max_workers=1)

                def perform_save():
                    try:
                        logger.info(f"Beginning asynchronous torch.save to {filename}")
                        async_callback_fn(filename)
                        logger.info(f"Asynchronous torch.save to {filename} complete.")
                    except Exception as e:
                        logger.exception(f"Asynchronous save failed: {e}")

                torch.save(state_dict, filename)
                if async_callback_fn is not None:
                    self.async_checkpoint.submit(perform_save)
            logger.info(f"Finished saving checkpoint to {filename}")

    def load_checkpoint(
        self,
        filename,
        reset_optimizer=False,
        reset_lr_scheduler=False,
        optimizer_overrides=None,
        reset_meters=False,
    ):
        """
        Load all training state from a checkpoint file.
        rank = 0 will load the checkpoint, and then broadcast it to all
        other ranks.
        """
        extra_state, self._optim_history, last_optim_state = None, [], None

        is_distributed = self.data_parallel_world_size > 1
        bexists = False

        logger.info(f"attempting to load checkpoint from: {filename}")

        if PathManager.isfile(filename):
            bexists = True
        else:
            # this is a big hacky as when we increase the world size, then filename doesn't really point
            # to a real file, we convert it to multiple files to be loaded later.
            # so here we just check if there are some files existing in the dir.
            files_in_local_dir = os.listdir(os.path.dirname(filename))
            filename_prefix = os.path.splitext(os.path.basename(filename))[0].replace(
                self.checkpoint_suffix, ""
            )
            matched_files = [
                f for f in files_in_local_dir if f.startswith(filename_prefix)
            ]
            bexists = len(matched_files) > 0

        if bexists:
            logger.info(f"Preparing to load checkpoint {filename}")
            # FSDP requires loading checkpoint shards on all ranks
            load_on_all_ranks = self.is_fsdp

            if load_on_all_ranks or self.is_data_parallel_master:
                state = checkpoint_utils.load_checkpoint_to_cpu(
                    filename,
                )
                last_optim_state = state.get("last_optimizer_state", None)
                if last_optim_state == -1:
                    master_path = re.sub("shard[0-9]+", "shard0", filename)
                    last_optim_state = torch.load(master_path, map_location="cpu")[
                        "last_optimizer_state"
                    ]

                logger.info(f"Loaded state for {filename}")

            else:
                last_optim_state = None
                state = None

            if self.data_parallel_world_size > 1 and not load_on_all_ranks:
                state = distributed_utils.broadcast_object(
                    state,
                    src_rank=0,
                    group=self.data_parallel_process_group,
                    dist_device=self.device,
                )
                if self.data_parallel_rank > 0:
                    last_optim_state = state.get("last_optimizer_state", None)

            # load model parameters
            try:
                self.model.load_state_dict(state["model"], strict=True)
                # save memory for later steps
                del state["model"]
                if utils.has_parameters(self.get_criterion()):
                    self.get_criterion().load_state_dict(
                        state["criterion"], strict=True
                    )
                    del state["criterion"]

            except Exception:
                raise Exception(
                    "Cannot load model parameters from checkpoint {}; "
                    "please ensure that the architectures match.".format(filename)
                )
            extra_state = state["extra_state"]
            self._optim_history = state["optimizer_history"]

        if last_optim_state is not None and not reset_optimizer:
            # rebuild optimizer after loading model, since params may have changed
            self._build_optimizer()

            # only reload optimizer and lr_scheduler if they match
            last_optim = self._optim_history[-1]
            assert (
                last_optim["criterion_name"] == self.get_criterion().__class__.__name__
            ), (
                f"Criterion does not match; please reset the optimizer "
                f"(--reset-optimizer). {last_optim['criterion_name']} vs "
                f"{self.get_criterion().__class__.__name__}"
            )
            assert last_optim["optimizer_name"] == self.optimizer.__class__.__name__, (
                f"Optimizer does not match; please reset the optimizer "
                f"(--reset-optimizer). {last_optim['optimizer_name']} vs "
                f"{self.optimizer.__class__.__name__}"
            )

            if not reset_lr_scheduler:
                self.lr_scheduler.load_state_dict(last_optim["lr_scheduler_state"])

            if not load_on_all_ranks and is_distributed:
                last_optim_state = self.optimizer.broadcast_global_state_dict(
                    last_optim_state
                )
            elif self.is_fsdp and not self.use_sharded_state:
                last_optim_state = self.model.get_shard_from_optim_state_dict(
                    last_optim_state
                )
                logger.info(f"FSDP got shard from optim_state for {filename}")

            self.optimizer.load_state_dict(last_optim_state, optimizer_overrides)
            logger.info(f"Loaded optim_state for {filename}")
            self.set_num_updates(last_optim["num_updates"])

        if extra_state is not None:
            itr_state = extra_state["train_iterator"]
            epoch = itr_state["epoch"]

            if "previous_training_time" in extra_state:
                self._previous_training_time = extra_state["previous_training_time"]
                self._start_time = time.time()

            if "ewm_loss" in extra_state:
                self._ewm_loss = extra_state["ewm_loss"]

            self.lr_step(epoch)

            if (
                itr_state.get("version", 1) >= 2
                and itr_state["iterations_in_epoch"] == 0
            ):
                # reset meters at start of epoch
                reset_meters = True

            if "metrics" in extra_state and not reset_meters:
                metrics.load_state_dict(extra_state["metrics"])

                # reset TimeMeters, since their start times don't make sense anymore
                for meter in metrics.get_meters("default"):
                    if isinstance(meter, meters.TimeMeter):
                        meter.reset()

            if self.cfg.ema.store_ema:
                if "ema" not in extra_state:
                    logger.warn(
                        "EMA not found in checkpoint. But store_ema is True. "
                        "EMA is re-initialized from checkpoint."
                    )
                    self.ema.restore(
                        state["model"], build_fp32_params=self.cfg.ema.ema_fp32
                    )
                else:
                    logger.info("Loading EMA from checkpoint")
                    self.ema.restore(extra_state["ema"], build_fp32_params=False)

                    if self.cfg.ema.ema_fp32:
                        if "ema_fp32_params" in extra_state:
                            logger.info("Loading EMA fp32 params from checkpoint")
                            self.ema.build_fp32_params(extra_state["ema_fp32_params"])
                        else:
                            logger.info(
                                "Building EMA fp32 params from EMA model in checkpoint"
                            )
                            self.ema.build_fp32_params()

            logger.info(
                f"Loaded checkpoint {filename} (epoch {epoch} @ {self.get_num_updates()} updates)"
            )
        else:
            logger.info("No existing checkpoint found {}".format(filename))

        if extra_state is not None and "rng_tracker_states" in extra_state:
            get_cuda_rng_tracker().set_states(extra_state["rng_tracker_states"])
        return extra_state

    def get_train_iterator(
        self,
        epoch,
        combine=True,
        data_selector=None,
        shard_batch_itr=True,
        disable_iterator_cache=False,
    ):
        """Return an EpochBatchIterator over the training set for a given epoch."""
        logger.info("loading train data for epoch {}".format(epoch))
        self.task.load_dataset(
            self.cfg.dataset.train_subset,
            epoch=epoch,
            combine=combine,
            data_selector=data_selector,
        )
        batch_iterator = self.task.get_batch_iterator(
            dataset=self.task.dataset(self.cfg.dataset.train_subset),
            max_tokens=self.cfg.dataset.max_tokens,
            max_sentences=self.cfg.dataset.batch_size,
            max_positions=utils.resolve_max_positions(
                self.task.max_positions(),
                self.model.max_positions(),
                self.cfg.dataset.max_tokens,
            ),
            ignore_invalid_inputs=True,
            required_batch_size_multiple=self.cfg.dataset.required_batch_size_multiple,
            seed=self.cfg.common.seed,
            num_shards=self.data_parallel_world_size if shard_batch_itr else 1,
            shard_id=self.data_parallel_rank if shard_batch_itr else 0,
            num_workers=self.cfg.dataset.num_workers,
            epoch=epoch,
            data_buffer_size=self.cfg.dataset.data_buffer_size,
            disable_iterator_cache=disable_iterator_cache,
            skip_remainder_batch=True,
        )
        logger.info("finished creating batch iterator")
        self.reset_dummy_batch(batch_iterator.first_batch)
        return batch_iterator

    def get_valid_iterator(
        self,
        subset,
        disable_iterator_cache=False,
    ):
        """Return an EpochBatchIterator over given validation subset for a given epoch."""
        batch_iterator = self.task.get_batch_iterator(
            dataset=self.task.dataset(subset),
            max_tokens=self.cfg.dataset.max_tokens_valid,
            max_sentences=self.cfg.dataset.batch_size_valid,
            max_positions=utils.resolve_max_positions(
                self.task.max_positions(),
                self.model.max_positions(),
            ),
            ignore_invalid_inputs=self.cfg.dataset.skip_invalid_size_inputs_valid_test,
            required_batch_size_multiple=self.cfg.dataset.required_batch_size_multiple,
            seed=self.cfg.common.seed,
            num_shards=self.data_parallel_world_size,
            shard_id=self.data_parallel_rank,
            num_workers=self.cfg.dataset.num_workers_valid,
            # always pass a fixed "epoch" to keep validation data consistent
            # across training epochs
            epoch=1,
            data_buffer_size=self.cfg.dataset.data_buffer_size,
            disable_iterator_cache=disable_iterator_cache,
            skip_remainder_batch=False,
        )
        self.reset_dummy_batch(batch_iterator.first_batch)
        return batch_iterator

    def begin_epoch(self, epoch):
        """Called at the beginning of each epoch."""
        logger.info("begin training epoch {}".format(epoch))

        self.lr_step_begin_epoch(epoch)

        # task specific setup per epoch
        self.task.begin_epoch(epoch, self.get_model())

    def begin_valid_epoch(self, epoch):
        """Called at the beginning of each validation epoch."""

        # task specific setup per validation epoch
        self.task.begin_valid_epoch(epoch, self.get_model())

    def reset_dummy_batch(self, batch):
        self._dummy_batch = batch

    @metrics.aggregate("train")
    def train_step(self, samples):
        """Do forward, backward and parameter update."""
        self._set_seed()
        self.model.train()
        self.criterion.train()
        self.zero_grad()

        metrics.log_start_time("train_wall", priority=800, round=0)

        # If EMA is enabled through store_ema=True
        # and task.uses_ema is True, pass the EMA model as a keyword
        # argument to the task.
        extra_kwargs = {}
        if self.cfg.ema.store_ema and getattr(self.task, "uses_ema", False):
            extra_kwargs["ema_model"] = self.ema.get_model()

        # forward and backward pass
        logging_outputs, sample_size = [], 0
        for i, sample in enumerate(samples):  # delayed update loop
            if (
                self.get_num_updates() == 0
                and i == 0
                and distributed_utils.get_global_rank() == 0
            ):
                logger.info(f"First batch on first rank: " + str(sample))
            sample, is_dummy_batch = self._prepare_sample(sample)

            def maybe_no_sync():
                """
                Whenever *samples* contains more than one mini-batch, we
                want to accumulate gradients locally and only call
                all-reduce in the last backwards pass.
                """
                if (
                    self.data_parallel_world_size > 1
                    and hasattr(self.model, "no_sync")
                    and i < len(samples) - 1
                    # The no_sync context manager results in increased memory
                    # usage with FSDP, since full-size gradients will be
                    # accumulated on each GPU. It's typically a better tradeoff
                    # to do the extra communication with FSDP.
                    and not self.is_fsdp
                ):
                    return self.model.no_sync()
                else:
                    return contextlib.ExitStack()  # dummy contextmanager

            try:
                with maybe_no_sync(), (
                    set_rank_seed(self.cfg.common.seed, self.get_num_updates())
                    if self.cfg.common.seed_per_rank
                    else contextlib.nullcontext()
                ):
                    # forward and backward
                    loss, sample_size_i, logging_output = self.task.train_step(
                        sample=sample,
                        model=self.model,
                        criterion=self.criterion,
                        optimizer=self.optimizer,
                        update_num=self.get_num_updates(),
                        ignore_grad=is_dummy_batch,
                    )
                    del loss

                logging_outputs.append(logging_output)
                sample_size += sample_size_i

                # emptying the CUDA cache after the first step can
                # reduce the chance of OOM
                if self.cuda and self.get_num_updates() == 0:
                    torch.cuda.empty_cache()
            except RuntimeError as e:
                if "out of memory" in str(e):
                    self._log_oom(e)
                raise e

        if is_dummy_batch:
            if torch.is_tensor(sample_size):
                sample_size.zero_()
            else:
                sample_size *= 0.0

        if torch.is_tensor(sample_size):
            sample_size = sample_size.float()
        else:
            sample_size = float(sample_size)

        # gather logging outputs from all replicas
        if self._sync_stats():
            train_time = self._local_cumulative_training_time()
            logging_outputs, (
                sample_size,
                total_train_time,
            ) = self._aggregate_logging_outputs(
                logging_outputs, sample_size, train_time, ignore=is_dummy_batch
            )
            self._cumulative_training_time = (
                total_train_time / self.data_parallel_world_size
            )

        overflow = False
        logger.debug(f"[{self.get_num_updates()}] done with fwd, bwd")
        try:
            # reduce gradients across workers
            self.optimizer.all_reduce_grads(self.model)
            if utils.has_parameters(self.criterion):
                self.optimizer.all_reduce_grads(self.criterion)

            # multiply gradients by (data_parallel_size / sample_size) since
            # DDP normalizes by the number of data parallel workers for
            # improved fp16 precision.
            # Thus we get (sum_of_gradients / sample_size) at the end.
            # In case of fp16, this step also undoes loss scaling.
            # (Debugging note: Some optimizers perform this scaling on the
            # fly, so inspecting model.parameters() or optimizer.params may
            # still show the original, unscaled gradients.)
            numer = self.data_parallel_world_size if self._sync_stats() else 1
            self.optimizer.multiply_grads(numer / (sample_size or 1.0))
            # Note: (sample_size or 1.0) handles the case of a zero gradient, in a
            # way that avoids CPU/device transfers in case sample_size is a GPU or
            # TPU object. The assumption is that the gradient itself is also 0.

            # clip grads
            grad_norm = self.clip_grad_norm(
                self.cfg.optimization.clip_norm,
                self.cfg.optimization.clip_norm_type,
                self.cfg.optimization.skip_gradient_update_on_clip_norm,
            )
            # check that grad norms are consistent across workers
            self._check_grad_norms(grad_norm)
            if not torch.isfinite(grad_norm).all():
                # check local gradnorm single GPU case, trigger NanDetector
                raise FloatingPointError("gradients are Nan/Inf")
            # skip optimizer step if there is a loss spike
            ewm_loss_ratio = self.skip_spike(
                logging_outputs, self.cfg.optimization.ewm_ratio_to_skip_batch
            )
            # downscale grads by ewm_loss_ratio ** 4
            if ewm_loss_ratio > 1.0:
                grad_mult_factor = 1.0 / (ewm_loss_ratio**4)
                curr_lr = self.optimizer.get_lr()
                new_lr = curr_lr * grad_mult_factor
                self.optimizer.set_lr(new_lr)
                logger.info(f"Scaling LR by {grad_mult_factor:.2f} to {new_lr:.6f}")
            # take an optimization step
            self.task.optimizer_step(
                self.optimizer,
                model=self.model,
                update_num=self.get_num_updates(),
            )
            logger.debug(f"[{self.get_num_updates()}] done with optimizer step")

        except FloatingPointError:
            # re-run the forward and backward pass with hooks attached to print
            # out where it fails
            self.zero_grad()
            with NanDetector(self.get_model()), (
                set_rank_seed(self.cfg.common.seed, self.get_num_updates())
                if self.cfg.common.seed_per_rank
                else contextlib.nullcontext()
            ):
                for _, sample in enumerate(samples):
                    sample, _ = self._prepare_sample(sample)
                    self.task.train_step(
                        sample,
                        self.model,
                        self.criterion,
                        self.optimizer,
                        self.get_num_updates(),
                        ignore_grad=False,
                    )
            raise
        except OverflowError as e:
            overflow = True
            logger.info(
                f"NOTE: gradient overflow detected, ignoring gradient, {str(e)}"
            )
            grad_norm = torch.tensor(0.0).cuda()
            self.zero_grad()
        # except SpikeError as e:
        # overflow = True
        # logger.info(str(e))
        # self.zero_grad()
        except RuntimeError as e:
            if "out of memory" in str(e):
                self._log_oom(e)
                logger.error("OOM during optimization, irrecoverable")
            raise e

        logging_output = None
        if not overflow:
            self.set_num_updates(self.get_num_updates() + 1)

            # EMA update
            if self.cfg.ema.store_ema:
                # Step EMA forward with new model.
                self.ema.step(
                    self.get_model(),
                    self.get_num_updates(),
                )
                metrics.log_scalar(
                    "ema_decay",
                    self.ema.get_decay(),
                    priority=10000,
                    round=5,
                    weight=0,
                )

            if self.cuda and self.cuda_env is not None:
                # log minimum free memory over the iteration
                self._log_gpu_mem_stats()

            # log stats
            logging_output = self._reduce_and_log_stats(
                logging_outputs, sample_size, grad_norm, ewm_loss_ratio
            )

            # clear CUDA cache to reduce memory fragmentation
            if (
                self.cuda
                and self.cfg.common.empty_cache_freq > 0
                and (
                    (self.get_num_updates() + self.cfg.common.empty_cache_freq - 1)
                    % self.cfg.common.empty_cache_freq
                )
                == 0
            ):
                torch.cuda.empty_cache()

        if self.cfg.common.fp16 and not self.cfg.common.bf16:
            metrics.log_scalar(
                "loss_scale",
                self.optimizer.scaler.loss_scale,
                priority=700,
                round=4,
                weight=0,
            )
            metrics.log_scalar(
                "scale_window",
                self.optimizer.scaler.scale_window,
                priority=700,
                round=4,
                weight=0,
            )

        metrics.log_stop_time("train_wall")
        return logging_output

    @metrics.aggregate("valid")
    def valid_step(self, sample, num_step=0, raise_oom=False):
        """Do forward pass in evaluation mode."""

        # If EMA is enabled through store_ema=True
        # and task.uses_ema is True, pass the EMA model as a keyword
        # argument to the task.
        extra_kwargs = {}
        if self.cfg.ema.store_ema and getattr(self.task, "uses_ema", False):
            extra_kwargs["ema_model"] = self.ema.get_model()

        with torch.no_grad():
            self.model.eval()
            self.criterion.eval()
            self.zero_grad()

            sample, is_dummy_batch = self._prepare_sample(sample)

            try:
                with set_rank_seed(
                    self.cfg.common.seed, num_step + self.get_num_updates()
                ) if self.cfg.common.seed_per_rank else contextlib.nullcontext():
                    _loss, sample_size, logging_output = self.task.valid_step(
                        sample, self.model, self.criterion
                    )
            except RuntimeError as e:
                if "out of memory" in str(e):
                    self._log_oom(e)
                    if not raise_oom:
                        logger.warning(
                            "ran out of memory in validation step, retrying batch"
                        )
                        for p in self.model.parameters():
                            if p.grad is not None:
                                p.grad = None  # free some memory
                        if self.cuda:
                            torch.cuda.empty_cache()
                        return self.valid_step(sample, num_step, raise_oom=True)
                raise e

            logging_outputs = [logging_output]
            if is_dummy_batch:
                if torch.is_tensor(sample_size):
                    sample_size.zero_()
                else:
                    sample_size *= 0.0

        # gather logging outputs from all replicas
        if self.data_parallel_world_size > 1:
            logging_outputs, (sample_size,) = self._aggregate_logging_outputs(
                logging_outputs,
                sample_size,
                ignore=is_dummy_batch,
            )

        # log validation stats
        logging_output = self._reduce_and_log_stats(logging_outputs, sample_size)
        return logging_output

    def zero_grad(self):
        self.optimizer.zero_grad()

    def lr_step_begin_epoch(self, epoch):
        """Adjust the learning rate at the beginning of the epoch."""
        self.lr_scheduler.step_begin_epoch(epoch)
        # prefer updating the LR based on the number of steps
        return self.lr_step_update()

    def lr_step(self, epoch, val_loss=None):
        """Adjust the learning rate at the end of the epoch."""
        self.lr_scheduler.step(epoch, val_loss)
        # prefer updating the LR based on the number of steps
        return self.lr_step_update()

    def lr_step_update(self):
        """Update the learning rate after each update."""
        new_lr = self.lr_scheduler.step_update(self.get_num_updates())
        if isinstance(new_lr, dict):
            for k, v in new_lr.items():
                metrics.log_scalar(f"lr_{k}", v, weight=0, priority=300)
            new_lr = new_lr.get("default", next(iter(new_lr.values())))
        else:
            metrics.log_scalar("lr", new_lr, weight=0, priority=300)
        return new_lr

    def get_lr(self):
        """Get the current learning rate."""
        return self.optimizer.get_lr()

    def get_model(self):
        """Get the (non-wrapped) model instance."""
        return self._model

    def get_criterion(self):
        """Get the (non-wrapped) criterion instance."""
        return self._criterion

    def get_num_updates(self):
        """Get the number of parameters updates."""
        return self._num_updates

    def set_num_updates(self, num_updates):
        """Set the number of parameters updates."""
        self._num_updates = num_updates
        self.lr_step_update()
        metrics.log_scalar("num_updates", self._num_updates, weight=0, priority=200)

    def clip_grad_norm(
        self, clip_norm, clip_norm_type="l2", skip_gradient_update_on_clip_norm=False
    ):
        if self.model_parallel_size == 1:
            return self.optimizer.clip_grad_norm(
                clip_norm,
                clip_norm_type,
                aggregate_norm_fn=None,
                skip_gradient_update_on_clip_norm=skip_gradient_update_on_clip_norm,
            )
        else:

            def _aggregate_model_parallel_grad_norm(norm_type, total_norm):
                norm_type2_reduce_op = {
                    "l2": dist.ReduceOp.SUM,
                    "inf": dist.ReduceOp.MAX,
                }
                reduce_op = norm_type2_reduce_op[norm_type]
                if norm_type == "l2":
                    total_norm.pow_(2)
                dist.all_reduce(
                    total_norm,
                    group=distributed_utils.get_model_parallel_group(),
                    op=reduce_op,
                )
                if norm_type == "l2":
                    total_norm.sqrt_()
                return total_norm

            return self.optimizer.clip_grad_norm(
                clip_norm,
                clip_norm_type,
                aggregate_norm_fn=functools.partial(
                    _aggregate_model_parallel_grad_norm, clip_norm_type
                ),
                skip_gradient_update_on_clip_norm=skip_gradient_update_on_clip_norm,
            )

    def skip_spike(self, logging_outputs, ewm_ratio_to_skip_batch, span=9):
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        loss_t = float(loss_sum / sample_size / math.log(2))

        if self._ewm_loss is None:
            self._ewm_loss = loss_t

        ewm_t_1 = self._ewm_loss
        ewm_ratio = loss_t / ewm_t_1

        if ewm_ratio > ewm_ratio_to_skip_batch:
            self._skipped_loss_spikes += 1
            # raise SpikeError(
            #     f"Skip batch as we encountered a loss spike. In "
            #     f"num_update: {self.get_num_updates()} the loss is {loss_t:.2f}. "
            #     f"The ewm for the loss was only at {ewm_t:.2f} . "
            #     f"The loss to ewm loss ratio is {ewm_ratio:.2f}, which is higher than "
            #     f"ewm_ratio_to_skip_batch of {ewm_ratio_to_skip_batch} ."
            # )
        else:
            # update the moving average only if we are not loss spiking
            alpha = 2 / (span + 1)
            ewm_t = (1 - alpha) * ewm_t_1 + alpha * loss_t
            self._ewm_loss = ewm_t

        return ewm_ratio

    def cumulative_training_time(self):
        if self._cumulative_training_time is None:
            # single GPU
            return self._local_cumulative_training_time()
        else:
            return self._cumulative_training_time

    def _local_cumulative_training_time(self):
        """Aggregate training time in seconds."""
        return time.time() - self._start_time + self._previous_training_time

    def _prepare_sample(self, sample, is_dummy=False):
        if sample == "DUMMY":
            raise Exception(
                "Trying to use an uninitialized 'dummy' batch. This usually indicates "
                "that the total number of batches is smaller than the number of "
                "participating GPUs. Try reducing the batch size or using fewer GPUs."
            )

        if sample is None or len(sample) == 0:
            assert (
                self._dummy_batch is not None and len(self._dummy_batch) > 0
            ), "Invalid dummy batch: {}".format(self._dummy_batch)
            sample, _ = self._prepare_sample(self._dummy_batch, is_dummy=True)
            return sample, True

        if self.cuda:
            sample = utils.move_to_cuda(sample)

            # if False:  # turn on to double-check we do not have data loader issues
            #     # When we finish an epoch some dataloaders run short on data one iteration before others.
            #     # We want to check that the data loaders that are running short are returning correct data
            #     # on all their previous iterations.
            #
            #     # If they are returning the correct data, then we can rule out a lot of reasons why they would
            #     # run short.
            #
            #     ipt = sample["net_input"]["src_tokens"]
            #     if not hasattr(self, "input_errors"):
            #         self.input_errors = torch.tensor(
            #             0, dtype=torch.int, device=ipt.device
            #         )
            #
            #     min_ipt = ipt.clone()
            #
            #     torch.distributed.all_reduce(
            #         min_ipt,
            #         op=torch.distributed.ReduceOp.MIN,
            #         group=distributed_utils.get_model_parallel_group(),
            #     )
            #
            #     self.input_errors += (min_ipt != ipt).any()
            #
            #     if self.get_num_updates() % self.cfg.common.log_interval == 0:
            #         if int(self.input_errors) > 0:
            #             logger.error(
            #                 f"Data {self.data_parallel_rank} Model {distributed_utils.get_model_parallel_rank()} "
            #                 f"has {self.input_errors} data mismatch errors!"
            #             )

        def lower_precision(t):
            """Converts a tensor to the desired dtype based on our cfg."""
            if t.dtype is torch.float32:
                if self.cfg.common.bf16 or self.cfg.bf16:
                    return t.bfloat16()
                return t.half()
            return t

        if self.cfg.common.fp16:
            sample = utils.apply_to_sample(lower_precision, sample)

        if self._dummy_batch == "DUMMY":
            self._dummy_batch = sample

        return sample, False

    def _set_seed(self):
        # Set seed based on args.seed and the update number so that we get
        # reproducible results when resuming from checkpoints
        seed = self.cfg.common.seed + self.get_num_updates()
        utils.set_torch_seed(seed)

    def _sync_stats(self):
        # Return True if it's using multiple GPUs and DDP
        if self.data_parallel_world_size == 1:
            return False
        else:
            return True

    def _log_oom(self, exc):
        msg = "OOM: Ran out of memory with exception: {}".format(exc)
        logger.warning(msg)
        if torch.cuda.is_available() and hasattr(torch.cuda, "memory_summary"):
            for device_idx in range(torch.cuda.device_count()):
                logger.warning(torch.cuda.memory_summary(device=device_idx))
        sys.stderr.flush()

    def _aggregate_logging_outputs(
        self,
        logging_outputs: List[Dict[str, Any]],
        *extra_stats_to_sum,
        ignore=False,
    ):
        if self.task.__class__.logging_outputs_can_be_summed(self.get_criterion()):
            return self._fast_stat_sync_sum(
                logging_outputs, *extra_stats_to_sum, ignore=ignore
            )
        else:
            return self._all_gather_list_sync(
                logging_outputs, *extra_stats_to_sum, ignore=ignore
            )

    def _all_gather_list_sync(
        self,
        logging_outputs: List[Dict[str, Any]],
        *extra_stats_to_sum,
        ignore=False,
    ):
        """
        Sync logging outputs across workers. all_gather_list_sync is
        suitable when logging outputs are complex types.
        """
        if ignore:
            logging_outputs = []
        results = list(
            zip(
                *distributed_utils.all_gather_list(
                    [logging_outputs] + list(extra_stats_to_sum),
                    max_size=getattr(self.cfg.common, "all_gather_list_size", 16384),
                    group=self.data_parallel_process_group,
                )
            )
        )
        logging_outputs, extra_stats_to_sum = results[0], results[1:]
        logging_outputs = list(chain.from_iterable(logging_outputs))
        extra_stats_to_sum = [sum(s) for s in extra_stats_to_sum]
        return logging_outputs, extra_stats_to_sum

    def _fast_stat_sync_sum(
        self,
        logging_outputs: List[Dict[str, Any]],
        *extra_stats_to_sum,
        ignore=False,
    ):
        """
        Sync logging outputs across workers. fast_stat_sync_sum is
        faster than all_gather_list_sync, but is only suitable when
        logging outputs are scalars and can be summed. Note that
        *logging_outputs* cannot contain any nested dicts/lists.
        """
        data = {}
        for i, stat in enumerate(extra_stats_to_sum):
            data["extra_stats_" + str(i)] = stat
        if len(logging_outputs) > 0:
            log_keys = list(logging_outputs[0].keys())
            for k in log_keys:
                if not ignore:
                    v = sum(log[k] for log in logging_outputs if k in log)
                else:
                    v = logging_outputs[0][k]
                    v = torch.zeros_like(v) if torch.is_tensor(v) else 0
                data["logging_outputs_" + k] = v
        else:
            log_keys = None

        data = distributed_utils.all_reduce_dict(
            data, device=self.device, group=self.data_parallel_process_group
        )

        extra_stats_to_sum = [
            data["extra_stats_" + str(i)] for i in range(len(extra_stats_to_sum))
        ]
        if log_keys is not None:
            logging_outputs = [{k: data["logging_outputs_" + k] for k in log_keys}]
        else:
            logging_outputs = []
        return logging_outputs, extra_stats_to_sum

    def _check_grad_norms(self, grad_norm):
        """Check that grad norms are consistent across workers."""
        if self._grad_norm_buf is not None:
            self._grad_norm_buf.zero_()
            self._grad_norm_buf[self.data_parallel_rank] = grad_norm
            distributed_utils.all_reduce(
                self._grad_norm_buf, group=self.data_parallel_process_group
            )

            def is_consistent(tensor):
                max_abs_diff = torch.max(torch.abs(tensor - tensor[0]))
                return (
                    torch.isfinite(tensor).all()
                    and (max_abs_diff / (tensor[0] + 1e-6) < 1e-6).all()
                )

            if not is_consistent(self._grad_norm_buf):
                pretty_detail = "\n".join(
                    "rank {:3d} = {:.8f}".format(r, n)
                    for r, n in enumerate(self._grad_norm_buf.tolist())
                )
                error_detail = "grad_norm across the workers:\n{}\n".format(
                    pretty_detail
                )
                # use FloatingPointError to trigger NanDetector
                raise FloatingPointError(
                    "Fatal error: gradients are inconsistent between workers. "
                    "Try --ddp-backend=legacy_ddp. "
                    "Or are you mixing up different generation of GPUs in training?"
                    + "\n"
                    + "-" * 80
                    + "\n{}\n".format(error_detail)
                    + "-" * 80
                )

    def _reduce_and_log_stats(
        self, logging_outputs, sample_size, grad_norm=None, ewm_loss_ratio=0
    ):
        # perform a bunch of arch-specific gradient metrics
        for name, param in self.model.named_parameters():
            if (not self.is_fsdp) or self.quiet_logs:
                break
            if param.grad is None:
                continue
            nice_name = name.replace("module._fsdp_wrapped_module._fpw_module.", "")
            nice_name = nice_name.replace("_fsdp_wrapped_module._fpw_module.", "")
            nice_name = nice_name.replace("._fsdp_wrapped_module.flat_param_0", "")
            nice_name = nice_name.replace("decoder.layers.", "layer")
            # threshold for near zeros
            threshold = torch.finfo(param.grad.dtype).tiny * 2
            with torch.no_grad():
                g = param.grad
                if hasattr(self.optimizer, "_multiply_factor"):
                    g = self.optimizer._multiply_factor * g
                norm = g.norm(p=2, dim=-1, dtype=torch.float32)
                max_ = g.max()
                nz = ((g > -threshold) & (g < threshold)).sum() / g.numel()
            # priorities for printing order
            metrics.log_scalar(f"gnorm_{nice_name}", norm, priority=10)
            metrics.log_scalar(f"gmax_{nice_name}", max_, priority=11)
            metrics.log_scalar(f"gzero_{nice_name}", nz, priority=12)
            with torch.no_grad():
                norm = param.norm(p=2, dim=-1, dtype=torch.float32)
                max_ = param.max()
                nz = ((param > -threshold) & (param < threshold)).sum() / param.numel()
            # priorities for printing order
            metrics.log_scalar(f"pnorm_{nice_name}", norm, priority=13)
            metrics.log_scalar(f"pmax_{nice_name}", max_, priority=14)
            metrics.log_scalar(f"pzero_{nice_name}", nz, priority=15)

        # standard code
        if grad_norm is not None and (
            not torch.is_tensor(grad_norm) or torch.isfinite(grad_norm)
        ):
            metrics.log_speed("ups", 1.0, priority=100, round=2)
            metrics.log_scalar("gnorm", grad_norm, priority=400, round=3)
            metrics.log_scalar("ewm_loss", self._ewm_loss, priority=700, round=2)
            metrics.log_scalar("ewm_loss_ratio", ewm_loss_ratio, priority=710, round=4)
            metrics.log_scalar(
                "skipped_loss_spikes", self._skipped_loss_spikes, priority=720
            )
            self._skipped_loss_spikes = 0
            if self.cfg.optimization.clip_norm > 0:
                metrics.log_scalar(
                    "clip",
                    torch.where(
                        grad_norm > self.cfg.optimization.clip_norm,
                        grad_norm.new_tensor(100),
                        grad_norm.new_tensor(0),
                    ),
                    priority=500,
                    round=1,
                )

        with metrics.aggregate() as agg:
            if logging_outputs is not None:
                self.task.reduce_metrics(logging_outputs, self.get_criterion())
                del logging_outputs

            # extra warning for criterions that don't properly log a loss value
            if "loss" not in agg:
                if "loss" not in self._warn_once:
                    self._warn_once.add("loss")
                    logger.warning(
                        "Criterion.reduce_metrics did not log a 'loss' value, "
                        "which may break some functionality"
                    )
                metrics.log_scalar("loss", -1)

            # support legacy interface
            logging_output = agg.get_smoothed_values()
            logging_output["sample_size"] = sample_size
            for key_to_delete in ["ppl", "wps", "wpb", "bsz"]:
                if key_to_delete in logging_output:
                    del logging_output[key_to_delete]

            return logging_output

    def _log_gpu_mem_stats(self):
        # log minimum free memory over the iteration
        cuda_gb_allocated = torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024
        cuda_gb_reserved = torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024
        torch.cuda.reset_peak_memory_stats()
        cuda_gb_free = self.cuda_env.total_memory_in_GB - cuda_gb_allocated
        metrics.log_scalar(
            "cuda_gb_allocated", cuda_gb_allocated, priority=1500, round=1, weight=0
        )
        metrics.log_scalar(
            "cuda_gb_reserved", cuda_gb_reserved, priority=1500, round=1, weight=0
        )
        metrics.log_scalar(
            "cuda_gb_free", cuda_gb_free, priority=1500, round=1, weight=0
        )
        # log nvidia smi stats
        if self.cfg.common.log_nvidia_smi:
            nvidia_smi_stats = metrics.nvidia_smi_gpu_memory_stats()
            for key, val in nvidia_smi_stats.items():
                metrics.log_scalar(key, val, priority=1500, round=1, weight=0)


def _catalog_shared_params(module, memo=None, prefix=""):
    if memo is None:
        first_call = True
        memo = {}
    else:
        first_call = False
    for name, param in module._parameters.items():
        param_prefix = prefix + ("." if prefix else "") + name
        if param not in memo:
            memo[param] = []
        memo[param].append(param_prefix)
    for name, m in module._modules.items():
        if m is None:
            continue
        submodule_prefix = prefix + ("." if prefix else "") + name
        _catalog_shared_params(m, memo, submodule_prefix)
    if first_call:
        return [x for x in memo.values() if len(x) > 1]


def _get_module_by_path(module, path):
    path = path.split(".")
    for name in path:
        module = getattr(module, name)
    return module


def _set_module_by_path(module, path, value):
    path = path.split(".")
    for name in path[:-1]:
        module = getattr(module, name)
    setattr(module, path[-1], value)


class SpikeError(Exception):
    pass
