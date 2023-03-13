#!/usr/bin/env python3 -u
# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Train a new model on one or across multiple GPUs.
"""

import argparse
import functools
import logging
import math
import os
import subprocess
import sys
import time
import socket
import re
from typing import Dict, Optional, Any, List, Tuple, Callable
import warnings

import numpy as np
import torch
import torch.profiler as profiler
from omegaconf import DictConfig, OmegaConf

from metaseq import (
    checkpoint_utils,
    options,
    tasks,
    utils,
)
from metaseq.data import iterators, data_utils
from metaseq.data.plasma_utils import PlasmaStore
from metaseq.dataclass.utils import convert_namespace_to_omegaconf
from metaseq.distributed import fsdp_enable_wrap, fsdp_wrap, utils as distributed_utils
from metaseq.file_io import PathManager
from metaseq.logging import meters, metrics, progress_bar
from metaseq.trainer import Trainer

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logging.Formatter.converter = time.gmtime  # Enforce UTC timestamps
logger = logging.getLogger("metaseq.cli.train")


def main(cfg: DictConfig) -> None:
    utils.import_user_module(cfg.common)
    warnings.filterwarnings(
        "ignore",
        message="torch.distributed._all_gather_base is a private function and will be deprecated",
    )
    warnings.filterwarnings(
        "ignore",
        message="torch.distributed._reduce_scatter_base is a private function and will be deprecated",
    )
    # replace with actual job id
    slurm_jobid = os.environ.get("SLURM_JOBID", None)
    if "%jobid" in cfg.checkpoint.save_dir and slurm_jobid is not None:
        cfg.checkpoint.save_dir = cfg.checkpoint.save_dir.replace("%jobid", slurm_jobid)

    checkpoint_utils.verify_checkpoint_directory(cfg.checkpoint.save_dir)

    if distributed_utils.is_master(cfg.distributed_training) and os.environ.get(
        "METASEQ_SAVE_DIR"
    ):
        # save a (vaguely human readable) copy of the training config
        # TODO(roller): only works when launched with a sweep script
        # should fix that
        OmegaConf.save(
            config=_flatten_config(cfg),
            f=os.path.join(os.environ["METASEQ_SAVE_DIR"], "config.yml"),
        )

    if (
        distributed_utils.is_master(cfg.distributed_training)
        and "job_logging_cfg" in cfg
    ):
        # make hydra logging work with ddp (see # see https://github.com/facebookresearch/hydra/issues/1126)
        logging.config.dictConfig(OmegaConf.to_container(cfg.job_logging_cfg))

    assert (
        cfg.dataset.max_tokens is not None or cfg.dataset.batch_size is not None
    ), "Must specify batch size either with --max-tokens or --batch-size"
    metrics.reset()

    if cfg.checkpoint.local_save_interval_updates > 0:
        assert (
            cfg.checkpoint.save_interval_updates > 0
        ), "local save must be used with --save-interval-updates > 0"
        assert (
            cfg.checkpoint.save_interval_updates
            % cfg.checkpoint.local_save_interval_updates
            == 0
        ), "--save-interval-updates must be a multiple of --local-save-interval-updates"

    if cfg.common.log_file is not None:
        handler = logging.FileHandler(filename=cfg.common.log_file)
        logger.addHandler(handler)

    np.random.seed(cfg.common.seed)
    utils.set_torch_seed(cfg.common.seed)

    # Print nvidia smi stats
    logger.info(metrics.get_nvidia_smi_gpu_memory_stats_str())

    # Print args
    logger.info(cfg)

    # Setup task, e.g., translation, language modeling, etc.
    if cfg.distributed_training.task_ddp_backend == "fully_sharded":
        # As the task is non-trainable, we switch flags to more optimized ones.
        # See https://github.com/facebookresearch/metaseq/pull/668 for when/why this was added.
        orig_memory_efficient_fp16 = cfg.distributed_training.memory_efficient_fp16
        orig_fp32_reduce_scatter = cfg.distributed_training.fp32_reduce_scatter
        # Clobber memory_efficient_fp16 and fp32_reduce_scatter
        cfg.distributed_training.memory_efficient_fp16 = cfg.distributed_training.fp16
        cfg.distributed_training.fp32_reduce_scatter = not cfg.distributed_training.fp16

        with fsdp_enable_wrap(
            cfg.distributed_training,
            use_sharded_state=cfg.distributed_training.use_sharded_state,
        ):
            task = tasks.setup_task(cfg.task)

        # Reset memory_efficient_fp16 and fp32_reduce_scatter values.
        cfg.distributed_training.memory_efficient_fp16 = orig_memory_efficient_fp16
        cfg.distributed_training.fp32_reduce_scatter = orig_fp32_reduce_scatter
    else:
        task = tasks.setup_task(cfg.task)

    # Build model and criterion
    assert cfg.criterion, "Please specify criterion to train a model"

    if cfg.distributed_training.ddp_backend == "fully_sharded":
        with fsdp_enable_wrap(
            cfg.distributed_training,
            use_sharded_state=cfg.distributed_training.use_sharded_state,
        ):
            model = fsdp_wrap(
                task.build_model(cfg.model),
                process_group=distributed_utils.get_data_parallel_group(),
            )
    else:
        model = task.build_model(cfg.model)

    # TODO[Susan]: FSDP on criterion?
    criterion = task.build_criterion(cfg.criterion)

    logger.info(model)
    logger.info("task: {}".format(task.__class__.__name__))
    logger.info("model: {}".format(model.__class__.__name__))
    logger.info("criterion: {}".format(criterion.__class__.__name__))
    logger.info(
        "num. model params: {:,} (num. trained: {:,})".format(
            sum(getattr(p, "_orig_size", p).numel() for p in model.parameters()),
            sum(
                getattr(p, "_orig_size", p).numel()
                for p in model.parameters()
                if p.requires_grad
            ),
        )
    )
    logger.info(metrics.get_nvidia_smi_gpu_memory_stats_str())

    # Load valid dataset (we load training data below, based on the latest checkpoint)
    # We load the valid dataset AFTER building the model
    data_utils.raise_if_valid_subsets_unintentionally_ignored(cfg)
    if cfg.dataset.combine_valid_subsets:
        task.load_dataset("valid", combine=True, epoch=1)
    else:
        for valid_sub_split in cfg.dataset.valid_subset.split(","):
            task.load_dataset(valid_sub_split, combine=False, epoch=1)

    # Build trainer
    trainer = Trainer(cfg, task, model, criterion)
    logger.info(
        "training on {} devices (GPUs/TPUs)".format(
            cfg.distributed_training.distributed_world_size
        )
    )
    logger.info(
        "max tokens per GPU = {} and batch size per GPU = {}".format(
            cfg.dataset.max_tokens,
            cfg.dataset.batch_size,
        )
    )
    logger.info(metrics.get_nvidia_smi_gpu_memory_stats_str())

    # Load the latest checkpoint if one is available and restore the
    # corresponding train iterator
    extra_state, epoch_itr = checkpoint_utils.load_checkpoint(
        cfg.checkpoint,
        trainer,
        # don't cache epoch iterators for sharded datasets
        disable_iterator_cache=True,
    )

    max_epoch = cfg.optimization.max_epoch or math.inf
    train_meter = meters.StopwatchMeter()
    train_meter.start()
    while epoch_itr.next_epoch_idx <= max_epoch:
        # train for one epoch
        valid_losses, should_stop = train(cfg, trainer, task, epoch_itr)
        if should_stop:
            break

        # only use first validation loss to update the learning rate
        trainer.lr_step(epoch_itr.epoch, valid_losses[0])

        epoch_itr = trainer.get_train_iterator(
            epoch_itr.next_epoch_idx,
            # don't cache epoch iterators for sharded datasets
            disable_iterator_cache=True,
        )
    train_meter.stop()
    logger.info("done training in {:.1f} seconds".format(train_meter.sum))


def get_skip_batches(to_skip):
    skip_batches = {}
    for skip_range in to_skip.split(","):
        if "-" in skip_range:
            start, end = skip_range.split("-")
            skip_batches[int(start)] = int(end) - int(start) + 1
        else:
            skip_batches[int(skip_range)] = 1
    return skip_batches


@metrics.aggregate("train")
def train(
    cfg: DictConfig, trainer: Trainer, task: tasks.BaseTask, epoch_itr
) -> Tuple[List[Optional[float]], bool]:
    """Train the model for one epoch and return validation losses."""
    # Initialize data iterator
    itr = epoch_itr.next_epoch_itr(
        fix_batches_to_gpus=cfg.distributed_training.fix_batches_to_gpus,
        shuffle=True,
    )
    update_freq = (
        cfg.optimization.update_freq[epoch_itr.epoch - 1]
        if epoch_itr.epoch <= len(cfg.optimization.update_freq)
        else cfg.optimization.update_freq[-1]
    )
    if update_freq > 1:
        itr = iterators.GroupedIterator(
            itr,
            update_freq,
            skip_remainder_batch=True,
        )

    progress = progress_bar.get_progress_bar(
        itr,
        log_format=cfg.common.log_format,
        log_file=cfg.common.log_file,
        log_interval=cfg.common.log_interval,
        epoch=epoch_itr.epoch,
        tensorboard_logdir=(
            cfg.common.tensorboard_logdir
            if distributed_utils.is_master(cfg.distributed_training)
            else None
        ),
        aim_repo=(
            cfg.common.aim_repo
            if distributed_utils.is_master(cfg.distributed_training)
            else None
        ),
        aim_run_hash=(
            cfg.common.aim_run_hash
            if distributed_utils.is_master(cfg.distributed_training)
            else None
        ),
        aim_param_checkpoint_dir=cfg.checkpoint.save_dir,
        wandb_project=(
            cfg.common.wandb_project
            if distributed_utils.is_master(cfg.distributed_training)
            else None
        ),
        wandb_run_name=os.environ.get(
            "WANDB_NAME", os.path.basename(cfg.checkpoint.save_dir)
        ),
    )
    progress.update_config(_flatten_config(cfg))

    trainer.begin_epoch(epoch_itr.epoch)
    valid_subsets = cfg.dataset.valid_subset.split(",")
    should_stop = False
    logger.info("Start iterating over samples")

    def train(
        samples,
    ):
        with metrics.aggregate("train_inner"):
            if update_freq == 1:
                samples = [samples]
            log_output = trainer.train_step(samples)

        if log_output is not None:  # not OOM, overflow, ...
            # log mid-epoch stats
            num_updates = trainer.get_num_updates()
            if num_updates % cfg.common.log_interval == 0:
                stats = get_training_stats(metrics.get_smoothed_values("train_inner"))
                progress.log(stats, tag="train_inner", step=num_updates)

                # reset mid-epoch stats after each log interval
                # the end-of-epoch stats will still be preserved
                metrics.reset_meters("train_inner")

        end_of_epoch = not itr.has_next()
        if end_of_epoch:
            grank = distributed_utils.get_global_rank()
            dataset = epoch_itr.dataset
            while not hasattr(dataset, "len_cache"):
                dataset = dataset.dataset
            len_cache = tuple(dataset.len_cache.data)
            cache_hash = hash(len_cache)
            contains_zero = any([x == 0 for x in len_cache])
            logger.warning(
                " ".join(
                    [
                        f"End of Epoch on rank {grank}:",
                        f"sequences_consumed={itr.sequences_consumed}",
                        f"n={itr.n}",
                        f"len_cache_hash={cache_hash}",
                        f"len_cache_has_zeros={contains_zero}",
                    ]
                )
            )

        valid_losses, should_stop = validate_and_save(
            cfg,
            trainer,
            task,
            epoch_itr,
            valid_subsets,
            end_of_epoch,
            log_output is not None,
        )

        return valid_losses, should_stop

    skip_batches = None
    if len(cfg.dataset.skip_batches) > 0:
        skip_batches = get_skip_batches(cfg.dataset.skip_batches)

    progress_iter = iter(progress)
    i = 0
    while True:
        try:
            with metrics.aggregate("train_inner"):
                metrics.log_start_time("time_sample_fetch", round=4)
                samples = next(progress_iter)
                metrics.log_stop_time("time_sample_fetch")

            current_step = trainer.get_num_updates() + 1
            if (
                skip_batches is not None
                and current_step in skip_batches
                and skip_batches[current_step] > 0
            ):
                skip_batches[current_step] -= 1
                logger.info(
                    f"Skipping batches starting from step {current_step} with "
                    f"{skip_batches[current_step]} batches left to skip"
                )
                continue

            if (
                distributed_utils.get_global_rank() == 0
                and cfg.common.profile
                and i == 5
            ):
                logger.info("STARTING PROFILER")
                with profiler.profile(
                    profile_memory=True, with_stack=True, record_shapes=True
                ) as prof:
                    valid_losses, should_stop = train(samples)
                torch.cuda.synchronize()
                with open(
                    os.path.join(cfg.checkpoint.save_dir, "memory_usage.txt"), "a"
                ) as sourceFile:
                    print(
                        prof.key_averages(group_by_stack_n=5).table(
                            sort_by="self_cuda_memory_usage", row_limit=10
                        ),
                        file=sourceFile,
                    )
                prof.export_chrome_trace(
                    os.path.join(cfg.checkpoint.save_dir, "profiler_trace.json")
                )
            else:
                valid_losses, should_stop = train(samples)
            if should_stop:
                break

            i += 1
        except StopIteration:
            break

    # reset epoch-level meters
    metrics.reset_meters("train")
    return valid_losses, should_stop


def _flatten_config(cfg: DictConfig):
    config = OmegaConf.to_container(cfg)
    # remove any legacy Namespaces and replace with a single "args"
    namespace = None
    for k, v in list(config.items()):
        if isinstance(v, argparse.Namespace):
            namespace = v
            del config[k]
    if namespace is not None:
        config["args"] = vars(namespace)
    return config


def validate_and_save(
    cfg: DictConfig,
    trainer: Trainer,
    task: tasks.BaseTask,
    epoch_itr,
    valid_subsets: List[str],
    end_of_epoch: bool,
    was_successful_step: bool,
) -> Tuple[List[Optional[float]], bool]:
    num_updates = trainer.get_num_updates()
    max_update = cfg.optimization.max_update or math.inf

    # was_successful_step is necessary since we don't increment step counters
    # on OOM or overflow. Thus if we get multiple bad steps right after
    # loading a checkpoint (when step counter is exactly when we would step)
    # then we will start overwriting! omg!

    # Stopping conditions (and an additional one based on validation loss later
    # on)
    should_stop = False
    if num_updates >= max_update:
        should_stop = True
        logger.info(
            f"Stopping training due to "
            f"num_updates: {num_updates} >= max_update: {max_update}"
        )

    save_locally = (
        cfg.checkpoint.local_save_interval_updates > 0
        and num_updates > 0
        and num_updates % cfg.checkpoint.local_save_interval_updates == 0
    )
    save_to_NFS = (
        cfg.checkpoint.save_interval_updates > 0
        and num_updates > 0
        and num_updates % cfg.checkpoint.save_interval_updates == 0
    )

    do_save = (
        (
            end_of_epoch
            and cfg.checkpoint.save_interval_epochs > 0
            and epoch_itr.epoch % cfg.checkpoint.save_interval_epochs == 0
        )
        or (
            (save_locally or save_to_NFS)
            and num_updates >= cfg.dataset.validate_after_updates
            and was_successful_step
        )
        or should_stop
    )
    do_validate = (
        should_stop
        or (
            cfg.dataset.validate_interval_updates > 0
            and num_updates > 0
            and num_updates % cfg.dataset.validate_interval_updates == 0
            and was_successful_step
        )
    ) and not cfg.dataset.disable_validation

    # Save checkpoint before validating.
    if do_save:
        checkpoint_utils.save_checkpoint(
            cfg.checkpoint,
            trainer,
            epoch_itr,
            training_finished=should_stop,
            async_callback_fn=functools.partial(
                post_checkpoint_callback, cfg, num_updates, should_stop
            )
            if cfg.checkpoint.cloud_upload_path
            else None,
        )

    valid_losses = [None]
    if do_validate:
        valid_losses = validate(cfg, trainer, task, epoch_itr, valid_subsets)

    trainer.reset_dummy_batch(epoch_itr.first_batch)
    return valid_losses, should_stop


def _checkpoint_add_directory(basename):
    pattern = r"(checkpoint(\d+|_\d+|_last))(.*)"
    m = re.match(pattern, basename)
    assert m, f"checkpoint file doesn't follow pattern {pattern}"
    return m[1], f"checkpoint{m[3]}"


def post_checkpoint_callback(cfg, num_updates, training_finished, filename):
    if cfg.checkpoint.cloud_upload_path is not None:
        if "blob.core.windows.net" in cfg.checkpoint.cloud_upload_path:
            azcopy_logs = filename + "_azcopy_logs"
            os.environ["AZCOPY_CONCURRENCY_VALUE"] = "10"
            os.environ["AZCOPY_LOG_LOCATION"] = azcopy_logs
            os.makedirs(azcopy_logs, exist_ok=True)
            logger.info(
                f"preparing to azcopy {filename} to {cfg.checkpoint.cloud_upload_path}; logs in {azcopy_logs}"
            )
            cmd = [
                "azcopy",  # TODO(susanz): require azcopy to be installed.
                "copy",
                "--cap-mbps",
                "96.0",
                filename,
                cfg.checkpoint.cloud_upload_path,
            ]
            res = _run_azcopy(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if res.returncode != 0:
                print("Error: {}, azcopy failed".format(res.returncode))
                print("Azcopy stdout = {}".format(res.stdout))
                sys.exit(1)
            # Delete original checkpoint on local storage
            # TODO make this configurable
            logger.info(
                f"Successfully copied {filename} to {cfg.checkpoint.cloud_upload_path}"
            )
            os.remove(filename)
        elif cfg.checkpoint.cloud_upload_path.startswith("nfs:"):
            path, basename = os.path.split(filename)
            checkpoint_dir, checkpoint_file = _checkpoint_add_directory(basename)
            destination_checkpoints_dir = cfg.checkpoint.cloud_upload_path[4:]
            temporary_checkpoint_file = f"_{checkpoint_file}"
            try:
                os.mkdir(os.path.join(destination_checkpoints_dir, checkpoint_dir))
            except FileExistsError:
                pass  # another worker got here first
            logger.info(f"Beginning copy of {filename} to NFS")

            # copy the checkpoint from local storage to nfs in the background
            subprocess.run(
                [
                    "cp",
                    filename,
                    os.path.join(
                        destination_checkpoints_dir,
                        checkpoint_dir,
                        temporary_checkpoint_file,
                    ),
                ]
            )

            logger.info(f"Renaming {temporary_checkpoint_file} -> {checkpoint_file}")
            # atomic rename _checkpointfile -> checkpointfile
            # this way we know that if present the checkpoint file is complete
            os.rename(
                os.path.join(
                    destination_checkpoints_dir,
                    checkpoint_dir,
                    temporary_checkpoint_file,
                ),
                os.path.join(
                    destination_checkpoints_dir, checkpoint_dir, checkpoint_file
                ),
            )
            os.remove(filename)

            # Start running evals on uploaded checkpoint
            nfs_evaluation(
                cfg,
                num_updates,
                training_finished,
                checkpoint_dir,
                destination_checkpoints_dir,
            )

        else:
            try:
                # PathManager only supports writing to S3, but this function call
                # can be replaced with other APIs for copying checkpoints.
                PathManager.copy_from_local(
                    filename,
                    os.path.join(
                        cfg.checkpoint.cloud_upload_path, os.path.basename(filename)
                    ),
                    overwrite=True,
                )
            except (FileNotFoundError, AssertionError) as e:
                logger.info(f"could not upload {filename}: {e}")


def nfs_evaluation(
    cfg, num_updates, training_finished, checkpoint_dir, destination_checkpoints_dir
):
    if (
        cfg.checkpoint.nfs_eval_script_path is not None
        and distributed_utils.get_global_rank() == 0
        and (
            (
                cfg.checkpoint.nfs_eval_frequency > 0
                and num_updates % cfg.checkpoint.nfs_eval_frequency == 0
            )
            or training_finished
        )
    ):
        for retry in range(cfg.checkpoint.nfs_eval_num_attempts):
            time.sleep(cfg.checkpoint.nfs_eval_attempt_wait_minutes * 60)

            current_checkpoint_path = os.path.join(
                destination_checkpoints_dir, checkpoint_dir
            )
            num_files = os.listdir(current_checkpoint_path)
            # only count completed checkpoints
            finished_checkpoint_parts = len(
                [f for f in num_files if not f.startswith("_")]
            )
            if (
                finished_checkpoint_parts
                == cfg.distributed_training.distributed_world_size
            ):
                logger.info(
                    f"All checkpoint parts for {checkpoint_dir} are in NFS, will now start to run evals"
                )
                script_dir = os.path.join(
                    os.environ.get("METASEQ_SAVE_DIR"),
                    cfg.checkpoint.nfs_eval_script_path,
                )
                res = subprocess.run(
                    [
                        "bash",
                        script_dir,
                        os.path.join(current_checkpoint_path, "checkpoint.pt"),
                    ],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                if res.returncode == 0:
                    logger.info(f"Sucessfully evaluated {checkpoint_dir}")
                else:
                    logger.error(f"Error during evaluation: {res.returncode}")
                    logger.error(f"Eval script stdout = {res.stdout}")
                    logger.error(f"Eval script stderr = {res.stderr}")
                return
        logger.error(
            (
                f"Did not evaluate {checkpoint_dir}, as only {num_files}/"
                f"{cfg.distributed_training.distributed_world_size} "
                "checkpoint parts were copied to NFS within waiting time"
            )
        )


def _run_azcopy(cmd, stdout, stderr):
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def get_training_stats(stats: Dict[str, Any]) -> Dict[str, Any]:
    stats["wall"] = round(metrics.get_meter("default", "wall").elapsed_time, 0)
    return stats


def validate(
    cfg: DictConfig,
    trainer: Trainer,
    task: tasks.BaseTask,
    epoch_itr,
    subsets: List[str],
) -> List[Optional[float]]:
    """Evaluate the model on the validation set(s) and return the losses."""

    if cfg.dataset.fixed_validation_seed is not None:
        # set fixed seed for every validation
        utils.set_torch_seed(cfg.dataset.fixed_validation_seed)

    trainer.begin_valid_epoch(epoch_itr.epoch)
    valid_losses = []
    with metrics.aggregate(new_root=True) as combined_agg:
        for subset in subsets:
            logger.info(
                'begin validation on "{}" subset on rank {}'.format(
                    subset, distributed_utils.get_global_rank()
                )
            )

            # Initialize data iterator
            itr = trainer.get_valid_iterator(subset).next_epoch_itr(
                shuffle=False, set_dataset_epoch=False  # use a fixed valid set
            )

            logger.info(
                'got valid iterator on "{}" subset on rank {}'.format(
                    subset, distributed_utils.get_global_rank()
                )
            )

            progress = progress_bar.get_progress_bar(
                itr,
                log_format=cfg.common.log_format,
                log_interval=cfg.common.log_interval,
                epoch=epoch_itr.epoch,
                prefix=f"valid on '{subset}' subset",
                tensorboard_logdir=(
                    cfg.common.tensorboard_logdir
                    if distributed_utils.is_master(cfg.distributed_training)
                    else None
                ),
                aim_repo=(
                    cfg.common.aim_repo
                    if distributed_utils.is_master(cfg.distributed_training)
                    else None
                ),
                aim_run_hash=(
                    cfg.common.aim_run_hash
                    if distributed_utils.is_master(cfg.distributed_training)
                    else None
                ),
                aim_param_checkpoint_dir=cfg.checkpoint.save_dir,
                wandb_project=(
                    cfg.common.wandb_project
                    if distributed_utils.is_master(cfg.distributed_training)
                    else None
                ),
                wandb_run_name=os.environ.get(
                    "WANDB_NAME", os.path.basename(cfg.checkpoint.save_dir)
                ),
            )

            logger.info(
                'Begin looping over validation "{}" subset with length "{}"'.format(
                    subset, len(progress)
                )
            )

            # create a new root metrics aggregator so validation metrics
            # don't pollute other aggregators (e.g., train meters)
            with metrics.aggregate() as agg:
                for i, sample in enumerate(progress):
                    if (
                        cfg.dataset.max_valid_steps is not None
                        and i > cfg.dataset.max_valid_steps
                    ):
                        break
                    trainer.valid_step(sample, num_step=i)
            # log validation stats
            stats = add_num_updates_to_stats(trainer, agg.get_smoothed_values())
            progress.print(stats, tag=subset, step=trainer.get_num_updates())
    stats = add_num_updates_to_stats(trainer, combined_agg.get_smoothed_values())
    progress.print(stats, tag="valid/combined", step=trainer.get_num_updates())
    return valid_losses


def add_num_updates_to_stats(trainer: Trainer, stats: Dict[str, Any]) -> Dict[str, Any]:
    stats["num_updates"] = trainer.get_num_updates()
    return stats


def set_local_per_worker_env_variables():
    savedir = os.environ.get("METASEQ_SAVE_DIR")
    if savedir is not None:
        hostname = socket.gethostname()

        restart = int(os.environ.get("SLURM_RESTART_COUNT", "0"))
        nccl_dir = os.path.join(savedir, "nccl", f"restart_{restart:03d}")
        os.makedirs(nccl_dir, exist_ok=True)
        rank = int(os.environ.get("SLURM_PROCID", "0"))
        os.environ["NCCL_DEBUG_FILE"] = os.path.join(
            nccl_dir, f"rank_{rank:04d}_{hostname}"
        )

        # save a copy of all our environmental variables
        env_dir = os.path.join(savedir, "envs", f"restart_{restart:03d}")
        os.makedirs(env_dir, exist_ok=True)
        with open(os.path.join(env_dir, f"rank_{rank:04d}_{hostname}"), "w") as f:
            for key in sorted(os.environ.keys()):
                f.write(f"{key}={os.environ[key]}\n")


def cli_main(
    modify_parser: Optional[Callable[[argparse.ArgumentParser], None]] = None
) -> None:
    set_local_per_worker_env_variables()
    parser = options.get_training_parser()
    args = options.parse_args_and_arch(parser, modify_parser=modify_parser)

    # For training - this is where arg parsing happens.
    cfg = convert_namespace_to_omegaconf(args)

    if cfg.common.use_plasma_view:
        server = PlasmaStore(path=cfg.common.plasma_path)
        logger.info(
            f"Started plasma server pid {server.server.pid} {cfg.common.plasma_path}"
        )

    distributed_utils.call_main(cfg, main)


if __name__ == "__main__":
    cli_main()
