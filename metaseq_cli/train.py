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
from typing import Dict, Optional, Any, List, Tuple, Callable

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
from metaseq.model_parallel.megatron_trainer import MegatronTrainer
from metaseq.trainer import Trainer

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logging.Formatter.converter = time.gmtime  # Enforce UTC timestamps
logger = logging.getLogger("metaseq_cli.train")


def main(cfg: DictConfig) -> None:
    utils.import_user_module(cfg.common)

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

    if cfg.common.log_file is not None:
        handler = logging.FileHandler(filename=cfg.common.log_file)
        logger.addHandler(handler)

    np.random.seed(cfg.common.seed)
    utils.set_torch_seed(cfg.common.seed)

    checkpoint_utils.verify_checkpoint_directory(cfg.checkpoint.save_dir)

    # Print nvidia smi stats
    logger.info(metrics.get_nvidia_smi_gpu_memory_stats_str())

    # Print args
    logger.info(cfg)

    if cfg.checkpoint.write_checkpoints_asynchronously:
        try:
            import iopath  # noqa: F401
        except ImportError:
            logging.exception(
                "Asynchronous checkpoint writing is specified but iopath is "
                "not installed: `pip install iopath`"
            )
            return

    # Setup task, e.g., translation, language modeling, etc.
    task = tasks.setup_task(cfg.task)

    assert cfg.criterion, "Please specify criterion to train a model"

    # Build model and criterion
    if cfg.distributed_training.ddp_backend == "fully_sharded":
        extra = {
            "use_sharded_state": cfg.distributed_training.use_sharded_state,
        }

        with fsdp_enable_wrap(cfg.distributed_training, **extra):
            model = fsdp_wrap(
                task.build_model(cfg.model),
                process_group=distributed_utils.get_data_parallel_group(),
            )
    else:
        model = task.build_model(cfg.model)
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
    if cfg.common.model_parallel_size == 1:
        trainer = Trainer(cfg, task, model, criterion)
    else:
        trainer = MegatronTrainer(cfg, task, model, criterion)
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

    # ioPath implementation to wait for all asynchronous file writes to complete.
    if cfg.checkpoint.write_checkpoints_asynchronously:
        logger.info(
            "ioPath PathManager waiting for all asynchronous checkpoint "
            "writes to finish."
        )
        PathManager.async_close()
        logger.info("ioPath PathManager finished waiting.")


def should_stop_early(cfg: DictConfig, valid_loss: float) -> bool:
    # skip check if no validation was done in the current epoch
    if valid_loss is None:
        return False
    if cfg.checkpoint.patience <= 0:
        return False

    def is_better(a, b):
        return a > b if cfg.checkpoint.maximize_best_checkpoint_metric else a < b

    prev_best = getattr(should_stop_early, "best", None)
    if prev_best is None or is_better(valid_loss, prev_best):
        should_stop_early.best = valid_loss
        should_stop_early.num_runs = 0
        return False
    else:
        should_stop_early.num_runs += 1
        if should_stop_early.num_runs >= cfg.checkpoint.patience:
            logger.info(
                "early stop since valid performance hasn't improved for last {} runs".format(
                    cfg.checkpoint.patience
                )
            )
            return True
        else:
            return False


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
            skip_remainder_batch=(
                not cfg.optimization.train_with_epoch_remainder_batch
            ),
        )

    progress = progress_bar.progress_bar(
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
        default_log_format="json",
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
    num_updates = trainer.get_num_updates()
    logger.info("Start iterating over samples")

    def train(
        i,
        samples,
    ):
        with metrics.aggregate("train_inner"), torch.autograd.profiler.record_function(
            "train_step-%d" % i
        ):
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

    for i, samples in enumerate(progress):
        if (
            distributed_utils.get_global_rank() == 0
            and cfg.common.new_profiler
            and i == 5
        ):
            logger.info("STARTING PROFILER")
            with profiler.profile() as prof:
                valid_losses, should_stop = train(i, samples)
            torch.cuda.synchronize()
            prof.export_chrome_trace(
                os.path.join(cfg.checkpoint.save_dir, "profiler_trace.json")
            )
        else:
            valid_losses, should_stop = train(i, samples)
        if should_stop:
            break

    # log end-of-epoch stats
    logger.info("end of epoch {} (average epoch stats below)".format(epoch_itr.epoch))
    stats = get_training_stats(metrics.get_smoothed_values("train"))
    progress.print(stats, tag="train", step=num_updates)

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

    do_save = (
        (end_of_epoch and epoch_itr.epoch % cfg.checkpoint.save_interval == 0)
        or should_stop
        or (
            cfg.checkpoint.save_interval_updates > 0
            and num_updates > 0
            and num_updates % cfg.checkpoint.save_interval_updates == 0
            and num_updates >= cfg.dataset.validate_after_updates
            and was_successful_step
        )
    )
    do_validate = (
        (
            not end_of_epoch and do_save and not cfg.checkpoint.no_best_checkpoints
        )  # validate during mid-epoch saves
        or (end_of_epoch and epoch_itr.epoch % cfg.dataset.validate_interval == 0)
        or should_stop
        or (
            cfg.dataset.validate_interval_updates > 0
            and num_updates > 0
            and num_updates % cfg.dataset.validate_interval_updates == 0
            and was_successful_step
        )
    ) and not cfg.dataset.disable_validation
    valid_losses = [None]
    if do_validate:
        valid_losses = validate(cfg, trainer, task, epoch_itr, valid_subsets)

    should_stop |= should_stop_early(cfg, valid_losses[0])

    # Save checkpoint
    if do_save or should_stop:
        checkpoint_utils.save_checkpoint(
            cfg.checkpoint,
            trainer,
            epoch_itr,
            valid_losses[0],
            training_finished=should_stop,
            async_callback_fn=functools.partial(post_checkpoint_callback, cfg)
            if cfg.checkpoint.cloud_upload_path
            else None,
        )

    trainer.reset_dummy_batch(epoch_itr.first_batch)
    return valid_losses, should_stop


def post_checkpoint_callback(cfg, filename):
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
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
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

            progress = progress_bar.progress_bar(
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
                default_log_format="json",
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
                    trainer.valid_step(sample)
            # log validation stats
            stats = get_valid_stats(cfg, trainer, agg.get_smoothed_values())
            progress.print(stats, tag=subset, step=trainer.get_num_updates())
            valid_losses.append(stats[cfg.checkpoint.best_checkpoint_metric])
    stats = get_valid_stats(cfg, trainer, combined_agg.get_smoothed_values())
    progress.print(stats, tag="valid/combined", step=trainer.get_num_updates())
    return valid_losses


def get_valid_stats(
    cfg: DictConfig, trainer: Trainer, stats: Dict[str, Any]
) -> Dict[str, Any]:
    stats["num_updates"] = trainer.get_num_updates()
    if hasattr(checkpoint_utils.save_checkpoint, "best"):
        key = "best_{0}".format(cfg.checkpoint.best_checkpoint_metric)
        best_function = max if cfg.checkpoint.maximize_best_checkpoint_metric else min
        stats[key] = best_function(
            checkpoint_utils.save_checkpoint.best,
            stats[cfg.checkpoint.best_checkpoint_metric],
        )
    return stats


def cli_main(
    modify_parser: Optional[Callable[[argparse.ArgumentParser], None]] = None
) -> None:
    parser = options.get_training_parser()
    args = options.parse_args_and_arch(parser, modify_parser=modify_parser)

    # For training - this is where arg parsing happens.
    cfg = convert_namespace_to_omegaconf(args)

    if cfg.common.use_plasma_view:
        server = PlasmaStore(path=cfg.common.plasma_path)
        logger.info(
            f"Started plasma server pid {server.server.pid} {cfg.common.plasma_path}"
        )

    if args.profile:
        with torch.cuda.profiler.profile():
            with torch.autograd.profiler.emit_nvtx():
                distributed_utils.call_main(cfg, main)
    else:
        distributed_utils.call_main(cfg, main)


if __name__ == "__main__":
    cli_main()
