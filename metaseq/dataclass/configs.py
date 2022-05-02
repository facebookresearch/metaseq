# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import sys
from dataclasses import _MISSING_TYPE, dataclass, field
from typing import Any, List, Optional

import torch
from omegaconf import II, MISSING

from metaseq.dataclass.constants import (
    DATASET_IMPL_CHOICES,
    DDP_BACKEND_CHOICES,
    LOG_FORMAT_CHOICES,
    ZERO_SHARDING_CHOICES,
    CLIP_GRAD_NORM_TYPE_CHOICES,
)


@dataclass
class MetaseqDataclass:
    """metaseq base dataclass that supported fetching attributes and metas"""

    _name: Optional[str] = None

    @staticmethod
    def name():
        return None

    def positional_args(self):
        return ["data"]

    def _get_all_attributes(self) -> List[str]:
        return [k for k in self.__dataclass_fields__.keys()]

    def _get_meta(
        self, attribute_name: str, meta: str, default: Optional[Any] = None
    ) -> Any:
        return self.__dataclass_fields__[attribute_name].metadata.get(meta, default)

    def _get_name(self, attribute_name: str) -> str:
        return self.__dataclass_fields__[attribute_name].name

    def _get_default(self, attribute_name: str) -> Any:
        if hasattr(self, attribute_name):
            if str(getattr(self, attribute_name)).startswith("${"):
                return str(getattr(self, attribute_name))
            elif str(self.__dataclass_fields__[attribute_name].default).startswith(
                "${"
            ):
                return str(self.__dataclass_fields__[attribute_name].default)
            elif (
                getattr(self, attribute_name)
                != self.__dataclass_fields__[attribute_name].default
            ):
                return getattr(self, attribute_name)

        f = self.__dataclass_fields__[attribute_name]
        if not isinstance(f.default_factory, _MISSING_TYPE):
            return f.default_factory()
        return f.default

    def _get_type(self, attribute_name: str) -> Any:
        return self.__dataclass_fields__[attribute_name].type

    def _get_help(self, attribute_name: str) -> Any:
        return self._get_meta(attribute_name, "help")

    def _get_argparse_const(self, attribute_name: str) -> Any:
        return self._get_meta(attribute_name, "argparse_const")

    def _get_argparse_alias(self, attribute_name: str) -> Any:
        return self._get_meta(attribute_name, "argparse_alias")

    def _get_choices(self, attribute_name: str) -> Any:
        return self._get_meta(attribute_name, "choices")


@dataclass
class CommonConfig(MetaseqDataclass):
    # This is the core dataclass including common parameters shared by all
    # different jobs. Please append your params to other dataclasses if they
    # were used for a particular purpose or task, such as those dedicated for
    # `distributed training`, `optimization`, etc.
    no_progress_bar: bool = field(
        default=False, metadata={"help": "disable progress bar"}
    )
    log_interval: int = field(
        default=100,
        metadata={
            "help": "log progress every N batches (when progress bar is disabled)"
        },
    )
    log_format: Optional[LOG_FORMAT_CHOICES] = field(
        default=None, metadata={"help": "log format to use"}
    )
    log_file: Optional[str] = field(
        default=None, metadata={"help": "log file to copy metrics to."}
    )
    tensorboard_logdir: Optional[str] = field(
        default=None,
        metadata={
            "help": "path to save logs for tensorboard, should match --logdir "
            "of running tensorboard (default: no tensorboard logging)"
        },
    )
    wandb_project: Optional[str] = field(
        default=None,
        metadata={"help": "Weights and Biases project name to use for logging"},
    )
    azureml_logging: Optional[bool] = field(
        default=False,
        metadata={"help": "Log scalars to AzureML context"},
    )
    seed: int = field(
        default=1, metadata={"help": "pseudo random number generator seed"}
    )
    cpu: bool = field(default=False, metadata={"help": "use CPU instead of CUDA"})
    fp16: bool = field(default=False, metadata={"help": "use FP16"})
    memory_efficient_fp16: bool = field(
        default=False,
        metadata={
            "help": "use a memory-efficient version of FP16 training; implies --fp16"
        },
    )
    fp16_no_flatten_grads: bool = field(
        default=False, metadata={"help": "don't flatten FP16 grads tensor"}
    )
    fp16_init_scale: int = field(
        default=2**7, metadata={"help": "default FP16 loss scale"}
    )
    fp16_scale_window: Optional[int] = field(
        default=None,
        metadata={"help": "number of updates before increasing loss scale"},
    )
    fp16_scale_tolerance: float = field(
        default=0.0,
        metadata={
            "help": "pct of updates that can overflow before decreasing the loss scale"
        },
    )
    min_loss_scale: float = field(
        default=1e-4,
        metadata={"help": "minimum FP16 loss scale, after which training is stopped"},
    )
    threshold_loss_scale: Optional[float] = field(
        default=None, metadata={"help": "threshold FP16 loss scale from below"}
    )
    user_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "path to a python module containing custom extensions (tasks and/or architectures)"
        },
    )
    empty_cache_freq: int = field(
        default=0,
        metadata={"help": "how often to clear the PyTorch CUDA cache (0 to disable)"},
    )
    all_gather_list_size: int = field(
        default=16384,
        metadata={"help": "number of bytes reserved for gathering stats from workers"},
    )
    model_parallel_size: int = field(
        default=1, metadata={"help": "total number of GPUs to parallelize model over"}
    )
    profile: bool = field(
        default=False, metadata={"help": "enable autograd profiler emit_nvtx"}
    )
    use_plasma_view: bool = field(
        default=False, metadata={"help": "Store indices and sizes in shared memory"}
    )
    plasma_path: Optional[str] = field(
        default="/tmp/plasma",
        metadata={
            "help": "path to run plasma_store, defaults to /tmp/plasma. Paths outside /tmp tend to fail."
        },
    )
    log_nvidia_smi: bool = field(
        default=False, metadata={"help": "log output from nvidia-smi during training"}
    )
    new_profiler: bool = field(
        default=False, metadata={"help": "use pytorch profiler (v2)"}
    )
    dont_log_param_and_grad_norm: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Don't log grad/param norms for each parameter.",
            "argparse_alias": "--quiet",
        },
    )


@dataclass
class DistributedTrainingConfig(MetaseqDataclass):
    distributed_world_size: int = field(
        default=max(1, torch.cuda.device_count()),
        metadata={
            "help": "total number of GPUs across all nodes (default: all visible GPUs)"
        },
    )
    distributed_rank: Optional[int] = field(
        default=0, metadata={"help": "rank of the current worker"}
    )
    distributed_backend: str = field(
        default="nccl", metadata={"help": "distributed backend"}
    )
    distributed_init_method: Optional[str] = field(
        default=None,
        metadata={
            "help": "typically tcp://hostname:port that will be used to "
            "establish initial connetion"
        },
    )
    distributed_port: int = field(
        default=-1,
        metadata={
            "help": "port number (not required if using --distributed-init-method)"
        },
    )
    device_id: int = field(
        default=0,
        metadata={
            "help": "which GPU to use (usually configured automatically)",
            "argparse_alias": "--local_rank",
        },
    )
    distributed_no_spawn: bool = field(
        default=False,
        metadata={
            "help": "do not spawn multiple processes even if multiple GPUs are visible"
        },
    )
    ddp_backend: DDP_BACKEND_CHOICES = field(
        default="pytorch_ddp", metadata={"help": "DistributedDataParallel backend"}
    )
    bucket_cap_mb: int = field(
        default=25, metadata={"help": "bucket size for reduction"}
    )
    fix_batches_to_gpus: bool = field(
        default=False,
        metadata={
            "help": "don't shuffle batches between GPUs; this reduces overall "
            "randomness and may affect precision but avoids the cost of re-reading the data"
        },
    )
    find_unused_parameters: bool = field(
        default=False,
        metadata={
            "help": "disable unused parameter detection (not applicable to "
            "--ddp-backend=legacy_ddp)"
        },
    )
    fast_stat_sync: bool = field(
        default=False,
        metadata={"help": "[deprecated] this is now defined per Criterion"},
    )
    heartbeat_timeout: int = field(
        default=-1,
        metadata={
            "help": "kill the job if no progress is made in N seconds; "
            "set to -1 to disable"
        },
    )
    broadcast_buffers: bool = field(
        default=False,
        metadata={
            "help": "Copy non-trainable parameters between GPUs, such as "
            "batchnorm population statistics"
        },
    )
    zero_sharding: ZERO_SHARDING_CHOICES = field(
        default="none", metadata={"help": "ZeRO sharding"}
    )
    fp16: bool = II("common.fp16")
    memory_efficient_fp16: bool = II("common.memory_efficient_fp16")
    # configuration for --ddp-backend=fully_sharded
    no_reshard_after_forward: bool = field(
        default=False,
        metadata={"help": "don't reshard parameters after forward pass"},
    )
    fp32_reduce_scatter: bool = field(
        default=False,
        metadata={"help": "reduce-scatter grads in FP32"},
    )
    cpu_offload: bool = field(
        default=False, metadata={"help": "offload FP32 params to CPU"}
    )
    use_sharded_state: Optional[bool] = field(
        default=False, metadata={"help": "load and save local state dict"}
    )
    gradient_predivide_factor: Optional[float] = field(
        default=None,
        metadata={"help": "factor to predivide gradients before reducee scatter"},
    )


@dataclass
class DatasetConfig(MetaseqDataclass):
    num_workers: int = field(
        default=1, metadata={"help": "how many subprocesses to use for data loading"}
    )
    num_workers_valid: int = field(
        default=0,
        metadata={
            "help": "how many subprocesses to use for data loading during validation"
        },
    )
    skip_invalid_size_inputs_valid_test: bool = field(
        default=False,
        metadata={"help": "ignore too long or too short lines in valid and test set"},
    )
    max_tokens: Optional[int] = field(
        default=None, metadata={"help": "maximum number of tokens in a batch"}
    )
    batch_size: Optional[int] = field(
        default=None,
        metadata={
            "help": "number of examples in a batch",
            "argparse_alias": "--max-sentences",
        },
    )
    required_batch_size_multiple: int = field(
        default=8, metadata={"help": "batch size will be a multiplier of this value"}
    )
    dataset_impl: Optional[DATASET_IMPL_CHOICES] = field(
        default=None, metadata={"help": "output dataset implementation"}
    )
    data_buffer_size: int = field(
        default=10, metadata={"help": "Number of batches to preload"}
    )
    train_subset: str = field(
        default="train",
        metadata={"help": "data subset to use for training (e.g. train, valid, test)"},
    )
    valid_subset: str = field(
        default="valid",
        metadata={
            "help": "comma separated list of data subsets to use for validation"
            " (e.g. train, valid, test)"
        },
    )
    combine_valid_subsets: Optional[bool] = field(
        default=None,
        metadata={
            "help": "comma separated list of data subsets to use for validation"
            " (e.g. train, valid, test)",
            "argparse_alias": "--combine-val",
        },
    )
    ignore_unused_valid_subsets: Optional[bool] = field(
        default=False,
        metadata={"help": "do not raise error if valid subsets are ignored"},
    )
    validate_interval: int = field(
        default=1, metadata={"help": "validate every N epochs"}
    )
    validate_interval_updates: int = field(
        default=0, metadata={"help": "validate every N updates"}
    )
    validate_after_updates: int = field(
        default=0, metadata={"help": "dont validate until reaching this many updates"}
    )
    fixed_validation_seed: Optional[int] = field(
        default=None, metadata={"help": "specified random seed for validation"}
    )
    disable_validation: bool = field(
        default=False, metadata={"help": "disable validation"}
    )
    max_tokens_valid: Optional[int] = field(
        default=II("dataset.max_tokens"),
        metadata={
            "help": "maximum number of tokens in a validation batch"
            " (defaults to --max-tokens)"
        },
    )
    batch_size_valid: Optional[int] = field(
        default=II("dataset.batch_size"),
        metadata={
            "help": "batch size of the validation batch (defaults to --batch-size)",
            "argparse_alias": "--max-sentences-valid",
        },
    )
    max_valid_steps: Optional[int] = field(
        default=None,
        metadata={"help": "How many batches to evaluate", "argparse_alias": "--nval"},
    )
    gen_subset: str = field(
        default="test",
        metadata={"help": "data subset to generate (train, valid, test)"},
    )
    num_shards: int = field(
        default=1, metadata={"help": "shard generation over N shards"}
    )
    shard_id: int = field(
        default=0, metadata={"help": "id of the shard to generate (id < num_shards)"}
    )


@dataclass
class OptimizationConfig(MetaseqDataclass):
    max_epoch: int = field(
        default=0, metadata={"help": "force stop training at specified epoch"}
    )
    max_update: int = field(
        default=0, metadata={"help": "force stop training at specified update"}
    )
    clip_norm: float = field(
        default=0.0, metadata={"help": "clip threshold of gradients"}
    )
    clip_norm_type: Optional[CLIP_GRAD_NORM_TYPE_CHOICES] = field(
        default="l2",
        metadata={"help": "either 'l2' or 'inf' to clip by l2 norm or max abs grad"},
    )
    skip_gradient_update_on_clip_norm: bool = field(
        default=False,
        metadata={
            "help": "Skip gradient update if gnorm is higher than --clip-norm value"
        },
    )
    update_freq: List[int] = field(
        default_factory=lambda: [1],
        metadata={"help": "update parameters every N_i batches, when in epoch i"},
    )
    lr: List[float] = field(
        default_factory=lambda: [0.25],
        metadata={
            "help": "learning rate for the first N epochs; all epochs >N using LR_N"
            " (note: this may be interpreted differently depending on --lr-scheduler)"
        },
    )
    train_with_epoch_remainder_batch: Optional[bool] = field(
        default=False,
        metadata={
            "help": "if set, include the last (partial) batch of each epoch in training"
            " (default is to skip it)."
        },
    )


@dataclass
class CheckpointConfig(MetaseqDataclass):
    save_dir: str = field(
        default="checkpoints", metadata={"help": "path to save checkpoints"}
    )
    restore_file: str = field(
        default="checkpoint_last.pt",
        metadata={
            "help": "filename from which to load checkpoint "
            "(default: <save-dir>/checkpoint_last.pt"
        },
    )
    finetune_from_model: Optional[str] = field(
        default=None,
        metadata={
            "help": "finetune from a pretrained model; note that meters and lr scheduler will be reset"
        },
    )
    reset_dataloader: bool = field(
        default=False,
        metadata={
            "help": "if set, does not reload dataloader state from the checkpoint"
        },
    )
    reset_lr_scheduler: bool = field(
        default=False,
        metadata={
            "help": "if set, does not load lr scheduler state from the checkpoint"
        },
    )
    reset_meters: bool = field(
        default=False,
        metadata={"help": "if set, does not load meters from the checkpoint"},
    )
    reset_optimizer: bool = field(
        default=False,
        metadata={"help": "if set, does not load optimizer state from the checkpoint"},
    )
    optimizer_overrides: str = field(
        default="{}",
        metadata={
            "help": "a dictionary used to override optimizer args when loading a checkpoint"
        },
    )
    save_interval: int = field(
        default=1, metadata={"help": "save a checkpoint every N epochs"}
    )
    save_interval_updates: int = field(
        default=0, metadata={"help": "save a checkpoint (and validate) every N updates"}
    )
    keep_interval_updates: int = field(
        default=-1,
        metadata={
            "help": "keep the last N checkpoints saved with --save-interval-updates"
        },
    )
    keep_last_epochs: int = field(
        default=-1, metadata={"help": "keep last N epoch checkpoints"}
    )
    keep_best_checkpoints: int = field(
        default=-1, metadata={"help": "keep best N checkpoints based on scores"}
    )
    no_save: bool = field(
        default=False, metadata={"help": "don't save models or checkpoints"}
    )
    no_epoch_checkpoints: bool = field(
        default=False, metadata={"help": "only store last and best checkpoints"}
    )
    no_last_checkpoints: bool = field(
        default=False, metadata={"help": "don't store last checkpoints"}
    )
    no_best_checkpoints: bool = field(
        default=False, metadata={"help": "don't store best checkpoints"}
    )
    no_save_optimizer_state: bool = field(
        default=False,
        metadata={"help": "don't save optimizer-state as part of checkpoint"},
    )
    no_save_optimizer_state_on_training_finished: bool = field(
        default=False,
        metadata={
            "help": "don't save optimizer-state as part of checkpoint when training is done"
        },
    )
    best_checkpoint_metric: str = field(
        default="loss", metadata={"help": 'metric to use for saving "best" checkpoints'}
    )
    maximize_best_checkpoint_metric: bool = field(
        default=False,
        metadata={
            "help": 'select the largest metric value for saving "best" checkpoints'
        },
    )
    patience: int = field(
        default=-1,
        metadata={
            "help": (
                "early stop training if valid performance doesn't "
                "improve for N consecutive validation runs; note "
                "that this is influenced by --validate-interval"
            )
        },
    )
    checkpoint_suffix: str = field(
        default="", metadata={"help": "suffix to add to the checkpoint file name"}
    )
    checkpoint_shard_count: int = field(
        default=1,
        metadata={
            "help": "Number of shards containing the checkpoint - "
            "if the checkpoint is over 300GB, it is preferable "
            "to split it into shards to prevent OOM on CPU while loading "
            "the checkpoint"
        },
    )
    load_checkpoint_on_all_dp_ranks: bool = field(
        default=False,
        metadata={
            "help": "load checkpoints on all data parallel devices "
            "(default: only load on rank 0 and broadcast to other devices)"
        },
    )
    write_checkpoints_asynchronously: bool = field(
        default=False,
        metadata={
            "help": (
                "Write checkpoints asynchronously in a separate "
                "thread. NOTE: This feature is currently being tested."
            ),
            "argparse_alias": "--save-async",
        },
    )
    cloud_upload_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Upload checkpoints asynchronously in a separate "
                "thread to blob store. NOTE: This feature is currently being tested."
            ),
            "argparse_alias": "--cloud-dir",
        },
    )
    # TODO(susanz): After https://github.com/fairinternal/fairseq-big-internal/issues/22 is tackled, modify this
    #  to use ComputeEnvs constant
    cluster_env: str = field(
        default="fair",
        metadata={"help": "cluster we are running on: azure/aws/fair/rsc"},
    )
    model_parallel_size: int = II("common.model_parallel_size")


@dataclass
class GenerationConfig(MetaseqDataclass):
    beam: int = field(
        default=5,
        metadata={"help": "beam size"},
    )
    nbest: int = field(
        default=1,
        metadata={"help": "number of hypotheses to output"},
    )
    max_len_a: float = field(
        default=0,
        metadata={
            "help": "generate sequences of maximum length ax + b, where x is the source length"
        },
    )
    max_len_b: int = field(
        default=200,
        metadata={
            "help": "generate sequences of maximum length ax + b, where x is the source length"
        },
    )
    min_len: int = field(
        default=1,
        metadata={"help": "minimum generation length"},
    )
    sampling: bool = field(
        default=False,
        metadata={"help": "sample hypotheses instead of using beam search"},
    )
    sampling_topk: int = field(
        default=-1,
        metadata={"help": "sample from top K likely next words instead of all words"},
    )
    sampling_topp: float = field(
        default=-1.0,
        metadata={
            "help": "sample from the smallest set whose cumulative probability mass exceeds p for next words"
        },
    )
    temperature: float = field(
        default=1.0,
        metadata={"help": "temperature for generation"},
    )
    no_seed_provided: bool = field(
        default=False,
        metadata={"help": "if set, dont use seed for initializing random generators"},
    )
    # former interactive args
    buffer_size: int = field(
        default=0,
        metadata={
            "help": "read this many sentences into a buffer before processing them"
        },
    )
    input: str = field(
        default="-",
        metadata={"help": "file to read from; use - for stdin"},
    )


@dataclass
class CommonEvalConfig(MetaseqDataclass):
    path: Optional[str] = field(
        default=None,
        metadata={"help": "path(s) to model file(s), colon separated"},
    )
    quiet: bool = field(default=False, metadata={"help": "only print final scores"})
    model_overrides: str = field(
        default="{}",
        metadata={
            "help": "a dictionary used to override model args at generation that were used during model training"
        },
    )
    results_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "path to save eval results (optional)",
            "argparse_alias": "--sp",
        },
    )


@dataclass
class ReshardConfig(MetaseqDataclass):
    save_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "where to save the resharded checkpoints",
            "argparse_alias": "--dest-dir",
        },
    )
    save_prefix: Optional[str] = field(
        default="reshard", metadata={"help": "save to dest-dir/save-prefix-shard{i}.pt"}
    )
    target_world_size: Optional[int] = field(
        default=128,
        metadata={
            "help": (
                "The maximum number of GPUs you want to use to evaluate. "
                "AssertionError if any FSDP module's number of parameters is not "
                "divisible by this."
            )
        },
    )
    do_pad: Optional[bool] = field(
        default=False,
        metadata={
            "help": (
                "Add padding to make sure that running on target world size "
                "works. This reduces flexibility for world sizes smaller than "
                "target world size."
            )
        },
    )


@dataclass
class EvalLMConfig(MetaseqDataclass):
    # TODO(anj): Remove this since we want to set this by default when running eval.
    score_sequences: bool = field(
        default=False,
        metadata={"help": "if set, uses the ScoreSequencer class for evaluating."},
    )
    output_word_probs: bool = field(
        default=False,
        metadata={
            "help": "if set, outputs words and their predicted log probabilities to standard output"
        },
    )
    output_word_stats: bool = field(
        default=False,
        metadata={
            "help": "if set, outputs word statistics such as word count, average probability, etc"
        },
    )
    context_window: int = field(
        default=0,
        metadata={
            "help": "ensures that every evaluated token has access to a context of at least this size, if possible"
        },
    )
    softmax_batch: int = field(
        default=sys.maxsize,
        metadata={
            "help": (
                "if BxT is more than this, will batch the softmax over vocab to "
                "this amount of tokens, in order to fit into GPU memory"
            )
        },
    )
    max_valid_steps: Optional[int] = field(
        default=None,
        metadata={"help": "How many batches to evaluate", "argparse_alias": "--nval"},
    )


@dataclass
class MetaseqConfig(MetaseqDataclass):
    common: CommonConfig = CommonConfig()
    common_eval: CommonEvalConfig = CommonEvalConfig()
    distributed_training: DistributedTrainingConfig = DistributedTrainingConfig()
    dataset: DatasetConfig = DatasetConfig()
    optimization: OptimizationConfig = OptimizationConfig()
    checkpoint: CheckpointConfig = CheckpointConfig()
    generation: GenerationConfig = GenerationConfig()
    eval_lm: EvalLMConfig = EvalLMConfig()
    reshard: ReshardConfig = ReshardConfig()
    model: Any = MISSING
    task: Any = MISSING
    criterion: Any = MISSING
    optimizer: Any = MISSING
    lr_scheduler: Any = MISSING
    bpe: Any = MISSING
    tokenizer: Any = None
