# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import warnings
from argparse import Namespace
from typing import Any, Callable, Dict, List

import torch
from omegaconf import DictConfig

from metaseq import metrics
from metaseq.data import Dictionary, BaseDataset, data_utils, encoders, iterators
from metaseq.dataclass import MetaseqDataclass
from metaseq.dataclass.utils import gen_parser_from_dataclass
from metaseq.utils import tokenize_line

logger = logging.getLogger(__name__)


class StatefulContainer(object):

    _state: Dict[str, Any] = dict()
    _factories: Dict[str, Callable[[], Any]] = dict()

    def add_factory(self, name, factory: Callable[[], Any]):
        self._factories[name] = factory

    def merge_state_dict(self, state_dict: Dict[str, Any]):
        self._state.update(state_dict)

    @property
    def state_dict(self) -> Dict[str, Any]:
        return self._state

    def __getattr__(self, name):
        if name not in self._state and name in self._factories:
            self._state[name] = self._factories[name]()

        if name in self._state:
            return self._state[name]

        raise AttributeError(f"Task state has no factory for attribute {name}")


class BaseTask(object):
    """
    Tasks store dictionaries and provide helpers for loading/iterating over
    Datasets, initializing the Model/Criterion and calculating the loss.

    Tasks have limited statefulness. In particular, state that needs to be
    saved to/loaded from checkpoints needs to be stored in the `self.state`
    :class:`StatefulContainer` object. For example::

        self.state.add_factory("dictionary", self.load_dictionary)
        print(self.state.dictionary)  # calls self.load_dictionary()

    This is necessary so that when loading checkpoints, we can properly
    recreate the task state after initializing the task instance.
    """

    @classmethod
    def add_args(cls, parser):
        """Add task-specific arguments to the parser."""
        dc = getattr(cls, "__dataclass", None)
        if dc is not None:
            gen_parser_from_dataclass(parser, dc())

    @staticmethod
    def logging_outputs_can_be_summed(criterion) -> bool:
        """
        Whether the logging outputs returned by `train_step` and `valid_step` can
        be summed across workers prior to calling `reduce_metrics`.
        Setting this to True will improve distributed training speed.
        """
        return criterion.logging_outputs_can_be_summed()

    cfg: MetaseqDataclass
    datasets: Dict[str, BaseDataset]
    dataset_to_epoch_iter: Dict[BaseDataset, Any]
    state: StatefulContainer = None

    def __init__(self, cfg: MetaseqDataclass, **kwargs):
        self.cfg = cfg
        self.datasets = dict()
        self.dataset_to_epoch_iter = dict()
        self.state = StatefulContainer()

    @classmethod
    def load_dictionary(cls, filename):
        """Load the dictionary from the filename

        Args:
            filename (str): the filename
        """
        return Dictionary.load(filename)

    @classmethod
    def build_dictionary(
        cls, filenames, workers=1, threshold=-1, nwords=-1, padding_factor=8
    ):
        """Build the dictionary

        Args:
            filenames (list): list of filenames
            workers (int): number of concurrent workers
            threshold (int): defines the minimum word count
            nwords (int): defines the total number of words in the final dictionary,
                including special symbols
            padding_factor (int): can be used to pad the dictionary size to be a
                multiple of 8, which is important on some hardware (e.g., Nvidia
                Tensor Cores).
        """
        d = Dictionary()
        for filename in filenames:
            Dictionary.add_file_to_dictionary(filename, d, tokenize_line, workers)
        d.finalize(threshold=threshold, nwords=nwords, padding_factor=padding_factor)
        return d

    @classmethod
    def setup_task(cls, cfg: DictConfig, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            cfg (omegaconf.DictConfig): parsed command-line arguments
        """
        return cls(cfg, **kwargs)

    def load_dataset(
        self,
        split: str,
        combine: bool = False,
        task_cfg: MetaseqDataclass = None,
        **kwargs,
    ):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
            combine (bool): combines a split segmented into pieces into one dataset
            task_cfg (MetaseqDataclass): optional task configuration stored in the checkpoint that can be used
                                         to load datasets
        """
        raise NotImplementedError

    def dataset(self, split):
        """
        Return a loaded dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)

        Returns:
            a :class:`~metaseq.data.BaseDataset` corresponding to *split*
        """
        from metaseq.data import BaseDataset

        if split not in self.datasets:
            raise KeyError("Dataset not loaded: " + split)
        if not isinstance(self.datasets[split], BaseDataset):
            raise TypeError("Datasets are expected to be of type BaseDataset")
        return self.datasets[split]

    def filter_indices_by_size(
        self, indices, dataset, max_positions=None, ignore_invalid_inputs=False
    ):
        """
        Filter examples that are too large

        Args:
            indices (np.array): original array of sample indices
            dataset (~metaseq.data.BaseDataset): dataset to batch
            max_positions (optional): max sentence length supported by the
                model (default: None).
            ignore_invalid_inputs (bool, optional): don't raise Exception for
                sentences that are too long (default: False).
        Returns:
            np.array: array of filtered sample indices
        """
        indices, ignored = dataset.filter_indices_by_size(indices, max_positions)
        if len(ignored) > 0:
            if not ignore_invalid_inputs:
                raise Exception(
                    (
                        "Size of sample #{} is invalid (={}) since max_positions={}, "
                        "skip this example with --skip-invalid-size-inputs-valid-test"
                    ).format(ignored[0], dataset.size(ignored[0]), max_positions)
                )
            logger.warning(
                (
                    "{:,} samples have invalid sizes and will be skipped, "
                    "max_positions={}, first few sample ids={}"
                ).format(len(ignored), max_positions, ignored[:10])
            )
        return indices

    def get_batch_iterator(
        self,
        dataset: BaseDataset,
        max_tokens=None,
        max_sentences=None,
        max_positions=None,
        ignore_invalid_inputs=False,
        required_batch_size_multiple=1,
        seed=1,
        num_shards=1,
        shard_id=0,
        num_workers=0,
        epoch=1,
        data_buffer_size=0,
        disable_iterator_cache=False,
        batch_by_size=True,
        skip_remainder_batch=True,
    ):
        """
        Get an iterator that yields batches of data from the given dataset.

        Args:
            dataset (~metaseq.data.BaseDataset): dataset to batch
            max_tokens (int, optional): max number of tokens in each batch
                (default: None).
            max_sentences (int, optional): max number of sentences in each
                batch (default: None).
            max_positions (optional): max sentence length supported by the
                model (default: None).
            ignore_invalid_inputs (bool, optional): don't raise Exception for
                sentences that are too long (default: False).
            required_batch_size_multiple (int, optional): require batch size to
                be a multiple of N (default: 1).
            seed (int, optional): seed for random number generator for
                reproducibility (default: 1).
            num_shards (int, optional): shard the data iterator into N
                shards (default: 1).
            shard_id (int, optional): which shard of the data iterator to
                return (default: 0).
            num_workers (int, optional): how many subprocesses to use for data
                loading. 0 means the data will be loaded in the main process
                (default: 0).
            epoch (int, optional): the epoch to start the iterator from
                (default: 1).
            data_buffer_size (int, optional): number of batches to
                preload (default: 0).
            disable_iterator_cache (bool, optional): don't cache the
                EpochBatchIterator
                (default: False).
            batch_by_size (bool, optional):
                batch sequences of similar length together to reduce padding.
                If false, each batch will be of size max_sentences.
            skip_remainder_batch (bool, optional): if set, discard the last
                batch in each training epoch, as the last batch is often smaller than
                    local_batch_size * distributed_word_size (default: ``True``).
        Returns:
            ~metaseq.iterators.EpochBatchIterator: a batched iterator over the
                given dataset split
        """
        if not disable_iterator_cache and dataset in self.dataset_to_epoch_iter:
            logger.debug("reusing EpochBatchIterator for epoch {}".format(epoch))
            return self.dataset_to_epoch_iter[dataset]

        assert isinstance(dataset, BaseDataset)

        # initialize the dataset with the correct starting epoch
        dataset.set_epoch(epoch)

        # get indices ordered by example size
        with data_utils.numpy_seed(seed):
            indices = dataset.ordered_indices()

        # filter examples that are too large
        if max_positions is not None:
            indices = self.filter_indices_by_size(
                indices, dataset, max_positions, ignore_invalid_inputs
            )

        if batch_by_size:
            # create mini-batches with given size constraints
            batch_sampler = dataset.batch_by_size(
                indices,
                max_tokens=max_tokens,
                max_sentences=max_sentences,
                required_batch_size_multiple=required_batch_size_multiple,
            )
        else:
            assert (
                max_sentences is not None
            ), "If batch_by_size=False, max_sentences must be passed. Got None"
            starts = indices[::max_sentences]
            batch_sampler = [indices[s : s + max_sentences] for s in starts]
        # return a reusable, sharded iterator
        epoch_iter = iterators.EpochBatchIterator(
            dataset=dataset,
            collate_fn=dataset.collater,
            batch_sampler=batch_sampler,
            seed=seed,
            num_shards=num_shards,
            shard_id=shard_id,
            num_workers=num_workers,
            epoch=epoch,
            buffer_size=data_buffer_size,
            skip_remainder_batch=skip_remainder_batch,
        )
        self.dataset_to_epoch_iter[dataset] = epoch_iter
        return epoch_iter

    def build_model(self, cfg: MetaseqDataclass):
        """
        Build the :class:`~metaseq.models.BaseModel` instance for this
        task.

        Args:
            cfg (MetaseqDataclass): configuration object

        Returns:
            a :class:`~metaseq.models.BaseModel` instance
        """
        from metaseq import models

        model = models.build_model(cfg, self)
        return model

    def build_criterion(self, cfg: DictConfig):
        """
        Build the :class:`~metaseq.criterions.BaseCriterion` instance for
        this task.

        Args:
            cfg (omegaconf.DictConfig): configration object

        Returns:
            a :class:`~metaseq.criterions.BaseCriterion` instance
        """
        from metaseq import criterions

        return criterions.build_criterion(cfg, self)

    def build_generator(
        self, models, args, seq_gen_cls=None, extra_gen_cls_kwargs=None
    ):
        from metaseq.sequence_generator import SequenceGenerator

        # Choose search strategy.
        sampling = getattr(args, "sampling", False)
        sampling_topp = getattr(args, "sampling_topp", -1.0)
        assert sampling_topp < 0 or sampling, "--sampling-topp requires --sampling"

        if getattr(args, "sampling_topk", False):
            logger.warning(
                "sampling with topk is not supported, ignoring the sampling_topk argument"
            )

        extra_gen_cls_kwargs = extra_gen_cls_kwargs or {}
        if seq_gen_cls is None:
            seq_gen_cls = SequenceGenerator

        return seq_gen_cls(
            models,
            self.target_dictionary,
            beam_size=getattr(args, "beam", 5),
            max_len_a=getattr(args, "max_len_a", 0),
            max_len_b=getattr(args, "max_len_b", 200),
            min_len=getattr(args, "min_len", 1),
            temperature=getattr(args, "temperature", 1.0),
            topp=sampling_topp,
            **extra_gen_cls_kwargs,
        )

    def train_step(
        self, sample, model, criterion, optimizer, update_num, ignore_grad=False
    ):
        """
        Do forward and backward, and return the loss as computed by *criterion*
        for the given *model* and *sample*.

        Args:
            sample (dict): the mini-batch. The format is defined by the
                :class:`~metaseq.data.BaseDataset`.
            model (~metaseq.models.BaseModel): the model
            criterion (~metaseq.criterions.BaseCriterion): the criterion
            optimizer (~metaseq.optim.BaseOptimizer): the optimizer
            update_num (int): the current update
            ignore_grad (bool): multiply loss by 0 if this is set to True

        Returns:
            tuple:
                - the loss
                - the sample size, which is used as the denominator for the
                  gradient
                - logging outputs to display while training
        """
        model.train()
        model.set_num_updates(update_num)
        if update_num == 0:
            logger.info(
                "Starting first forward pass and waiting for dataloader in other ranks"
            )
        # forward
        loss, sample_size, logging_output = criterion(model, sample)
        if ignore_grad:
            loss *= 0
        if update_num == 0:
            logger.info("Starting backward pass")
        # backward
        optimizer.backward(loss)
        if update_num == 0:
            logger.info("Finished first backward pass")
        return loss, sample_size, logging_output

    def valid_step(self, sample, model, criterion):
        model.eval()
        with torch.no_grad():
            loss, sample_size, logging_output = criterion(model, sample)
        return loss, sample_size, logging_output

    def optimizer_step(self, optimizer, model, update_num):
        optimizer.step()

    def build_dataset_for_inference(
        self, src_tokens: List[torch.Tensor], src_lengths: List[int], **kwargs
    ) -> torch.utils.data.Dataset:
        raise NotImplementedError

    def inference_step(self, generator, models, sample, prefix_tokens=None):
        with torch.no_grad():
            return generator.generate(models, sample, prefix_tokens=prefix_tokens)

    def begin_epoch(self, epoch, model):
        """Hook function called before the start of each epoch."""
        pass

    def begin_valid_epoch(self, epoch, model):
        """Hook function called before the start of each validation epoch."""
        pass

    def reduce_metrics(self, logging_outputs, criterion):
        """Aggregate logging outputs from data parallel training."""
        if not any("ntokens" in log for log in logging_outputs):
            warnings.warn(
                "ntokens not found in Criterion logging outputs, cannot log wpb or wps"
            )
        else:
            ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
            metrics.log_scalar("wpb", ntokens, priority=180, round=1)
            metrics.log_speed("wps", ntokens, priority=90, round=1)

        if not any("nsentences" in log for log in logging_outputs):
            warnings.warn(
                "nsentences not found in Criterion logging outputs, cannot log bsz"
            )
        else:
            nsentences = sum(log.get("nsentences", 0) for log in logging_outputs)
            metrics.log_scalar("bsz", nsentences, priority=190, round=1)

        criterion.__class__.reduce_metrics(logging_outputs)

    def state_dict(self):
        if self.state is not None:
            return self.state.state_dict
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        if self.state is not None:
            self.state.merge_state_dict(state_dict)

    def max_positions(self):
        """Return the max input length allowed by the task."""
        return None

    @property
    def source_dictionary(self):
        """Return the source :class:`~metaseq.data.Dictionary` (if applicable
        for this task)."""
        raise NotImplementedError

    @property
    def target_dictionary(self):
        """Return the target :class:`~metaseq.data.Dictionary` (if applicable
        for this task)."""
        raise NotImplementedError

    def build_tokenizer(self, args):
        """Build the pre-tokenizer for this task."""
        return encoders.build_tokenizer(args)

    def build_bpe(self, args):
        """Build the tokenizer for this task."""
        return encoders.build_bpe(args)

    def get_interactive_tokens_and_lengths(self, lines, encode_fn):
        tokens = [
            self.source_dictionary.encode_line(
                encode_fn(src_str), add_if_not_exist=False
            ).long()
            for src_str in lines
        ]
        lengths = [t.numel() for t in tokens]
        return tokens, lengths


class LegacyTask(BaseTask):
    def __init__(self, args: Namespace):
        self.args = args
        self.datasets = {}
        self.dataset_to_epoch_iter = {}

    @classmethod
    def setup_task(cls, args: Namespace, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            args (argparse.Namespace): parsed command-line arguments
        """
        return cls(args, **kwargs)

    def build_model(self, args: Namespace):
        """
        Build the :class:`~metaseq.models.BaseModel` instance for this
        task.

        Args:
            args (argparse.Namespace): parsed command-line arguments

        Returns:
            a :class:`~metaseq.models.BaseModel` instance
        """
        from metaseq import models

        model = models.build_model(args, self)
        return model

    def build_criterion(self, args: Namespace):
        """
        Build the :class:`~metaseq.criterions.BaseCriterion` instance for
        this task.

        Args:
            args (argparse.Namespace): parsed command-line arguments

        Returns:
            a :class:`~metaseq.criterions.BaseCriterion` instance
        """
        from metaseq import criterions

        return criterions.build_criterion(args, self)
