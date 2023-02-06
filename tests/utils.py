# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import random
from typing import Optional

import numpy as np
import torch

import metaseq.distributed.utils as distributed_utils
from metaseq import options
from metaseq.data import Dictionary, data_utils
from metaseq.dataclass.utils import convert_namespace_to_omegaconf
from metaseq.models import (
    BaseModel,
    BaseDecoder,
)
from metaseq.tasks import LegacyTask
from metaseq.cli import validate, train


def collate(
    samples,
    pad_idx,
    eos_idx,
    left_pad_source=True,
    left_pad_target=False,
    input_feeding=True,
    pad_to_length=None,
    pad_to_multiple=1,
):
    if len(samples) == 0:
        return {}

    def merge(key, left_pad, move_eos_to_beginning=False, pad_to_length=None):
        return data_utils.collate_tokens(
            [s[key] for s in samples],
            pad_idx,
            eos_idx,
            left_pad,
            move_eos_to_beginning,
            pad_to_length=pad_to_length,
            pad_to_multiple=pad_to_multiple,
        )

    def check_alignment(alignment, src_len, tgt_len):
        if alignment is None or len(alignment) == 0:
            return False
        if (
            alignment[:, 0].max().item() >= src_len - 1
            or alignment[:, 1].max().item() >= tgt_len - 1
        ):
            return False
        return True

    def compute_alignment_weights(alignments):
        """
        Given a tensor of shape [:, 2] containing the source-target indices
        corresponding to the alignments, a weight vector containing the
        inverse frequency of each target index is computed.
        For e.g. if alignments = [[5, 7], [2, 3], [1, 3], [4, 2]], then
        a tensor containing [1., 0.5, 0.5, 1] should be returned (since target
        index 3 is repeated twice)
        """
        align_tgt = alignments[:, 1]
        _, align_tgt_i, align_tgt_c = torch.unique(
            align_tgt, return_inverse=True, return_counts=True
        )
        align_weights = align_tgt_c[align_tgt_i[np.arange(len(align_tgt))]]
        return 1.0 / align_weights.float()

    id = torch.LongTensor([s["id"] for s in samples])
    src_tokens = merge(
        "source",
        left_pad=left_pad_source,
        pad_to_length=pad_to_length["source"] if pad_to_length is not None else None,
    )
    # sort by descending source length
    src_lengths = torch.LongTensor(
        [s["source"].ne(pad_idx).long().sum() for s in samples]
    )
    src_lengths, sort_order = src_lengths.sort(descending=True)
    id = id.index_select(0, sort_order)
    src_tokens = src_tokens.index_select(0, sort_order)

    prev_output_tokens = None
    target = None
    if samples[0].get("target", None) is not None:
        target = merge(
            "target",
            left_pad=left_pad_target,
            pad_to_length=pad_to_length["target"]
            if pad_to_length is not None
            else None,
        )
        target = target.index_select(0, sort_order)
        tgt_lengths = torch.LongTensor(
            [s["target"].ne(pad_idx).long().sum() for s in samples]
        ).index_select(0, sort_order)
        ntokens = tgt_lengths.sum().item()

        if samples[0].get("prev_output_tokens", None) is not None:
            prev_output_tokens = merge(
                "prev_output_tokens",
                left_pad=left_pad_target,
                pad_to_length=pad_to_length["target"]
                if pad_to_length is not None
                else None,
            )
        elif input_feeding:
            # we create a shifted version of targets for feeding the
            # previous output token(s) into the next decoder step
            prev_output_tokens = merge(
                "target",
                left_pad=left_pad_target,
                move_eos_to_beginning=True,
                pad_to_length=pad_to_length["target"]
                if pad_to_length is not None
                else None,
            )
    else:
        ntokens = src_lengths.sum().item()

    batch = {
        "id": id,
        "nsentences": len(samples),
        "ntokens": ntokens,
        "net_input": {
            "src_tokens": src_tokens,
            "src_lengths": src_lengths,
        },
        "target": target,
    }
    if prev_output_tokens is not None:
        batch["net_input"]["prev_output_tokens"] = prev_output_tokens.index_select(
            0, sort_order
        )

    if samples[0].get("alignment", None) is not None:
        bsz, tgt_sz = batch["target"].shape
        src_sz = batch["net_input"]["src_tokens"].shape[1]

        offsets = torch.zeros((len(sort_order), 2), dtype=torch.long)
        offsets[:, 1] += torch.arange(len(sort_order), dtype=torch.long) * tgt_sz
        if left_pad_source:
            offsets[:, 0] += src_sz - src_lengths
        if left_pad_target:
            offsets[:, 1] += tgt_sz - tgt_lengths

        alignments = [
            alignment + offset
            for align_idx, offset, src_len, tgt_len in zip(
                sort_order, offsets, src_lengths, tgt_lengths
            )
            for alignment in [samples[align_idx]["alignment"].view(-1, 2)]
            if check_alignment(alignment, src_len, tgt_len)
        ]

        if len(alignments) > 0:
            alignments = torch.cat(alignments, dim=0)
            align_weights = compute_alignment_weights(alignments)

            batch["alignments"] = alignments
            batch["align_weights"] = align_weights

    if samples[0].get("constraints", None) is not None:
        # Collate the packed constraints across the samples, padding to
        # the length of the longest sample.
        lens = [sample.get("constraints").size(0) for sample in samples]
        constraints = torch.zeros((len(samples), max(lens))).long()
        for i, sample in enumerate(samples):
            constraints[i, 0 : lens[i]] = samples[i].get("constraints")
        batch["constraints"] = constraints.index_select(0, sort_order)

    return batch


def dummy_dictionary(vocab_size, prefix="token_"):
    d = Dictionary()
    for i in range(vocab_size):
        token = prefix + str(i)
        d.add_symbol(token)
    d.finalize(padding_factor=1)  # don't add extra padding symbols
    return d


def dummy_dataloader(
    samples,
    padding_idx=1,
    eos_idx=2,
    batch_size=None,
):
    if batch_size is None:
        batch_size = len(samples)

    # add any missing data to samples
    for i, sample in enumerate(samples):
        if "id" not in sample:
            sample["id"] = i

    # create dataloader
    dataset = TestDataset(samples)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=(lambda samples: collate(samples, padding_idx, eos_idx)),
    )
    return iter(dataloader)


def sequence_generator_setup():
    # construct dummy dictionary
    d = dummy_dictionary(vocab_size=2)

    eos = d.eos()
    w1 = 4
    w2 = 5

    # construct source data
    src_tokens = torch.LongTensor([[w1, w2, eos], [w1, w2, eos]])
    src_lengths = torch.LongTensor([2, 2])

    args = argparse.Namespace()
    unk = 0.0
    args.beam_probs = [
        # step 0:
        torch.FloatTensor(
            [
                # eos      w1   w2
                # sentence 1:
                [0.0, unk, 0.9, 0.1],  # beam 1
                [0.0, unk, 0.9, 0.1],  # beam 2
                # sentence 2:
                [0.0, unk, 0.7, 0.3],
                [0.0, unk, 0.7, 0.3],
            ]
        ),
        # step 1:
        torch.FloatTensor(
            [
                # eos      w1   w2       prefix
                # sentence 1:
                [1.0, unk, 0.0, 0.0],  # w1: 0.9  (emit: w1 <eos>: 0.9*1.0)
                [0.0, unk, 0.9, 0.1],  # w2: 0.1
                # sentence 2:
                [0.25, unk, 0.35, 0.4],  # w1: 0.7  (don't emit: w1 <eos>: 0.7*0.25)
                [0.00, unk, 0.10, 0.9],  # w2: 0.3
            ]
        ),
        # step 2:
        torch.FloatTensor(
            [
                # eos      w1   w2       prefix
                # sentence 1:
                [0.0, unk, 0.1, 0.9],  # w2 w1: 0.1*0.9
                [
                    0.6,
                    unk,
                    0.2,
                    0.2,
                ],  # w2 w2: 0.1*0.1  (emit: w2 w2 <eos>: 0.1*0.1*0.6)
                # sentence 2:
                [
                    0.60,
                    unk,
                    0.4,
                    0.00,
                ],  # w1 w2: 0.7*0.4  (emit: w1 w2 <eos>: 0.7*0.4*0.6)
                [0.01, unk, 0.0, 0.99],  # w2 w2: 0.3*0.9
            ]
        ),
        # step 3:
        torch.FloatTensor(
            [
                # eos      w1   w2       prefix
                # sentence 1:
                [
                    1.0,
                    unk,
                    0.0,
                    0.0,
                ],  # w2 w1 w2: 0.1*0.9*0.9  (emit: w2 w1 w2 <eos>: 0.1*0.9*0.9*1.0)
                [
                    1.0,
                    unk,
                    0.0,
                    0.0,
                ],  # w2 w1 w1: 0.1*0.9*0.1  (emit: w2 w1 w1 <eos>: 0.1*0.9*0.1*1.0)
                # sentence 2:
                [
                    0.1,
                    unk,
                    0.5,
                    0.4,
                ],  # w2 w2 w2: 0.3*0.9*0.99  (emit: w2 w2 w2 <eos>: 0.3*0.9*0.99*0.1)
                [
                    1.0,
                    unk,
                    0.0,
                    0.0,
                ],  # w1 w2 w1: 0.7*0.4*0.4  (emit: w1 w2 w1 <eos>: 0.7*0.4*0.4*1.0)
            ]
        ),
    ]

    task = TestTranslationTask.setup_task(args, d, d)
    model = task.build_model(args)
    tgt_dict = task.target_dictionary

    return tgt_dict, w1, w2, src_tokens, src_lengths, model


def create_dummy_data(
    data_dir, num_examples=100, maxlen=20, alignment=False, languages=None
):
    def _create_dummy_data(dir, filename):
        data = torch.rand(num_examples * maxlen)
        data = 97 + torch.floor(26 * data).int()
        with open(os.path.join(dir, filename), "w") as h:
            offset = 0
            for _ in range(num_examples):
                ex_len = random.randint(1, maxlen)
                ex_str = " ".join(map(chr, data[offset : offset + ex_len]))
                print(ex_str, file=h)
                offset += ex_len

    def _create_dummy_alignment_data(filename_src, filename_tgt, filename):
        with open(os.path.join(data_dir, filename_src), "r") as src_f, open(
            os.path.join(data_dir, filename_tgt), "r"
        ) as tgt_f, open(os.path.join(data_dir, filename), "w") as h:
            for src, tgt in zip(src_f, tgt_f):
                src_len = len(src.split())
                tgt_len = len(tgt.split())
                avg_len = (src_len + tgt_len) // 2
                num_alignments = random.randint(avg_len // 2, 2 * avg_len)
                src_indices = torch.floor(torch.rand(num_alignments) * src_len).int()
                tgt_indices = torch.floor(torch.rand(num_alignments) * tgt_len).int()
                ex_str = " ".join(
                    [
                        "{}-{}".format(src, tgt)
                        for src, tgt in zip(src_indices, tgt_indices)
                    ]
                )
                print(ex_str, file=h)

    files_to_write = [
        "train.in",
        "train.out",
        "valid.in",
        "valid.out",
        "test.in",
        "test.out",
    ]
    if languages is None:  # En only dummy dataset
        for f in files_to_write:
            _create_dummy_data(data_dir, f)
    else:
        for lang in languages:
            lang_dir = os.path.join(data_dir, lang)
            os.makedirs(lang_dir, exist_ok=True)
            for f in files_to_write:
                _create_dummy_data(lang_dir, f)

    if alignment:
        _create_dummy_alignment_data("train.in", "train.out", "train.align")
        _create_dummy_alignment_data("valid.in", "valid.out", "valid.align")
        _create_dummy_alignment_data("test.in", "test.out", "test.align")


class TestDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        super().__init__()
        self.data = data
        self.sizes = None

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


class TestTranslationTask(LegacyTask):
    def __init__(self, args, src_dict, tgt_dict, model):
        super().__init__(args)
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict
        self.model = model

    @classmethod
    def setup_task(cls, args, src_dict=None, tgt_dict=None, model=None):
        return cls(args, src_dict, tgt_dict, model)

    def build_model(self, args):
        return TestModel.build_model(args, self)

    @property
    def source_dictionary(self):
        return self.src_dict

    @property
    def target_dictionary(self):
        return self.tgt_dict


class TestModel(BaseModel):
    def __init__(self, decoder):
        super().__init__(decoder)

    @classmethod
    def build_model(cls, args, task):
        decoder = TestBaseDecoder(args, task.target_dictionary)
        return cls(decoder)


class TestBaseDecoder(BaseDecoder):
    def __init__(self, args, dictionary):
        super().__init__(dictionary)
        assert hasattr(args, "beam_probs") or hasattr(args, "probs")
        args.max_decoder_positions = getattr(args, "max_decoder_positions", 100)
        self.args = args

    def forward(self, src_tokens, src_lengths, prev_output_tokens):
        bbsz = prev_output_tokens.size(0)
        vocab = len(self.dictionary)
        src_len = src_tokens.size(1)
        tgt_len = prev_output_tokens.size(1)

        steps = list(range(tgt_len))

        # define output in terms of raw probs
        if hasattr(self.args, "probs"):
            assert (
                self.args.probs.dim() == 3
            ), "expected probs to have size bsz*steps*vocab"
            probs = self.args.probs.index_select(1, torch.LongTensor(steps))
        else:
            probs = torch.FloatTensor(bbsz, len(steps), vocab).zero_()
            for i, step in enumerate(steps):
                # args.beam_probs gives the probability for every vocab element,
                # starting with eos, then unknown, and then the rest of the vocab
                if step < len(self.args.beam_probs):
                    probs[:, i, self.dictionary.eos() :] = self.args.beam_probs[step]
                else:
                    probs[:, i, self.dictionary.eos()] = 1.0

        # random attention
        attn = torch.rand(bbsz, tgt_len, src_len)

        dev = prev_output_tokens.device
        return probs.to(dev), {"attn": [attn.to(dev)]}

    def get_normalized_probs(self, logits, log_probs):
        # the decoder returns probabilities directly
        if log_probs:
            return logits.log()
        else:
            return logits

    def max_positions(self):
        return self.args.max_decoder_positions


def train_language_model(
    data_dir,
    arch,
    extra_flags=None,
    run_validation=False,
    extra_valid_flags=None,
    task="language_modeling",
    world_size=1,
    max_tokens: Optional[int] = 500,
):
    train_parser = options.get_training_parser()
    train_args = options.parse_args_and_arch(
        train_parser,
        [
            "--task",
            task,
            data_dir,
            "--arch",
            arch,
            "--optimizer",
            "adam",
            "--lr",
            "0.0001",
            "--tokens-per-sample",
            "500",
            "--save-dir",
            data_dir,
            "--max-epoch",
            "1",
            "--distributed-world-size",
            str(world_size),
            "--ddp-backend",
            "no_c10d",
            "--num-workers",
            "0",
        ]
        + (["--max-tokens", str(max_tokens)] if max_tokens is not None else [])
        + (extra_flags or []),
    )
    cfg = convert_namespace_to_omegaconf(train_args)
    distributed_utils.call_main(cfg, train.main)

    if run_validation:
        # test validation
        validate_parser = options.get_validation_parser()
        validate_args = options.parse_args_and_arch(
            validate_parser,
            [
                "--task",
                task,
                data_dir,
                "--path",
                os.path.join(data_dir, "checkpoint_last.pt"),
                "--valid-subset",
                "valid",
                "--max-tokens",
                "500",
                "--num-workers",
                "0",
            ]
            + (extra_valid_flags or []),
        )
        validate.main(validate_args)
