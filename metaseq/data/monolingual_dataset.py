# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch

from . import BaseDataset, data_utils


def collate(samples, pad_idx, eos_idx, fixed_pad_length=None, pad_to_bsz=None):
    if len(samples) == 0:
        return {}

    def merge(key, is_list=False):
        if is_list:
            res = []
            for i in range(len(samples[0][key])):
                res.append(
                    data_utils.collate_tokens(
                        [s[key][i] for s in samples],
                        pad_idx,
                        eos_idx,
                        left_pad=False,
                        pad_to_length=fixed_pad_length,
                        pad_to_bsz=pad_to_bsz,
                    )
                )
            return res
        else:
            return data_utils.collate_tokens(
                [s[key] for s in samples],
                pad_idx,
                eos_idx,
                left_pad=False,
                pad_to_length=fixed_pad_length,
                pad_to_bsz=pad_to_bsz,
            )

    src_tokens = merge("source")
    if samples[0]["target"] is not None:
        is_target_list = isinstance(samples[0]["target"], list)
        target = merge("target", is_target_list)
    else:
        target = src_tokens

    return {
        "id": torch.LongTensor([s["id"] for s in samples]),
        "nsentences": len(samples),
        "ntokens": sum(len(s["source"]) for s in samples),
        "net_input": {
            "src_tokens": src_tokens,
            "src_lengths": torch.LongTensor([s["source"].numel() for s in samples]),
        },
        "target": target,
    }


class MonolingualDataset(BaseDataset):
    """
    A wrapper around torch.utils.data.Dataset for monolingual data.
    """

    def __init__(
        self,
        dataset,
        sizes,
        src_vocab,
        tgt_vocab=None,
        add_eos_for_other_targets=False,
        shuffle=False,
        add_bos_token=False,
        fixed_pad_length=None,
        pad_to_bsz=None,
        src_lang_idx=None,
        tgt_lang_idx=None,
    ):
        self.dataset = dataset
        self.sizes = np.array(sizes)
        self.vocab = src_vocab
        self.tgt_vocab = tgt_vocab or src_vocab
        self.add_eos_for_other_targets = add_eos_for_other_targets
        self.shuffle = shuffle
        self.add_bos_token = add_bos_token
        self.fixed_pad_length = fixed_pad_length
        self.pad_to_bsz = pad_to_bsz
        self.src_lang_idx = src_lang_idx
        self.tgt_lang_idx = tgt_lang_idx

    def __getitem__(self, index):
        # *future_target* is the original sentence
        # *source* is shifted right by 1 (maybe left-padded with eos)
        #
        # Left-to-right language models should condition on *source* and
        # predict *future_target*.
        source, future_target, _ = self.dataset[index]
        target = self._filter_vocab(future_target)

        source, target = self._maybe_add_bos(source, target)
        return {"id": index, "source": source, "target": target}

    def __len__(self):
        return len(self.dataset)

    def _maybe_add_bos(self, source, target):
        if self.add_bos_token:
            # src_lang_idx and tgt_lang_idx are passed in for multilingual LM, with the
            # first token being a lang_id token.
            bos = self.src_lang_idx or self.vocab.bos()
            source = torch.cat([source.new([bos]), source])
            if target is not None:
                tgt_bos = self.tgt_lang_idx or self.tgt_vocab.bos()
                target = torch.cat([target.new([tgt_bos]), target])
        return source, target

    def num_tokens_vec(self, indices):
        """Return the number of tokens for a set of positions defined by indices.
        This value is used to enforce ``--max-tokens`` during batching."""
        return self.sizes[indices]

    def _filter_vocab(self, target):
        if len(self.tgt_vocab) != len(self.vocab):

            def _filter(target):
                mask = target.ge(len(self.tgt_vocab))
                if mask.any():
                    target[mask] = self.tgt_vocab.unk()
                return target

            if isinstance(target, list):
                return [_filter(t) for t in target]
            return _filter(target)
        return target

    def collater(self, samples):
        """Merge a list of samples to form a mini-batch.

        Args:
            samples (List[dict]): samples to collate

        Returns:
            dict: a mini-batch with the following keys:

                - `id` (LongTensor): example IDs in the original input order
                - `ntokens` (int): total number of tokens in the batch
                - `net_input` (dict): the input to the Model, containing keys:

                  - `src_tokens` (LongTensor): a padded 2D Tensor of tokens in
                    the source sentence of shape `(bsz, src_len)`. Padding will
                    appear on the right.

                - `target` (LongTensor): a padded 2D Tensor of tokens in the
                  target sentence of shape `(bsz, tgt_len)`. Padding will appear
                  on the right.
        """
        return collate(
            samples,
            self.vocab.pad(),
            self.vocab.eos(),
            self.fixed_pad_length,
            self.pad_to_bsz,
        )

    def num_tokens(self, index):
        """Return the number of tokens in a sample. This value is used to
        enforce ``--max-tokens`` during batching."""
        return self.sizes[index]

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        return self.sizes[index]

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        if self.shuffle:
            order = [np.random.permutation(len(self))]
        else:
            order = [np.arange(len(self))]
        order.append(self.sizes)
        return np.lexsort(order)

    @property
    def supports_prefetch(self):
        return getattr(self.dataset, "supports_prefetch", False)

    def prefetch(self, indices):
        self.dataset.prefetch(indices)
