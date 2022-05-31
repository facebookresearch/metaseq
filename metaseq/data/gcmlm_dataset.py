# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch

from metaseq.data import BaseDataset, data_utils

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
    positions = merge("positions")
    doc_ids = merge("doc_ids")
    bidir_attn_mask = merge("bidir_attn_mask")
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
            "positions": positions,
            "doc_ids": doc_ids,
            "bidir_attn_mask": bidir_attn_mask,
        },
        "target": target,
    }


class GCMLMDataset(BaseDataset):
    """Generalized Dataset for CLM and MLM.

    Args:
        dataset (~torch.utils.data.Dataset): dataset to break into blocks
        sizes (List[int]): sentence lengths (required for 'complete' and 'eos')
        mask_idx (int): token idx of mask token
        mask_prob (float, Optional): ratio of masked tokens
        clm_prob (float, Optional): ratio of sentences which are fully causual and no mask
        fully_causal_prob (float, Optional): ratio of sentences which are fully causual
        fully_bidir_prob (float, Optional): ratio of sentences which are fully bidirectional
        predict_masked_only (boolean, Optional): if true, only predict masked tokens

    """

    def __init__(
        self,
        dataset,
        sizes,
        vocab,
        mask_idx,
        mask_prob=0.15,
        clm_prob=0.0,
        fully_causal_prob=0.0,
        fully_bidir_prob=0.0,
        predict_masked_only=False,
        shuffle=False,
        fixed_pad_length=None,
        pad_to_bsz=None,
    ):
        assert mask_prob <= 1.0
        assert clm_prob <= 1.0
        assert fully_causal_prob + fully_bidir_prob <= 1.0
        self.dataset = dataset
        self.sizes = np.array(sizes)
        self.vocab = vocab
        self.mask_idx = mask_idx
        self.mask_prob = mask_prob
        self.clm_prob = clm_prob
        self.fully_causal_prob = fully_causal_prob
        self.fully_bidir_prob = fully_bidir_prob
        self.predict_masked_only = predict_masked_only
        self.shuffle = shuffle
        self.fixed_pad_length = fixed_pad_length
        self.pad_to_bsz = pad_to_bsz

    def __getitem__(self, index):
        target = self.dataset[index]
        seqlen = target.numel()
        source = torch.roll(target, 1)

        is_docstart = ((source == target) & (source == 2)).roll(1)
        is_docstart[0] = True
        doc_ids = torch.cumsum(is_docstart, dim=0, dtype=torch.long) + self.vocab.pad()
        docstart_indices = is_docstart.nonzero().squeeze(1).tolist()
        ndocs = len(docstart_indices)
        doc_ranges = list(zip(docstart_indices, docstart_indices[1:] + [seqlen]))
        doc_lens = [end - start for start, end in doc_ranges]

        r = np.random.rand()
        if r < self.clm_prob:
            bidir_attn_prefix = [1] * ndocs  # The first token always has bidirectional attention
            mask_prob = 0
        else:
            mask_prob = self.mask_prob
            r = np.random.rand()
            if r < self.fully_causal_prob:
                bidir_attn_prefix = [1] * ndocs
            elif r < self.fully_causal_prob + self.fully_bidir_prob:
                bidir_attn_prefix = doc_lens
            else:
                bidir_attn_prefix = np.random.randint(min(doc_lens,[2] * len(doc_lens))) + 1

        # Position numbers begin at padding_idx+1. Padding symbols are ignored.
        positions = torch.arange(seqlen) + self.vocab.pad() + 1

        # Start positions from the beginning for each document
        # Only makes sense if masking attention across documents
        # positions = torch.cat([torch.arange(n) + self.vocab.pad() + 1 for n in doc_lens])

        # Mask tokens
        if mask_prob > 0:
            mask = torch.rand(seqlen) < mask_prob
            target[mask] = source[mask]
            source[mask] = self.mask_idx

        bidir_attn_mask = torch.zeros(seqlen, dtype=torch.bool)
        for (i, j), doclen, prefix in zip(doc_ranges, doc_lens, bidir_attn_prefix):
            if mask_prob > 0:
                current_mask = mask[i:j]
                nmask = current_mask.sum()
                if nmask > 0:
                    source[i:j] = torch.cat([source[i:j][~current_mask], source[i:j][current_mask]])
                    target[i:j] = torch.cat([target[i:j][~current_mask], target[i:j][current_mask]])
                    positions[i:j] = torch.cat([positions[i:j][~current_mask], positions[i:j][current_mask]])
            else:
                nmask = 0
            # Apply bidirectional prefix
            target[i:i+min(prefix - 1, doclen - nmask)] = self.vocab.pad()
            bidir_attn_mask[i:i+prefix] = True
            if self.predict_masked_only:
                target[i:i+doclen-nmask] = self.vocab.pad()

        return {
            "id": index,
            "source": source,
            "target": target,
            "positions": positions,
            "doc_ids": doc_ids,
            "bidir_attn_mask": bidir_attn_mask,
        }

    def __len__(self):
        return len(self.dataset)

    def num_tokens_vec(self, indices):
        """Return the number of tokens for a set of positions defined by indices.
        This value is used to enforce ``--max-tokens`` during batching."""
        return self.sizes[indices]

    def collater(self, samples):
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