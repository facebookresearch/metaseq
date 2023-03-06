# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List
from metaseq.data.cm3_dataset import CausalMaskedDocumentToSequenceDataset, adjust_spans

import torch
import unittest
import random
import numpy as np


class TensorListDataset(torch.utils.data.Dataset):
    def __init__(self, tensor_list):
        self.tensor_list = tensor_list
        self.queried = 0

    def __len__(self):
        return len(self.tensor_list)

    def __getitem__(self, idx):
        self.queried += 1
        return self.tensor_list[idx]


def get_simple_dataset(
    sentinel_token_expectation: int = 1,
    sentinel_tokens: List[int] = [10],
    sentinel_method: str = "fixed",
    sentinel_eos: int = 1,
    allow_rotation_across_eod: bool = True,
    eod: int = 0,
):
    dataset = TensorListDataset(
        [
            torch.LongTensor([2, 2, 2, eod]),
            torch.LongTensor([3, 3, 3, eod]),
            torch.LongTensor([4, 4, 4, eod]),
            torch.LongTensor([5, 5, 5, eod]),
        ]
    )
    dataset = CausalMaskedDocumentToSequenceDataset(
        sentinel_token_expectation,
        sentinel_tokens,
        sentinel_method,
        sentinel_eos,
        allow_rotation_across_eod,
        eod,
        dataset,
        block_size=16,
        permute_documents=False,
        break_mode="none",
        padding_idx=1,
    )
    dataset.set_epoch(0)
    return dataset


class TestDocumentBoundaryMethods(unittest.TestCase):
    def test_get_document_boundaries(self):
        eod = 0
        causal_masked_dataset = get_simple_dataset(eod=0)

        item = torch.LongTensor([1, 1, eod, 1, 1, 1, eod, 1])
        assert causal_masked_dataset.get_document_boundaries(item) == [
            (0, 2),
            (2, 6),
            (6, 8),
        ]

        item = torch.LongTensor([eod, 1, 1, eod, 1, 1, 1, eod, 1, eod])
        assert causal_masked_dataset.get_document_boundaries(item) == [
            (0, 3),
            (3, 7),
            (7, 9),
        ]


class TestAdjustSpans(unittest.TestCase):
    def test_no_overlap(self):
        spans_1 = [(1, 3), (5, 7), (9, 11)]
        constraints = [(0, 1), (1, 4), (4, 5), (5, 8), (8, 9), (9, 11)]
        self.assertEqual(adjust_spans(spans_1, constraints), spans_1)

    def test_overlap_within(self):
        spans_1 = [(1, 4), (5, 7), (9, 11)]
        spans_2 = [(2, 3), (3, 4), (4, 5), (5, 8), (8, 10), (10, 11)]
        self.assertEqual(adjust_spans(spans_1, spans_2), [(2, 3), (5, 7), (9, 10)])

    def test_overlap_outside(self):
        spans_1 = [(1, 4), (5, 7), (9, 11)]
        spans_2 = [(0, 3), (3, 5), (5, 7), (7, 8), (8, 10), (10, 12)]
        self.assertEqual(adjust_spans(spans_1, spans_2), [(1, 3), (5, 7), (9, 10)])

    def test_overlap_edge(self):
        spans_1 = [(1, 4), (5, 7), (9, 12)]
        spans_2 = [(0, 3), (3, 5), (5, 7), (7, 9), (9, 10), (10, 12)]
        self.assertEqual(adjust_spans(spans_1, spans_2), [(1, 3), (5, 7), (10, 12)])

    def test_no_similar_span(self):
        spans_1 = [(1, 3), (5, 7), (9, 11)]
        spans_2 = [(0, 1), (4, 5), (12, 13)]
        self.assertEqual(adjust_spans(spans_1, spans_2), spans_1)

    def test_empty_spans(self):
        spans_1 = []
        spans_2 = []
        self.assertEqual(adjust_spans(spans_1, spans_2), [])


class TestCM3Dataset(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)
        np.random.seed(0)
        random.seed(0)

    def test_rotation_fim(self):
        dataset = get_simple_dataset()
        for dataset in [
            get_simple_dataset(),
            get_simple_dataset(allow_rotation_across_eod=True),
        ]:
            for item in dataset:
                self.assertEqual(len(item["block"]), dataset.block_size)
                self.assertEqual(
                    (item["block"] == dataset.sentinel_tokens[0]).nonzero().size(0),
                    2,
                    f"{item['block']} should contain only 2 {dataset.sentinel_tokens[0]} tokens."
                    "One for mask, one for prefix for generating mask.",
                )


if __name__ == "__main__":
    unittest.main()
