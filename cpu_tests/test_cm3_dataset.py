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
            (torch.LongTensor([2, 2, 2, eod]), None),
            (torch.LongTensor([3, 3, 3, eod]), None),
            (torch.LongTensor([4, 4, 4, eod]), None),
            (torch.LongTensor([5, 5, 5, eod]), None),
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


def get_image_span_dataset(
    sentinel_token_expectation: int = 0,
    sentinel_tokens: List[int] = [10],
    sentinel_method: str = "fixed",
    sentinel_eos: int = 1,
    allow_rotation_across_eod: bool = False,
    eod: int = 0,
    block_size: int = 6,
):  # [(),]
    dataset = TensorListDataset(
        [                     #0, 1, 2, 3,   4, 5, 6, 7, 8, 9
            (torch.LongTensor([2, 2, 2, eod, 6, 6, 6, 6, 6, eod]), [(4, 10),]),
            (torch.LongTensor([3, 3, 3, eod, 7, 7, 7, eod, 17, 17]), [(4, 8), (8, 10),]),
            (torch.LongTensor([4, 4, 4, eod, 8, 8, eod, 18, 18, eod]), [(7, 10),]),
            (torch.LongTensor([5, 5, 5, eod, 9, eod, 19, eod, 20, eod]), [(4, 6), (8, 10)]),
            (torch.LongTensor([6, 6, eod, 9, eod, 22, eod, 20, 21, eod]), None)
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
        block_size=block_size,
        permute_documents=False,
        break_mode="none",
        padding_idx=1,
        no_break_image=True,
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


class TestNoImageBreak(unittest.TestCase):
    def test_no_image_break(self):
        eod = 0

        dataset = TensorListDataset(
        [                     #0, 1, 2, 3,   4, 5, 6, 7, 8, 9
            (torch.LongTensor([2, 2, 2, eod, 6, 6, 6, 6, 6, eod]), [(4, 10),]),
            (torch.LongTensor([3, 3, 3, eod, 7, 7, 7, eod, 17, 17]), [(4, 8), (8, 10),]),
            (torch.LongTensor([4, 4, 4, eod, 8, 8, eod, 18, 18, eod]), [(7, 10),]),
            (torch.LongTensor([5, 5, 5, eod, 9, eod, 19, eod, 20, eod]), [(4, 6), (8, 10)]),
            (torch.LongTensor([6, 6, eod, 9, eod, 22, eod, 20, 21, eod]), None)
        ]
        )

        causal_masked_dataset = get_image_span_dataset(eod=0, block_size=5)
        expected_datasets = [[2, 2, 2, eod, 6,],
                             [3, 3, 3, eod, 7,],
                             [17, 17, 4, 4, 4,],
                             [eod, 8, 8, eod, 18,],
                             [5, 5, 5, eod, 9,],
                             [19, eod, 20, eod, 6],
                             [6, eod, 9, eod, 22,],
                            ]
        for ii, item in enumerate(causal_masked_dataset):
            item = item["block"].numpy().tolist()
            assert item == expected_datasets[ii], "data item doesn't match expected!"
        assert ii == len(expected_datasets) - 1, "data size doesn't match expected!"


        causal_masked_dataset = get_image_span_dataset(eod=0, block_size=7)
        expected_datasets = [[2, 2, 2, eod, 6, 6, 6,],
                             [3, 3, 3, eod, 7, 7, 7,],
                             [17, 17, 4, 4, 4, eod, 8,],
                             [8, eod, 18, 18, eod, 5, 5,],
                             [5, eod, 9, eod, 19, eod, 20],
                             [6, 6, eod, 9, eod, 22, eod,],
                            ]
        for ii, item in enumerate(causal_masked_dataset):
            item = item["block"].numpy().tolist()
            assert item == expected_datasets[ii], "data item doesn't match expected!"
        assert ii == len(expected_datasets) - 1, "data size doesn't match expected!"

if __name__ == "__main__":
    unittest.main()
