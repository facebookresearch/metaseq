# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
import random
import string
import json
import os
from metaseq import utils


def write_one_jsonl(jsonl_path, num_lines=5, text_len_min=5, text_len_max=50):
    data = []
    with open(jsonl_path, "w") as h:
        for _ in range(num_lines):
            text_len = random.choice(range(text_len_min, text_len_max))
            data.append(
                {"text": "".join(random.choices(string.ascii_letters, k=text_len))}
            )
            print(json.dumps(data[-1]), file=h)
    return data


def write_dummy_jsonl_data_dir(data_dir, num_lines=500):
    for subset in ["train", "valid"]:
        for shard in range(2):
            shard_dir = os.path.join(data_dir, subset, f"{shard:02}")
            os.makedirs(shard_dir)
            for dataset in ["a", "b"]:
                write_one_jsonl(
                    os.path.join(shard_dir, f"dataset_{dataset}.jsonl"),
                    num_lines=num_lines,
                )


def write_dummy_bpe(data_dir):
    from tokenizers import ByteLevelBPETokenizer

    tokenizer = ByteLevelBPETokenizer(add_prefix_space=True)
    tokenizer.train(
        [],
        vocab_size=500,
        special_tokens=["<s>", "<pad>", "</s>", "<unk>"],
        show_progress=False,
    )
    vocab, merges = tokenizer.save_model(data_dir)
    return vocab, merges


class TestUtils(unittest.TestCase):
    def test_make_positions(self):
        pad = 1
        left_pad_input = torch.LongTensor(
            [
                [9, 9, 9, 9, 9],
                [1, 9, 9, 9, 9],
                [1, 1, 1, 9, 9],
            ]
        )
        left_pad_output = torch.LongTensor(
            [
                [2, 3, 4, 5, 6],
                [1, 2, 3, 4, 5],
                [1, 1, 1, 2, 3],
            ]
        )
        right_pad_input = torch.LongTensor(
            [
                [9, 9, 9, 9, 9],
                [9, 9, 9, 9, 1],
                [9, 9, 1, 1, 1],
            ]
        )
        right_pad_output = torch.LongTensor(
            [
                [2, 3, 4, 5, 6],
                [2, 3, 4, 5, 1],
                [2, 3, 1, 1, 1],
            ]
        )

        self.assertAlmostEqual(
            left_pad_output,
            utils.make_positions(left_pad_input, pad),
        )
        self.assertAlmostEqual(
            right_pad_output,
            utils.make_positions(right_pad_input, pad),
        )

    def test_clip_grad_norm_(self):
        params = torch.nn.Parameter(torch.zeros(5)).requires_grad_(False)
        grad_norm = utils.clip_grad_norm_(params, 1.0)
        self.assertTrue(torch.is_tensor(grad_norm))
        self.assertEqual(grad_norm, 0.0)

        params = [torch.nn.Parameter(torch.zeros(5)) for _ in range(3)]
        for p in params:
            p.grad = torch.arange(1.0, 6.0)
        grad_norm = utils.clip_grad_norm_(params, 1.0, "l2")
        exp_grad_norm = torch.arange(1.0, 6.0).repeat(3).norm()
        self.assertTrue(torch.is_tensor(grad_norm))
        self.assertAlmostEqual(grad_norm, exp_grad_norm)

        grad_norm = utils.clip_grad_norm_(params, 1.0, "l2")
        self.assertAlmostEqual(grad_norm, torch.tensor(1.0))

        for p in params:
            p.grad = torch.arange(1.0, 6.0)
        grad_norm = utils.clip_grad_norm_(params, 1.0, "inf")
        exp_grad_norm = torch.arange(1.0, 6.0).max()
        self.assertEqual(grad_norm, exp_grad_norm)

    def test_resolve_max_positions_with_tuple(self):
        resolved = utils.resolve_max_positions(None, (2000, 100, 2000), 12000)
        self.assertEqual(resolved, (2000, 100, 2000))

    def assertAlmostEqual(self, t1, t2):
        self.assertEqual(t1.size(), t2.size(), "size mismatch")
        self.assertLess(utils.item((t1 - t2).abs().max()), 1e-4)


if __name__ == "__main__":
    unittest.main()
