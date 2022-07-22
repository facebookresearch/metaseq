# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import tempfile
import unittest

import torch

import tests.utils as test_utils
from metaseq.data.dictionary import Dictionary
from metaseq.models.transformer_lm import TransformerLanguageModel
from metaseq.sequence_generator import SequenceGenerator
from metaseq.tasks.base_task import LegacyTask

DEFAULT_TEST_VOCAB_SIZE = 100


class DummyTask(LegacyTask):
    def __init__(self, args):
        super().__init__(args)
        self.dictionary = get_dummy_dictionary()
        if getattr(self.args, "ctc", False):
            self.dictionary.add_symbol("<ctc_blank>")
        self.src_dict = self.dictionary
        self.tgt_dict = self.dictionary

    @property
    def source_dictionary(self):
        return self.src_dict

    @property
    def target_dictionary(self):
        return self.dictionary


def get_dummy_dictionary(vocab_size=DEFAULT_TEST_VOCAB_SIZE):
    dummy_dict = Dictionary()
    # add dummy symbol to satisfy vocab size
    for id, _ in enumerate(range(vocab_size)):
        dummy_dict.add_symbol("{}".format(id), n=1000)
    return dummy_dict


def get_dummy_task_and_parser():
    """
    to build a fariseq model, we need some dummy parse and task. This function
    is used to create dummy task and parser to faciliate model/criterion test

    Note: we use FbSpeechRecognitionTask as the dummy task. You may want
    to use other task by providing another function
    """
    parser = argparse.ArgumentParser(
        description="test_dummy_s2s_task", argument_default=argparse.SUPPRESS
    )
    DummyTask.add_args(parser)
    args = parser.parse_args([])
    task = DummyTask.setup_task(args)
    return task, parser


class TestJitSequenceGeneratorBase(unittest.TestCase):
    def setUp(self):
        self.task, self.parser = get_dummy_task_and_parser()
        eos = self.task.tgt_dict.eos()
        src_tokens = torch.randint(3, 50, (2, 10)).long()
        src_tokens = torch.cat((src_tokens, torch.LongTensor([[eos], [eos]])), -1)
        src_lengths = torch.LongTensor([2, 10])
        self.sample = {
            "net_input": {"src_tokens": src_tokens, "src_lengths": src_lengths}
        }
        TransformerLanguageModel.add_args(self.parser)
        args = self.parser.parse_args([])
        args.decoder_layers = 1
        self.transformer_model = TransformerLanguageModel.build_model(args, self.task)

    def assertOutputEqual(self, hypo, pos_probs):
        pos_scores = torch.FloatTensor(pos_probs).log()
        self.assertTensorSizeEqual(hypo["positional_scores"], pos_scores)
        self.assertTensorSizeEqual(pos_scores.numel(), hypo["tokens"].numel())

    def assertTensorSizeEqual(self, t1, t2):
        self.assertEqual(t1.size(), t2.size(), "size mismatch")

    def assertAlmostEqual(self, t1, t2):
        self.assertEqual(t1.size(), t2.size(), "size mismatch")
        self.assertLess((t1 - t2).abs().max(), 1e-4)

    def assertTensorEqual(self, t1, t2):
        self.assertEqual(t1.size(), t2.size(), "size mismatch")
        self.assertEqual(t1.ne(t2).long().sum(), 0)

    def assertHypoEqual(self, h1, h2):
        "Check two hypos are equal"
        self.assertTensorEqual(h1["tokens"], h2["tokens"])
        self.assertAlmostEqual(h1["positional_scores"], h2["positional_scores"])
        self.assertLess(abs(h1["score"] - h2["score"]), 1e-6)
        self.assertAlmostEqual(h1["attention"], h2["attention"])

    def _test_save_and_load(self, scripted_module):
        with tempfile.NamedTemporaryFile() as f:
            scripted_module.save(f.name)
            torch.jit.load(f.name)


JIT_MSG = "Targeting OSS scriptability for the 1.6 release"


@unittest.skipIf(torch.__version__ < "1.6.0", JIT_MSG)
class TestJitSequenceGenerator(TestJitSequenceGeneratorBase):
    def test_export_transformer(self):
        model = self.transformer_model
        torch.jit.script(model)

    def test_sequence_generator(self):
        model = self.transformer_model
        generator = SequenceGenerator(
            [model],
            self.task.tgt_dict,
            beam_size=2,
            max_len_b=10,
        )
        scripted_model = torch.jit.script(generator)
        self._test_save_and_load(scripted_model)


class TestSequenceGeneratorBase(unittest.TestCase):
    def assertHypoTokens(self, hypo, tokens):
        self.assertTensorEqual(hypo["tokens"], torch.LongTensor(tokens))

    def assertHypoScore(self, hypo, pos_probs, normalized=True):
        pos_scores = torch.FloatTensor(pos_probs).log()
        self.assertAlmostEqual(hypo["positional_scores"], pos_scores)
        self.assertEqual(pos_scores.numel(), hypo["tokens"].numel())
        score = pos_scores.sum()
        if normalized:
            score /= pos_scores.numel()
        self.assertLess(abs(score - hypo["score"]), 1e-6)

    def assertAlmostEqual(self, t1, t2):
        self.assertEqual(t1.size(), t2.size(), "size mismatch")
        self.assertLess((t1 - t2).abs().max(), 1e-4)

    def assertTensorEqual(self, t1, t2):
        self.assertEqual(t1.size(), t2.size(), "size mismatch")
        self.assertEqual(t1.ne(t2).long().sum(), 0)


class TestSequenceGenerator(TestSequenceGeneratorBase):
    def setUp(self):
        (
            self.tgt_dict,
            self.w1,
            self.w2,
            src_tokens,
            src_lengths,
            self.model,
        ) = test_utils.sequence_generator_setup()
        self.sample = {
            "net_input": {"src_tokens": src_tokens, "src_lengths": src_lengths}
        }

    def test_with_normalization(self):
        generator = SequenceGenerator([self.model], self.tgt_dict, beam_size=2)
        hypos = generator.forward(self.sample)
        eos, w1, w2 = self.tgt_dict.eos(), self.w1, self.w2
        # sentence 1, beam 1
        self.assertHypoTokens(hypos[0][0], [w1, eos])
        self.assertHypoScore(hypos[0][0], [0.9, 1.0])
        # sentence 1, beam 2
        self.assertHypoTokens(hypos[0][1], [w2, w1, w2, eos])
        self.assertHypoScore(hypos[0][1], [0.1, 0.9, 0.9, 1.0])
        # sentence 2, beam 1
        self.assertHypoTokens(hypos[1][0], [w1, w2, w1, eos])
        self.assertHypoScore(hypos[1][0], [0.7, 0.4, 0.4, 1.0])
        # sentence 2, beam 2
        self.assertHypoTokens(hypos[1][1], [w1, w2, eos])
        self.assertHypoScore(hypos[1][1], [0.7, 0.4, 0.6])

    def test_without_normalization(self):
        # Sentence 1: unchanged from the normalized case
        # Sentence 2: beams swap order
        generator = SequenceGenerator([self.model], self.tgt_dict, beam_size=2)
        hypos = generator.forward(self.sample)
        eos, w1, w2 = self.tgt_dict.eos(), self.w1, self.w2
        # sentence 1, beam 1
        self.assertHypoTokens(hypos[0][0], [w1, eos])
        self.assertHypoScore(hypos[0][0], [0.9, 1.0], normalized=False)
        # sentence 1, beam 2
        self.assertHypoTokens(hypos[0][1], [w2, w1, w2, eos])
        self.assertHypoScore(hypos[0][1], [0.1, 0.9, 0.9, 1.0], normalized=False)
        # sentence 2, beam 1
        self.assertHypoTokens(hypos[1][0], [w1, w2, eos])
        self.assertHypoScore(hypos[1][0], [0.7, 0.4, 0.6], normalized=False)
        # sentence 2, beam 2
        self.assertHypoTokens(hypos[1][1], [w1, w2, w1, eos])
        self.assertHypoScore(hypos[1][1], [0.7, 0.4, 0.4, 1.0], normalized=False)

    def test_maxlen(self):
        generator = SequenceGenerator(
            [self.model], self.tgt_dict, beam_size=2, max_len_b=2
        )
        hypos = generator.forward(self.sample)
        eos, w1, w2 = self.tgt_dict.eos(), self.w1, self.w2
        # sentence 1, beam 1
        self.assertHypoTokens(hypos[0][0], [w1, eos])
        self.assertHypoScore(hypos[0][0], [0.9, 1.0])
        # sentence 1, beam 2
        self.assertHypoTokens(hypos[0][1], [w2, w2, eos])
        self.assertHypoScore(hypos[0][1], [0.1, 0.1, 0.6])
        # sentence 2, beam 1
        self.assertHypoTokens(hypos[1][0], [w1, w2, eos])
        self.assertHypoScore(hypos[1][0], [0.7, 0.4, 0.6])
        # sentence 2, beam 2
        self.assertHypoTokens(hypos[1][1], [w2, w2, eos])
        self.assertHypoScore(hypos[1][1], [0.3, 0.9, 0.01])


if __name__ == "__main__":
    unittest.main()
