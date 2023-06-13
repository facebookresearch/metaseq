import json
import torch
import unittest

from argparse import Namespace
from metaseq.tasks.streaming_distillation_language_modeling import StreamingDistillationLanguageModelingTask


class TestResamplingDataset(unittest.TestCase):

    def setUp(self):
        self.dict_args = {}
        self.dict_args['hf_tokenizer'] = ""
        self.dict_args['vocab_filename'] = "./tests/tokenizer_files/gpt2-vocab.json"
        self.dict_args['merges_filename'] = "./tests/tokenizer_files/gpt2-merges.txt"
        self.dict_args['update_freq'] = [1]
        self.dict_args['final_vocab_size'] = None
        self.dict_args['distillation_mode'] = "logprobs_distillation"
        self.dict_args['target_key'] = "top_logprobs"
        self.dict_args['beam_results_key'] = "beam_results"
        self.dict_args['end_of_document_symbol'] = "</s>"
        self.dict_args['prompt_eos_text'] = "TL;DR"
        self.dict_args['tokens_per_sample'] = 128
        self.dict_args['truncation'] = "right"

        # This test loads the same training sample in two different input formats.
        # The first one is the training format, the second one is the inference format.
        # Both of them must generate the same tokens, so we can use the same method
        # to test the consistency of the data.
        self.path_to_training_data_sample = './tests/input_files/distillation_training_format_sample.json'
        self.path_to_inference_data_sample = './tests/input_files/distillation_inference_format_sample.json'

    def test_tokenization_training_format(self):
        self.dict_args['source_key'] = "src"
        distillation_task = StreamingDistillationLanguageModelingTask(Namespace(**self.dict_args))
        with open(self.path_to_training_data_sample, 'r') as f:
            json_line = f.read()
        json_dict = json.loads(json_line)
        test = distillation_task._tokenize_source_target_json(json_dict)
        source_and_target_tokens = test[0]
        masked_source_and_target_tokens = test[1]
        src_size = len(distillation_task.tokenizer.encode(json_dict['src']).ids)
        if self.dict_args['prompt_eos_text'] is not None and self.dict_args['prompt_eos_text'] != "":
            src_size += len(distillation_task.tokenizer.encode(self.dict_args['prompt_eos_text']).ids)
        self._test_data_consistency(source_and_target_tokens, masked_source_and_target_tokens, src_size)

    def test_tokenization_inference_format(self):
        self.dict_args['source_key'] = "prompt_text"
        distillation_task = StreamingDistillationLanguageModelingTask(Namespace(**self.dict_args))
        with open(self.path_to_inference_data_sample, 'r') as f:
            json_line = f.read()
        json_dict = json.loads(json_line)
        test = distillation_task._tokenize_source_target_json(json_dict)
        source_and_target_tokens = test[0]
        masked_source_and_target_tokens = test[1]
        src_size = len(distillation_task.tokenizer.encode(json_dict['prompt_text']).ids)
        if self.dict_args['prompt_eos_text'] is not None and self.dict_args['prompt_eos_text'] != "":
            src_size += len(distillation_task.tokenizer.encode(self.dict_args['prompt_eos_text']).ids)
        self._test_data_consistency(source_and_target_tokens, masked_source_and_target_tokens, src_size)

    def _test_data_consistency(self, source_and_target_tokens, masked_source_and_target_tokens, src_size):
        self.assertTrue(source_and_target_tokens.shape == masked_source_and_target_tokens.shape)

        # Hard labels test
        expected_source_and_target_tokens = torch.LongTensor(
            [
                7061, 111, 766, 10975, 7083, 8913, 783, 7479, 443, 10975, 14853, 2100, 7479, 284, 40700, 352, 10975, 2362, 742,
                50118, 50118, 565, 462, 131, 10232, 22290, 131, 10644, 726, 8913, 783, 11, 5, 343, 2100, 16, 45, 284, 5192,
                6928, 4, 2
            ]
        )
        expected_masked_source_and_target_tokens = torch.LongTensor(
            [
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 726, 8913, 783, 11, 5, 343,
                2100, 16, 45, 284, 5192, 6928, 4, 2
            ]
        )
        hard_label_source_and_target_tokens = source_and_target_tokens[:, 0, 0].long()
        hard_label_masked_source_and_target_tokens = masked_source_and_target_tokens[:, 0, 0].long()
        self.assertTrue(torch.all(hard_label_source_and_target_tokens == expected_source_and_target_tokens))
        self.assertTrue(torch.all(hard_label_masked_source_and_target_tokens == expected_masked_source_and_target_tokens))

        # top 5 tokens for src must be 0 except for the hard labels
        src_tokens_not_hard_label = source_and_target_tokens[:src_size, 1:, 0].long()
        self.assertTrue(torch.all(src_tokens_not_hard_label == 0))

        # on the masked version, all src tokens must be 1 (padding mask)
        masked_src_tokens_not_hard_label = masked_source_and_target_tokens[:src_size, 1:, 0].long()
        self.assertTrue(torch.all(masked_src_tokens_not_hard_label == 1))

        # top 5 tokens for target must not be 0 as all tokens are model predictions
        src_tokens_not_hard_label = source_and_target_tokens[src_size:-1, :, 0].long()
        self.assertTrue(torch.all(src_tokens_not_hard_label != 0))
        masked_src_tokens_not_hard_label = masked_source_and_target_tokens[src_size:-1, :, 0].long()
        self.assertTrue(torch.all(masked_src_tokens_not_hard_label != 0))

        # the last token must be the EOS with only the hard label different from 0
        eos_token_id = 2
        last_token = source_and_target_tokens[-1, :, 0].long()
        self.assertTrue(last_token[0] == eos_token_id)
        self.assertTrue(torch.all(last_token[1:] == 0))
        masked_last_token = masked_source_and_target_tokens[-1, :, 0].long()
        self.assertTrue(masked_last_token[0] == eos_token_id)
        self.assertTrue(torch.all(masked_last_token[1:] == 0))

        # Check if source predictions are 0 for source tokens
        source_and_target_predictions = source_and_target_tokens[:, :, 1]
        masked_source_and_target_predictions = masked_source_and_target_tokens[:, :, 1]
        self.assertTrue(torch.all(source_and_target_predictions[:src_size] == 0))
        self.assertTrue(torch.all(masked_source_and_target_predictions[:src_size] == 0))

        # Check if target prediction match in both tensors
        self.assertTrue(torch.all(source_and_target_predictions[src_size:] == masked_source_and_target_predictions[src_size:]))


if __name__ == "__main__":
    unittest.main()
