#!/usr/bin/env python3
import os
from transformers import AutoTokenizer, GPT2Tokenizer
from metaseq import checkpoint_utils
from transformers import OPTForCausalLM
import torch
import unittest


# forward passes
def single_batch_forward_logits(model, tokenizer, prompts):
    input_ids = tokenizer(prompts, return_tensors="pt").input_ids
    input_ids = torch.cat([torch.tensor([[0]]), input_ids], dim=-1)
    input_ids = input_ids
    with torch.no_grad():
        logits = model(input_ids)[0]
    return logits

# forward hf
def forward_hf(hf_model, tokenizer, prompts):
    input_ids = tokenizer(prompts, return_tensors="pt").input_ids
    input_ids = torch.cat([torch.tensor([[0]]), input_ids], dim=-1)
    input_ids = input_ids
    with torch.no_grad():
        logits = hf_model(input_ids)[0]
    return logits

@unittest.skipIf(not torch.cuda.is_available(), "test requires a GPU")
class TestTraining(unittest.TestCase):
    def test_metaseq_hf_compatibility(self):
        model_path = "./125m"

        vocab_file = os.path.join(model_path, "gpt2-vocab.json")
        merges_file = os.path.join(model_path, "gpt2-merges.txt")

        tokenizer = GPT2Tokenizer(vocab_file, merges_file)
        tokenizer.save_pretrained(model_path)

        checkpoint = checkpoint_utils.load_model_ensemble_and_task(
            [os.path.join(model_path, "restored.pt")],
            arg_overrides={
                "vocab_filename": vocab_file,
                "merges_filename": merges_file,
            }
        )

        model = checkpoint[0][0].eval()
        model = model

        hf_model = OPTForCausalLM.from_pretrained(model_path)

        prompts = [
            "Today is a beautiful day and I want to",
            "In the city of",
            "Paris is the capital of France and",
            "Computers and mobile phones have taken",
        ]

        for prompt in prompts:
            logits_metaseq = single_batch_forward_logits(model, tokenizer, prompt)
            pred_next_token = torch.argmax(logits_metaseq[0, -1], -1)
            metaseq_next_token = tokenizer.convert_ids_to_tokens([pred_next_token])
            metaseq_next_token = metaseq_next_token[0].replace("Ġ", "")

            logits_hf = forward_hf(model, tokenizer, prompt)
            pred_next_token = torch.argmax(logits_hf[0, -1], -1)
            hf_next_token = tokenizer.convert_ids_to_tokens([pred_next_token])
            hf_next_token = hf_next_token[0].replace("Ġ", "")

            # Assert that HF and metaseq versions of the same model predict the same logits
            self.assertTrue(torch.allclose(logits_metaseq.cpu(), logits_hf.cpu(), atol=1e-3))

            # Assert that HF and metaseq versions of the same model predict the same word
            self.assertEqual(metaseq_next_token, hf_next_token)


if __name__ == "__main__":
    unittest.main()
