#!/usr/bin/env python3
import os
from transformers import GPT2Tokenizer
from metaseq import checkpoint_utils
from transformers import OPTForCausalLM
from packaging import version
import torch
import unittest


# forward passes
def forward(model, tokenizer, prompts):
    input_ids = tokenizer(prompts, return_tensors="pt").input_ids
    input_ids = torch.cat([torch.tensor([[0]]), input_ids], dim=-1)
    input_ids = input_ids
    with torch.no_grad():
        logits = model(input_ids)[0]
    return logits


def get_next_token(logits, tokenizer):
    pred_next_token = torch.argmax(logits[0, -1], -1)
    next_token = tokenizer.convert_ids_to_tokens([pred_next_token])
    next_token = next_token[0].replace("Ġ", "")
    return next_token


@unittest.skipIf(not torch.cuda.is_available(), "test requires a GPU")
@unittest.skipIf(
    version.parse(torch.__version__) < version.parse("1.9.1"),
    "test requires a pytorch version of at least 1.9.1",
)
class TestTraining(unittest.TestCase):
    def test_metaseq_hf_compatibility(self):
        model_path = os.path.join(os.path.dirname(__file__), "125m")

        vocab_file = os.path.join(model_path, "gpt2-vocab.json")
        merges_file = os.path.join(model_path, "gpt2-merges.txt")

        tokenizer = GPT2Tokenizer(vocab_file, merges_file)
        tokenizer.save_pretrained(model_path)

        checkpoint = checkpoint_utils.load_model_ensemble_and_task(
            [os.path.join(model_path, "restored.pt")],
            arg_overrides={
                "vocab_filename": vocab_file,
                "merges_filename": merges_file,
            },
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
            logits_metaseq = forward(model, tokenizer, prompt)
            metaseq_next_token = get_next_token(logits_metaseq, tokenizer)

            logits_hf = forward(hf_model, tokenizer, prompt)
            hf_next_token = get_next_token(logits_hf, tokenizer)

            # Assert that HF and metaseq versions of the same model predict the same logits
            self.assertTrue(
                torch.allclose(logits_metaseq.cpu(), logits_hf.cpu(), atol=1e-3)
            )

            # Assert that HF and metaseq versions of the same model predict the same word
            self.assertEqual(metaseq_next_token, hf_next_token)


if __name__ == "__main__":
    unittest.main()
