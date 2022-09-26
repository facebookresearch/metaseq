#!/usr/bin/env python3
import torch
import unittest
import os
from metaseq.scripts.convert_to_singleton import create_generation_config_with_defaults
from metaseq.hub_utils import GeneratorInterface, GeneratorHubInterface
from metaseq.dataclass.utils import convert_namespace_to_omegaconf
from metaseq.distributed import utils as distributed_utils
from metaseq.dataclass.configs import MetaseqConfig
import numpy as np
from metaseq import hub_utils
from metaseq.distributed import fsdp_wrap, fsdp_enable_wrap
from megatron.mpu import destroy_model_parallel
import torch.distributed as dist


PROMPT = [133, 313, 1224, 15, 5, 856, 17527, 594, 98, 5, 11471, 3820, 19, 514, 4]
TOKEN_SCORES = [
    None,
    -2.436279535293579,
    -6.024158477783203,
    -7.712856292724609,
    -4.207103729248047,
    -1.188389539718628,
    -6.652636528015137,
    -0.9851812124252319,
    -0.04523317888379097,
    -4.7183380126953125,
    -2.6682472229003906,
    -4.960669040679932,
    -7.002664089202881,
    -1.146365761756897,
    -1.3687334060668945,
    -1.1196397542953491,
]


def generate_using_generator_interface(cfg: MetaseqConfig, **kwargs):
    generator = GeneratorInterface(cfg)
    models = generator.load_model()  # noqa: F841

    # print(f"generator_interface {[elem for elem in models[0].parameters()]} + + + + rank is {torch.distributed.get_rank()}")

    request_object = {
        "inputs": [PROMPT],
        "temperature": 1.0,
        "max_tokens": [0],
        "min_tokens": [0],
        "top_p": 1.0,
        "n": 1,
        "best_of": 1,
        "echo": True,
        "logprobs": 0,
        "seed": 1,
    }

    generated_text = generator.generate(**request_object)
    dist.destroy_process_group()
    return generated_text


class TestGeneratorInterface(unittest.TestCase):
    def test_generator_interface(self):
        model_path = os.path.join(os.path.dirname(__file__), "125m")
        cfg = create_generation_config_with_defaults(
            model_path, ddp_backend="fully_sharded"
        )
        # cfg = convert_namespace_to_omegaconf(cfg)
        # cfg.model.tensor_parallel_init_model_on_gpu = True

        overall_generation = distributed_utils.call_main(
            cfg, generate_using_generator_interface
        )
        # We can potentially pass a list of inputs to the generate function.
        # Here, we look at the first (and only) generation
        [generation_for_prompt] = overall_generation
        # We use best_of = 1, so we get only one beam search result
        [generated_beam] = generation_for_prompt

        EXPECTED_OUTPUT = {
            "text": "The man turned on the faucet so the toilet filled with water.",
            "tokens": [
                "</s>",
                "The",
                " man",
                " turned",
                " on",
                " the",
                " f",
                "auc",
                "et",
                " so",
                " the",
                " toilet",
                " filled",
                " with",
                " water",
                ".",
            ],
            "text_offset": [0, 3, 7, 14, 17, 21, 23, 26, 28, 31, 35, 42, 49, 54, 60],
            "token_scores": TOKEN_SCORES,
            "top_logprobs": None,
        }

        for key in EXPECTED_OUTPUT:
            if key == "token_scores":
                # Getting rid of the initial "None" element
                self.assertTrue(
                    np.allclose(EXPECTED_OUTPUT[key][1:], generated_beam[key][1:])
                )
                # Ensure that the score associated to </s> is None
                self.assertIsNone(generated_beam[key][0])
            else:
                self.assertEqual(EXPECTED_OUTPUT[key], generated_beam[key])

    def tearDown(self):
        # Tear down model parallel
        destroy_model_parallel()


if __name__ == "__main__":
    unittest.main()
