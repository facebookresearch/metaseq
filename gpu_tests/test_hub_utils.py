#!/usr/bin/env python3
import unittest
import os
from metaseq.scripts.convert_to_singleton import create_generation_config_with_defaults
from metaseq.hub_utils import GeneratorInterface, GeneratorHubInterface
from metaseq.distributed import utils as distributed_utils
from metaseq.dataclass.configs import MetaseqConfig
import numpy as np
from megatron.mpu import destroy_model_parallel
from metaseq.distributed import fsdp_wrap, fsdp_enable_wrap
from metaseq import hub_utils
import torch.distributed as dist
import torch


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


def default_post_build_model_hook(model, task):
    # For fp16
    model = model.half()
    # fsdp_wrap will be a no-op if not using FSDP
    if hasattr(model, "make_generation_fast_"):
        model.make_generation_fast_()
    return fsdp_wrap(model)


def override_task(model):
    # HACK: Use special version of the language_modeling task for inference with the models
    # trained with streaming_language_modeling.
    # TODO: More elegant solution would be to decouple the training and inference logic for the tasks

    from omegaconf import OmegaConf
    from metaseq import tasks

    task_config = OmegaConf.create(
        tasks.TASK_DATACLASS_REGISTRY[
            "language_modeling_inference_for_models_trained_with_streaming"
        ]()
    )
    task_config._name = "language_modeling_inference_for_models_trained_with_streaming"
    task_config.data = model.cfg.task.data
    task_config.vocab_filename = model.cfg.task.vocab_filename
    task_config.merges_filename = model.cfg.task.merges_filename

    new_task = tasks.setup_task(task_config, criterion_args=model.cfg.criterion)
    model.task = new_task
    model.cfg.task = task_config
    model.setup_task()


def generate_using_generator_interface(cfg: MetaseqConfig, **kwargs):
    generator = GeneratorInterface(cfg)
    models = generator.load_model()  # noqa: F841

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

    # Destroy torch distributed process groups. This needs to be executed in each spawned process
    # https://github.com/pytorch/pytorch/issues/48203
    dist.destroy_process_group()
    return generated_text


def generate_using_generator_hub_interface(cfg: MetaseqConfig, **kwargs):
    model_path = os.path.join(os.path.dirname(__file__), "125m")
    kwargs = {
        "suffix": f"-model_part-{torch.distributed.get_rank()}",
        "criterion": "cross_entropy",
        "bpe": "hf_byte_bpe",
        "post_build_model_hook": default_post_build_model_hook,
        "bpe_merges": os.path.join(kwargs["model_path"], "gpt2-merges.txt"),
        "merges_filename": os.path.join(kwargs["model_path"], "gpt2-merges.txt"),
        "bpe_vocab": os.path.join(kwargs["model_path"], "gpt2-vocab.json"),
        "vocab_filename": os.path.join(kwargs["model_path"], "gpt2-vocab.json"),
        "bpe_add_prefix_space": False,
        "specify_arch": True,
        "tensor_parallel_init_model_on_gpu": True,
        # 'batch_size': None,
        # 'batch_size_valid': None
    }

    # Need to enable this context otherwise fsdp_wrap is a no op. See fairscale.nn.wrap
    with fsdp_enable_wrap(cfg.distributed_training, use_sharded_state=True):
        x = hub_utils.from_pretrained(
            model_path,
            "reshard.pt",
            **kwargs,
        )

    generator = GeneratorHubInterface(
        x["args"],
        x["task"],
        x["models"],
        moe_disable_padding=False,
        skip_prepare_for_inference=True,
    )

    override_task(generator)
    generator = generator.cuda()

    tokenized_sentences = [torch.tensor(PROMPT)]
    hypothesis = generator.generate(
        tokenized_sentences=tokenized_sentences,
        score_reference=True,
        batch_size=1,
        compute_vocab_dist=False,
    )

    # Destroy torch distributed process groups. This needs to be executed in each spawned process
    # https://github.com/pytorch/pytorch/issues/48203
    dist.destroy_process_group()
    return hypothesis


class TestHubUtils(unittest.TestCase):
    def test_generator_interface(self):
        model_path = os.path.join(os.path.dirname(__file__), "125m")
        cfg = create_generation_config_with_defaults(
            model_path, ddp_backend="fully_sharded"
        )

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
                    np.allclose(
                        EXPECTED_OUTPUT[key][1:], generated_beam[key][1:], atol=1e-2
                    )
                )
                # Ensure that the score associated to </s> is None
                self.assertIsNone(generated_beam[key][0])
            else:
                self.assertEqual(EXPECTED_OUTPUT[key], generated_beam[key])

    def test_generator_hub_interface(self):
        model_path = os.path.join(os.path.dirname(__file__), "125m")
        cfg = create_generation_config_with_defaults(
            model_path, ddp_backend="fully_sharded"
        )

        overall_generation = distributed_utils.call_main(
            cfg, generate_using_generator_hub_interface, model_path=model_path
        )
        # We can potentially pass a list of inputs to the generate function.
        # Here, we look at the first (and only) generation
        [generation_for_prompt] = overall_generation
        # We use best_of = 1, so we get only one beam search result
        [generated_beam] = generation_for_prompt

        self.assertTrue(
            torch.equal(torch.tensor(PROMPT), generated_beam["tokens"].cpu())
        )
        # TOKEN_SCORES[1:] is done to skip the first "None"
        self.assertTrue(
            np.allclose(
                TOKEN_SCORES[1:], generated_beam["positional_scores"].cpu(), atol=1e-2
            )
        )
        self.assertTrue(np.isclose(-3.4824, generated_beam["score"].item(), atol=1e-2))

    def tearDown(self):
        # Tear down model parallel
        destroy_model_parallel()
        distributed_utils._USE_MEGATRON = False


if __name__ == "__main__":
    unittest.main()
