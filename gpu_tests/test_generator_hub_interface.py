#!/usr/bin/env python3
import torch
import unittest
import os
from metaseq.scripts.convert_to_singleton import create_generation_config_with_defaults
from metaseq.hub_utils import GeneratorHubInterface
from metaseq.distributed import utils as distributed_utils
from metaseq.dataclass.configs import MetaseqConfig
import numpy as np
from metaseq import hub_utils
from metaseq.distributed import fsdp_wrap, fsdp_enable_wrap
from megatron.mpu import destroy_model_parallel
import torch.distributed as dist
from test_generator_interface import PROMPT, TOKEN_SCORES


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
    dist.destroy_process_group()
    return hypothesis


class TestGeneratorHubInterface(unittest.TestCase):
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
            np.allclose(TOKEN_SCORES[1:], generated_beam["positional_scores"].cpu())
        )
        self.assertTrue(np.isclose(-3.4824, generated_beam["score"].item()))

    def tearDown(self):
        # Tear down model parallel
        destroy_model_parallel()


if __name__ == "__main__":
    unittest.main()
