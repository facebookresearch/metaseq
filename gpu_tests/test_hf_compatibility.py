#!/usr/bin/env python3
import os
from metaseq import checkpoint_utils, tasks, utils
from transformers import OPTForCausalLM
from packaging import version
import torch
import unittest
import torch.nn.functional as F
from metaseq.scripts.convert_to_singleton import create_generation_config_with_defaults
from metaseq.distributed import utils as distributed_utils
from metaseq.distributed import fsdp_enable_wrap, fsdp_wrap
from metaseq.dataclass.configs import MetaseqConfig
from metaseq.hub_utils import tensorize_input, get_next_token, setup_vocab_and_merges


prompts = [
    "Today is a beautiful day and I want to",
    "In the city of",
    "Paris is the capital of France and",
    "Computers and mobile phones have taken",
]


def load_mp_model_and_run_eval(cfg: MetaseqConfig, **kwargs):
    vocab_file, merges_file, tokenizer = setup_vocab_and_merges(kwargs["model_path"])
    orig_dims = []

    prompt_ids = []
    for prompt in prompts:
        input_ids = tensorize_input(tokenizer, prompt)
        # Pad sequence to length 32 to avoid Megatron assertion errors
        orig_dims.append(input_ids.shape[1])
        input_ids = F.pad(
            input=input_ids, pad=(0, 32 - input_ids.shape[1], 0, 0), value=1
        )
        prompt_ids.append(input_ids)

    prompt_ids = torch.cat(prompt_ids).cuda()

    task = tasks.setup_task(cfg.task)

    def _build_model(cfg, task):
        cfg.model.tensor_parallel_init_model_on_gpu = True
        model = task.build_model(cfg.model).cuda()
        return fsdp_wrap(model)

    with fsdp_enable_wrap(
        cfg.distributed_training,
        use_sharded_state=cfg.distributed_training.use_sharded_state,
    ):
        models, _model_args, _task = checkpoint_utils.load_model_ensemble_and_task(
            utils.split_paths(cfg.common_eval.path),
            arg_overrides=None,
            task=task,
            suffix=cfg.checkpoint.checkpoint_suffix,
            strict=True,
            num_shards=cfg.checkpoint.checkpoint_shard_count,
            build_model_hook=_build_model,
        )
        model = models[0]

    model.summon_full_params()
    model = model.eval()

    with torch.no_grad():
        logits = model(prompt_ids)[0]

    gathered_logits = [
        torch.zeros_like(logits)
        for _ in range(distributed_utils.get_model_parallel_world_size())
    ]
    torch.distributed.all_gather(
        gathered_logits, logits, group=distributed_utils.get_global_group()
    )
    gathered_logits = torch.cat(gathered_logits, dim=2)

    # Unwrap gathered logits into separate components for each prompt, and
    # trim them to match orig_dims
    trimmed_logits = [
        logits[:orig_dim].unsqueeze(0)
        for logits, orig_dim in zip(gathered_logits, orig_dims)
    ]
    return trimmed_logits


@unittest.skipIf(not torch.cuda.is_available(), "test requires a GPU")
@unittest.skipIf(
    version.parse(torch.__version__) < version.parse("1.9.1"),
    "test requires a pytorch version of at least 1.9.1",
)
class TestHFCompatibility(unittest.TestCase):
    def test_singleton_metaseq_hf_compatibility(self):
        model_path = os.path.join(os.path.dirname(__file__), "125m")
        vocab_file, merges_file, tokenizer = setup_vocab_and_merges(model_path)

        checkpoint = checkpoint_utils.load_model_ensemble_and_task(
            [os.path.join(model_path, "restored.pt")],
            arg_overrides={
                "vocab_filename": vocab_file,
                "merges_filename": merges_file,
            },
        )

        model = checkpoint[0][0].eval()

        hf_model = OPTForCausalLM.from_pretrained(model_path)

        for prompt in prompts:
            input_ids = tensorize_input(tokenizer, prompt)
            with torch.no_grad():
                logits_metaseq = model(input_ids)[0]
                logits_hf = hf_model(input_ids)[0]

            metaseq_next_token = get_next_token(logits_metaseq, tokenizer)
            hf_next_token = get_next_token(logits_hf, tokenizer)

            # Assert that HF and metaseq versions of the same model predict the same logits
            self.assertTrue(
                torch.allclose(logits_metaseq.cpu(), logits_hf.cpu(), atol=1e-3)
            )

            # Assert that HF and metaseq versions of the same model predict the same word
            self.assertEqual(metaseq_next_token, hf_next_token)

    def test_model_parallel_metaseq_hf_compatibility(self):
        model_path = os.path.join(os.path.dirname(__file__), "125m")

        cfg = create_generation_config_with_defaults(model_path)
        mp_logits_list = distributed_utils.call_main(
            cfg, load_mp_model_and_run_eval, model_path=model_path
        )

        # Verify that the generated logits match the consolidated model logits

        vocab_file, merges_file, tokenizer = setup_vocab_and_merges(model_path)
        checkpoint = checkpoint_utils.load_model_ensemble_and_task(
            [os.path.join(model_path, "restored.pt")],
            arg_overrides={
                "vocab_filename": vocab_file,
                "merges_filename": merges_file,
            },
        )
        model = checkpoint[0][0].eval()

        for prompt, logits_mp in zip(prompts, mp_logits_list):
            input_ids = tensorize_input(tokenizer, prompt)
            with torch.no_grad():
                logits_metaseq = model(input_ids)[0]

            metaseq_next_token = get_next_token(logits_metaseq, tokenizer)
            mp_next_token = get_next_token(logits_mp, tokenizer)

            # Assert that MP and metaseq versions of the same model predict the same logits
            self.assertTrue(
                torch.allclose(
                    logits_metaseq.cpu().float(), logits_mp.cpu().float(), atol=1e-1
                )
            )

            # Assert that MP and metaseq versions of the same model predict the same word
            self.assertEqual(metaseq_next_token, mp_next_token)


if __name__ == "__main__":
    unittest.main()
