# Run the following to get a single consolidated checkpoint file e.g. `checkpoint_last_consolidated.pt
"""
model_dir=/private/home/myasu/projects/cm3-metaseq_minimum/models/vanilla
python -m metaseq.scripts.consolidate_fsdp_shards $model_dir/checkpoint_last --new_arch_name transformer_lm
"""

import sys
import random
import torch.distributed as dist

from metaseq import options
from metaseq.hub_utils import GeneratorInterface
from metaseq.dataclass.utils import convert_namespace_to_omegaconf
from metaseq_internal.constants import TRAINED_MODEL_CONFIG

MICHI = False

caption = "A man riding a wave on a"
if MICHI:
    prompt1 = f'<img alt="{caption}"'
else:
    prompt1 = caption


dist.init_process_group(
    backend="nccl", init_method="tcp://localhost:13000", world_size=1, rank=0
)

parser = options.get_generation_parser()
LAUNCH_ARGS = {
    "task": "cm3_language_modeling_inference_for_models_trained_with_streaming",
    "bpe": "hf_cm3_unigram",
    "data": "/tmp",
}

flat_launch_args = []
if len(sys.argv) > 1:
    model_name = sys.argv[1]
    env_name = "azure" if len(sys.argv) < 3 else sys.argv[2]
else:
    model_name = "speech_consolidated"
try:
    model_config = TRAINED_MODEL_CONFIG[env_name][model_name]
except KeyError:
    print(
        f"{model_name} on {env_name} config not found, please add that in metaseq_internal.constants"
    )
del model_config["lauch_port"]
flat_launch_args = []
for k, v in model_config.items():
    k = k.replace("_", "-")
    flat_launch_args.append(f"--{k}")
    flat_launch_args.append(str(v))
flat_launch_args += [
    "--task",
    "cm3_language_modeling_inference_for_models_trained_with_streaming",
    "--bpe",
    "hf_cm3_unigram",
    "/tmp",
]
args = options.parse_args_and_arch(
    parser,
    input_args=flat_launch_args,
)
cfg = convert_namespace_to_omegaconf(args)
# stupid hack around for FSDP & gpu initialization in this singleton script
cfg.common_eval.model_overrides = '{"tensor_parallel_init_model_on_gpu": False}'

gi = GeneratorInterface(cfg)
gi.load_model()

# print('ready to generate')
if 'speech' in model_name:
    while True:
        try:
            text_tokens = list(gi.bpe.bpe.encode(input()).ids)
            tokens_orig = text_tokens
            # print(tokens_orig)
            response = gi.generate(
                inputs=[tokens_orig],
                echo=True,
                max_tokens=[1100],
                temperature=0.85,
                seed=random.randint(1, 2000),
            )
            # print(json.dumps(response, indent=2))
            output_text, output_tokens = response[0][0]["text"].split("<u2t>")
            print(output_tokens)
        except:
            print("ERROR")
            continue
else:
    while True:
        try:
            text_tokens = list(gi.bpe.bpe.encode(input()).ids)
            tokens_orig = text_tokens
            response = gi.generate(
                inputs=[tokens_orig],
                echo=True,
                max_tokens=[1100],
                temperature=0.85,
                top_p=1.0,
                seed=random.randint(1, 2000),
            )
            
            output_text, output_tokens = response[0][0]["text"].split(".")
            if ")" in output_tokens:
                output_tokens = output_tokens[output_tokens.rindex(")") + 1 :]
            # image_tokenizer take 1024 tokens
            output_image_tokens = output_tokens.split()
            print(len(output_image_tokens))
            while len(output_image_tokens) < 1024:
                output_image_tokens.append("I0")
            print(" ".join(output_image_tokens))
        except:
            print("ERROR")
            continue
