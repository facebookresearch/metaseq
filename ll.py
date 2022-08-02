# Run the following to get a single consolidated checkpoint file e.g. `checkpoint_last_consolidated.pt
"""
model_dir=/private/home/myasu/projects/cm3-metaseq_minimum/models/vanilla
python -m metaseq.scripts.consolidate_fsdp_shards $model_dir/checkpoint_last --new_arch_name transformer_lm
"""


import random
import torch.distributed as dist

from metaseq import options
from metaseq.hub_utils import GeneratorInterface
from metaseq.dataclass.utils import convert_namespace_to_omegaconf


MICHI = False

caption = "A man riding a wave on a"
if MICHI:
    prompt1 = f'<img alt=\"{caption}\"'
else:
    prompt1 = caption


dist.init_process_group(
    backend="nccl", init_method="tcp://localhost:13000", world_size=1, rank=0
)

parser = options.get_generation_parser()
args = options.parse_args_and_arch(
    parser,
    input_args=[
        '--task',
        'cm3_language_modeling_inference_for_models_trained_with_streaming',
        '--spm-path',
        (
            '/shared/home/roller/V65536_I8192_S512_M512_R1024.json'
            if MICHI
            else '/shared/home/roller/V262144_I8192_S512_M512_R1024.json'
        ),
        '--path',
        (
            '/shared/home/roller/michi.pt'
            if MICHI
            else '/shared/home/roller/checkpoint_47_40000_consolidated_inference.pt'
        ),
        '--bpe',
        'hf_cm3_unigram',
        '/tmp',
    ],
)
cfg = convert_namespace_to_omegaconf(args)
# stupid hack around for FSDP & gpu initialization in this singleton script
cfg.common_eval.model_overrides = '{"tensor_parallel_init_model_on_gpu": False}'

gi = GeneratorInterface(cfg)
gi.load_model()
text_tokens = list(gi.bpe.bpe.encode(prompt1).ids)
tokens_orig = text_tokens
print(tokens_orig)
print(len(tokens_orig))

response = gi.generate(
    inputs=[tokens_orig],
    echo=True,
    max_tokens=[1080],
    temperature=0.85,
    top_p=1.0,
    seed=random.randint(1, 2000),
)
# print(json.dumps(response, indent=2))
output_text, output_tokens = (
    response[0][0]['text']
    .replace("<sentinel:0>", "")
    .replace('">', '')
    .replace('  ', ' ')
    .split('src="' if MICHI else ".")
)
print(response[0][0]['text'][:60])
if ')' in output_tokens:
    output_tokens = output_tokens[output_tokens.rindex(')') + 1 :]
output_image_tokens = output_tokens.split()
print(len(output_image_tokens))
while len(output_image_tokens) < 1024:
    output_image_tokens.append("I0")
print(" ".join(output_image_tokens))
