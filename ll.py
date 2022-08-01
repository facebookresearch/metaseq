# Run the following to get a single consolidated checkpoint file e.g. `checkpoint_last_consolidated.pt
"""
model_dir=/private/home/myasu/projects/cm3-metaseq_minimum/models/vanilla
python -m metaseq.scripts.consolidate_fsdp_shards $model_dir/checkpoint_last --new_arch_name transformer_lm
"""


import random
import json, os, re
import numpy as np
from tqdm import tqdm
from typing import Any, Dict, Iterator, List, Optional
from tokenizers import (
    ByteLevelBPETokenizer,
    Tokenizer,
    decoders,
    models,
    normalizers,
    pre_tokenizers,
)
from tokenizers.pre_tokenizers import ByteLevel, Digits
from metaseq.service.utils import encode_fn

TOKENIZER_FILE = "/shared/home/roller/V262144_I8192_S512_M512_R1024.json"


def load_tokenizer():
    tokenizer = Tokenizer(models.Unigram()).from_file(TOKENIZER_FILE)
    tokenizer.normalizer = normalizers.NFKC()
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
        [ByteLevel(), Digits(individual_digits=True)]
    )
    tokenizer.decoder = decoders.ByteLevel()
    return tokenizer


tokenizer = load_tokenizer()


import torch
from metaseq.models.transformer_lm import TransformerLanguageModel


caption = "A man riding a wave on a"
prompt1 = f'<img alt=\"{caption}\"'

from metaseq import options
from metaseq.hub_utils import GeneratorInterface
from metaseq.dataclass.utils import convert_namespace_to_omegaconf

parser = options.get_generation_parser()
args = options.parse_args_and_arch(
    parser,
    input_args=[
        '--task',
        'cm3_language_modeling_inference_for_models_trained_with_streaming',
        '--spm-path',
        # '/shared/home/roller/V262144_I8192_S512_M512_R1024.json',
        '/shared/home/roller/V65536_I8192_S512_M512_R1024.json',
        '--path',
        # '/shared/home/roller/checkpoint_47_40000_consolidated_inference.pt',
        '/shared/home/roller/michi.pt',
        # '--beam',
        # '1',
        # '--temperature',
        # '0.0',
        '--bpe',
        'hf_cm3_unigram',
        '/tmp',
    ],
)
cfg = convert_namespace_to_omegaconf(args)

gi = GeneratorInterface(cfg)
gi.load_model()
text_tokens = list(gi.bpe.bpe.encode(prompt1).ids)
text_tokens = [2] + text_tokens
# text_tokens += [1] * (255 - len(text_tokens))
# img_tokens = list(gi.bpe.bpe.encode(prompt2).ids)
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
    .split('src="')
)
print(response[0][0]['text'][:60])
output_image_tokens = output_tokens.split()
print(len(output_image_tokens))
while len(output_image_tokens) < 1024:
    output_image_tokens.append("I0")
print(" ".join(output_image_tokens))
