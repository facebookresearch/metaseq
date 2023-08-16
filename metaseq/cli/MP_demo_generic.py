#!/usr/bin/env python3 -u
# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Host the demo.

Launch with `python -m metaseq.cli.interactive_hosted` to run locally.

See docs/api.md for more information.
"""

import os
import ast
import random
import sys
import logging

import base64
from PIL import Image
import io


import torch
from flask import Flask, request, jsonify

import torch.distributed as dist
from metaseq import options, checkpoint_utils, tasks
from metaseq.dataclass.configs import MetaseqConfig
from metaseq.dataclass.utils import convert_namespace_to_omegaconf
from metaseq.distributed import utils as distributed_utils
from metaseq.hub_utils import GeneratorInterface
from metaseq.service.utils import build_logger
from metaseq.distributed import fsdp_enable_wrap, fsdp_wrap
from metaseq import utils, distributed_utils
from metaseq.file_io import PathManager

import importlib

import re
import functools
import gradio as gr


import torch
import torch.nn as nn
import math
import numpy as np
import logging

from PIL import Image
from typing import Optional, Dict, List
from metaseq import checkpoint_utils, file_utils
from tqdm import tqdm
from diffusers import StableDiffusionUpscalePipeline
from transformers import AutoImageProcessor, Swin2SRForImageSuperResolution
sys.path.append('/data/home/aiema/metaseq-internal/demo/')
from image_tokenizer import VqganTokenizer
from sequence_generators import (
    ImageSequenceGenerator,
    MixedSequenceGenerator,
    extract_image_tokens,
    map_old_image_token_to_new_image_token,
)

#from demo_utils import model_factory

#from retrieval import Retriever

app = Flask(__name__)

if "METASEQ_SERVICE_CONSTANTS_MODULE" not in os.environ:
    constants_module = importlib.import_module("metaseq.service.constants")
else:
    constants_module = importlib.import_module(
        os.environ["METASEQ_SERVICE_CONSTANTS_MODULE"]
    )
TOTAL_WORLD_SIZE = constants_module.TOTAL_WORLD_SIZE
LAUNCH_ARGS = constants_module.LAUNCH_ARGS
INFERENCE_ARG_OVERRIDES = constants_module.INFERENCE_ARG_OVERRIDES
SEED = 42

logger = build_logger()

model_id = "stabilityai/stable-diffusion-x4-upscaler"
sr_pipeline = StableDiffusionUpscalePipeline.from_pretrained(
    model_id, torch_dtype=torch.float16
)
sr_pipeline = sr_pipeline.to("cuda:1")

image_model = VqganTokenizer()

shutterstock_ret_kwargs = {
    "retrieve_meta": "/fsx-onellm/olggol/data/mscoco_racm3_data/sst_retrieval/collate_0_0/reclip_text_jsonl/final.jsonl",
    "faiss_index": "/fsx-onellm/olggol/data/mscoco_racm3_data/sst_retrieval/collate_0_0/reclip_text_index/text.index",
    # "retrieve_meta": "/fsx-cm3/liliyu/data/mscoco_racm3_data/sst_retrieval/collate_0_0/reclip_text_jsonl/final.jsonl",
    # "faiss_index": "/fsx-cm3/liliyu/data/mscoco_racm3_data/sst_retrieval/collate_0_0/reclip_text_index/text.index",
    "preprocess_text_type": "mscoco",
}

#retriever = Retriever(**shutterstock_ret_kwargs)

def image_to_base64(image: Image):
    byte_arr = io.BytesIO()
    image.save(byte_arr, format='PNG')
    byte_arr = byte_arr.getvalue()
    return base64.b64encode(byte_arr).decode('utf-8')  # .decode() converts bytes to string

def base64_to_image(base64_string: str):
    byte_arr = base64.b64decode(base64_string)  # .b64decode() converts string to bytes
    byte_arr = io.BytesIO(byte_arr)
    return Image.open(byte_arr)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def load_model(cfg):
    #global models
    global task 
    global ra_cm3_model
    
    if cfg.common.model_parallel_size == 1:
        r = distributed_utils.get_global_rank()
    else:
        r = distributed_utils.get_data_parallel_rank()

    suffix = cfg.checkpoint.checkpoint_suffix

    sharded_files = PathManager.ls(
        re.sub(".pt$", f"{suffix}*", cfg.common_eval.path)
    )
    if len(sharded_files) > 0 and "-shard" in sharded_files[0]:
        # We are loading a sharded checkpoint
        suffix += f"-shard{r}"
    else:
        suffix += ""

    cfg.checkpoint.checkpoint_suffix = suffix

    utils.import_user_module(cfg.common)

    # Fix seed for stochastic decoding
    if (
        cfg.common.seed is not None
        and not cfg.generation.no_seed_provided
    ):
        np.random.seed(cfg.common.seed)
        utils.set_torch_seed(cfg.common.seed)

    # Setup task, e.g., translation
    #task = #tasks.setup_task(cfg.task)

    # Load the model
    overrides = ast.literal_eval(cfg.common_eval.model_overrides)
    logger.info("loading model(s) from {}".format(cfg.common_eval.path))

    def _load_checkpoint():
        return checkpoint_utils.load_model_ensemble_and_task_demo(
            utils.split_paths(cfg.common_eval.path),
            arg_overrides=overrides,
            #task=task,
            suffix=cfg.checkpoint.checkpoint_suffix,
            strict=(cfg.checkpoint.checkpoint_shard_count == 1),
            num_shards=cfg.checkpoint.checkpoint_shard_count,
        )

    if cfg.distributed_training.ddp_backend == "fully_sharded":
        with fsdp_enable_wrap(
            cfg.distributed_training,
            use_sharded_state=cfg.distributed_training.use_sharded_state,
        ):
            models, _model_args, task = _load_checkpoint()
    else:
        models, _model_args, task = _load_checkpoint()
    ra_cm3_model = models[0] 
    
    
def parse_doc(doc):
    obj = re.match(r'<img alt="(.*?)" src="(I\d.*?)">', doc)
    text, image = obj.group(1), obj.group(2)
    return text, image


def tokenizer_encode(text: str, add_cm3_break_ind=True, add_eod=True):
    to_add = [task.cm3_break_ind] if add_cm3_break_ind else []
    to_prepend = [task.eod] if add_eod else []
    text_encoded = [] if len(text) == 0 else task.tokenizer.encode(text).ids

    return torch.tensor(to_prepend + text_encoded + to_add, dtype=torch.long)


def tokenize_text(text):
    text_indexes = task.tokenizer.encode(text.rstrip()).ids
    return text_indexes


def tokenize_image(image):
    image = image.rstrip() + " "
    image = map_old_image_token_to_new_image_token(image)
    image_indexes = task.tokenizer.encode(image.rstrip()).ids
    return image_indexes



def add_sr(image_full, text):
    with torch.inference_mode():
        return sr_pipeline(prompt=text, image=image_full).images[0]


def batchfy(inputs: List, batch_size: int):
    """
    Create batches of inputs
    """
    batches = []
    for batch_start in range(0, len(inputs), batch_size):
        curr_batch = inputs[batch_start : batch_start + batch_size]
        batches.append(curr_batch)
    return batches


@app.route('/', methods=['GET', 'POST'])
def handle_request_image():
    request_params = {
        'query': request.args.get('query'),
        'num_samples': int(request.args.get('num_samples')),
        'temp': float(request.args.get('temp')),
        'topp': float(request.args.get('topp')),
        'cfg_weight': float(request.args.get('cfg_weight')),
    }
    if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
        distributed_utils.broadcast_object(request_params, src_rank=0, group=distributed_utils.get_global_group())
   
    return generate_mixed(**request_params)

    
def generate_mixed(
    query,
    num_samples,
    temp,
    topp,
    cfg_weight,
    progress=None,
):
    set_seed(SEED)
    min_tokens = 64
    max_tokens = 512
    
    
    text_conditional = torch.tensor(form_prompt_text_only(query), dtype=torch.long).unsqueeze(0).to("cuda")
    text_unconditional = torch.tensor(
        [
            task.eod,
            task.cm3_sentinel_tokens_ind[0],
            task.cm3_break_ind,
        ],
        dtype=torch.long,
    ).unsqueeze(0).to("cuda")
    
    total_max_tokens = text_conditional.size(-1) + max_tokens
    total_min_tokens = text_conditional.size(-1) + min_tokens
    

    # 1. Initialize the MixedSequenceGenerator
    mixed_generator = MixedSequenceGenerator(
        model=ra_cm3_model,
        tgt_dict=task.source_dictionary,
        progress=progress,
        beam_size=num_samples,
        temperature=temp,
        topp=topp,
        cfg_weight=cfg_weight,
        eod = task.eod,
        max_len_b=total_max_tokens,
        min_len=total_min_tokens,
        mask_tokens=torch.tensor(
            [ind for x, ind in task.source_dictionary.indices.items() if "IMGIMG" in x]
        ),
        racm3_break=task.cm3_break_ind,
        cm3_sentinel = task.cm3_sentinel_tokens_ind[0]
        
    ).cuda()
    
  
    
    # 3. Generate using the combined_generate method
    generated_tokens = mixed_generator.combined_generate(
        {"net_input": {"src_tokens": text_conditional}},
        text_unconditional
    )
    print(f"generated_tokens: {generated_tokens}")
    #print(f"shape tokens {generated_tokens.shape}")
    # 4. Process the generated tokens
    all_captions = []
    all_images_base64 = []
    current_tokens = []

    is_image_modality = False
    last_text = ""
    image_delimiter = 99999
    
    flat_generated_tokens = [item for sublist in generated_tokens for item in (sublist if isinstance(sublist, (list, tuple)) else [sublist])]


    print(f"flattened:{flat_generated_tokens}")
    for token in flat_generated_tokens:
        if token == image_delimiter:
            if len(current_tokens) > 0:
                if not is_image_modality:
                    caption = task.tokenizer.decode(current_tokens, skip_special_tokens=False).split("<eoss>")[0].strip()
                    print(caption)
                    all_captions.append(caption)
                    last_text = caption
                    current_tokens = []
                else:
                    image = current_tokens[:1024]  # Take only first 1024 tokens
                    image = task.tokenizer.decode(image, skip_special_tokens=False)
                    image = extract_image_tokens(image)
                    image_full = image_model.decode(image)
                    #all_images_base64.append(image_to_base64(add_sr(image_full, last_text)))
                    all_images_base64.append(image_to_base64(image_full))
                    current_tokens = []
                    
            is_image_modality = not is_image_modality  # Flip modality
        else:
            current_tokens.append(token)
            
    print(all_captions)

    return {
        'all_captions': all_captions,
        'all_images': all_images_base64
    }

def form_prompt_text_only(text):
    all_tokens = []
    text_tokenized = task.tokenizer.encode(text).ids
    all_tokens.extend(
        [task.eod]
        + text_tokenized
        +[task.cm3_sentinel_tokens_ind[0]]
    )
    all_tokens = torch.tensor(all_tokens, dtype=torch.long)
    return all_tokens


def worker_main(cfg: MetaseqConfig, namespace_args=None):
    #os.environ["TOKENIZERS_PARALLELISM"] = "false"
    #torch.set_num_threads(1)
    global model
    # make sure generations are stochastic since we have many workers
    torch.manual_seed(random.randint(1, 20000))
    torch.cuda.manual_seed(random.randint(1, 20000))
    model = load_model(cfg)
    
    if torch.distributed.is_initialized():
        request_object = distributed_utils.broadcast_object(
            None, src_rank=0, group=distributed_utils.get_global_group()
        )
    
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        app.run(host="0.0.0.0", port=5000, threaded=True)
    else:
        logger.info(f"Looping engaged!")
        while True:
            try:
                request_object = distributed_utils.broadcast_object(
                    None, src_rank=0, group=distributed_utils.get_global_group()
                )
                _ = generate_mixed(**request_object)
            except Exception:
                # continue looping for the next generation so we don't lock up
                pass
    
def cli_main():
    """
    Command line interactive.
    """
    parser = options.get_generation_parser()
    # dumb defaults overriding
    parser.set_defaults(lr_scheduler=None, criterion=None)
    flat_launch_args = []
    for s in LAUNCH_ARGS:
        flat_launch_args += s.split()
    args = options.parse_args_and_arch(parser, input_args=flat_launch_args)
    args.data = os.path.dirname(args.path)  # hardcode the data arg
    cfg = convert_namespace_to_omegaconf(args)
    cfg.distributed_training.distributed_world_size = TOTAL_WORLD_SIZE

    model_overrides = ast.literal_eval(cfg.common_eval.model_overrides)
    model_overrides.update(INFERENCE_ARG_OVERRIDES)
    cfg.common_eval.model_overrides = str(model_overrides)
    distributed_utils.call_main(cfg, worker_main, namespace_args=args)


if __name__ == "__main__":
    if os.getenv("SLURM_NODEID") is None:
        logger.warning(
            f"Missing slurm configuration, defaulting to 'use entire node' for API"
        )
        os.environ["SLURM_NODEID"] = "0"
        os.environ["SLURM_NNODES"] = "1"
        os.environ["SLURM_NTASKS"] = "1"
        import socket
        os.environ["SLURM_STEP_NODELIST"] = socket.gethostname()
    cli_main()
