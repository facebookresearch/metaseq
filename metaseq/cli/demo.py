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

import torch

import torch.distributed as dist
from metaseq import options
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
    TextSequenceGenerator,
    extract_image_tokens,
    map_old_image_token_to_new_image_token,
)
#from demo_utils import model_factory

#from retrieval import Retriever



if "METASEQ_SERVICE_CONSTANTS_MODULE" not in os.environ:
    constants_module = importlib.import_module("metaseq.service.constants")
else:
    constants_module = importlib.import_module(
        os.environ["METASEQ_SERVICE_CONSTANTS_MODULE"]
    )
TOTAL_WORLD_SIZE = constants_module.TOTAL_WORLD_SIZE
LAUNCH_ARGS = constants_module.LAUNCH_ARGS
INFERENCE_ARG_OVERRIDES = constants_module.INFERENCE_ARG_OVERRIDES

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

# task_instruction = {
#     "pix2pix": "Edit first image following the instruction",
#     "canny": "Make high quality image from canny edge features",
#     "uniformer": "Make high quality image from uniformer segementation features",
#     "hed": "Make high quality image from hed features",
#     "openpose": "Make high quality image from openpose features",
#     "mlsd": "Make high quality image from mlsd lines",
#     "depth": "Make high quality image from midas depth features",
#     "norm": "Make high quality image from midas norm features",
#     "scribble": "Make high quality image from scribble"
# }

#retriever = Retriever(**shutterstock_ret_kwargs)

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
        re.sub(".pt$", f"{suffix}*", "/fsx-llm/aiema/checkpoints/joint/eval/mp_1500.pt")
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
    _, _, task = checkpoint_utils.load_model_ensemble_and_task_demo(
        filenames=[
            # "/data/cm3z/armenag/code/ra_cm3/models/760m/maiden_760_v1/checkpoint_182000_consolidated.pt"
            "/fsx-llm/aiema/checkpoints/uniform/consolidated/consolidated_2000_demo.pt"
            #"/fsx-cm3/liliyu/trained_model/pretrained/7b/maiden_7b_v1/checkpoint_298000_consolidated.pt"
    ]
    )


    def _build_model(cfg, task):
        model = task.build_model(cfg.model).cuda()
        model.make_generation_fast_()
        return fsdp_wrap(model)

    # Load the model
    overrides = ast.literal_eval(cfg.common_eval.model_overrides)
    logger.info("loading model(s) from {}".format(cfg.common_eval.path))

    def _load_checkpoint():
        return checkpoint_utils.load_model_ensemble_and_task(
            utils.split_paths(cfg.common_eval.path),
            arg_overrides=overrides,
            task=task,
            suffix=cfg.checkpoint.checkpoint_suffix,
            strict=(cfg.checkpoint.checkpoint_shard_count == 1),
            num_shards=cfg.checkpoint.checkpoint_shard_count,
            build_model_hook=_build_model,
        )

    if cfg.distributed_training.ddp_backend == "fully_sharded":
        with fsdp_enable_wrap(
            cfg.distributed_training,
            use_sharded_state=cfg.distributed_training.use_sharded_state,
        ):
            models, _model_args, _task = _load_checkpoint()
    else:
        models, _model_args, task = _load_checkpoint()
        
        
    #ra_cm3_model = models[0].eval()
    #print(f"racm3_evalf{ra_cm3_model}")
    #print(f"models[0]:{models[0]}") 
    #print(f"type{type(models[0])}")
    ra_cm3_model = models[0] 
    print(f"racm3:{ra_cm3_model}")
    
    #return ra_cm3_model 
    #print(ra_cm3_model)
  

# def load_model(model_ckpt):

#     global ra_cm3_model
#     global task

#     #del ra_cm3_model
#     #del models
#     empty_cuda_cache()

#     print("Loading new model")
#     models, args, task = checkpoint_utils.load_model_ensemble_and_task(
#         filenames=[
#             ""
#         ],
#         arg_overrides={
#             # some tokenizers are stored in FAIR dev, override
#             "hf_tokenizer": "/fsx-onellm/olggol/trained_model/tokenizers/racm3/gpt2-unified-image-racm3-patch.json",
#         },
#     )
#     print("Done loading model: ")
#     ra_cm3_model = models[0].cuda().eval()
    
    
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


def form_prompt(query, query1, query2, image1, image2, image_type="path", dropquery=False):
    if image_type == "path":
        tokenize_func = lambda x: " ".join(image_model.encode(x).ids)
    elif image_type == "token":
        tokenize_func = lambda x: x
    all_tokens, doc1, doc2 = [], [], []
    if query1 is not None:
        doc1 = (
            tokenize_text(query1)
        )
    if image1 is not None:
        image1 = tokenize_func(image1)
        doc1.extend([task.cm3_break_ind] + tokenize_image(image1) + [task.cm3_break_ind])
    if image2 is not None and query2 is not None:
        image2 = tokenize_func(image2)
        doc2 = (
            tokenize_text(query2)
            + [task.cm3_break_ind]
            + tokenize_image(image2)
            + [task.cm3_break_ind]
        )
    if not dropquery:
        query_token = tokenize_text(query) + [task.cm3_break_ind]
        all_tokens = [task.eod] + doc1 + doc2 + query_token
    else:
        all_tokens = [task.eod] + doc1 + doc2
    return all_tokens

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


def get_sr_list(lowr_image, image_sr_processor, image_sr_model):
    max_batch = 16
    all_images = []
    for batch in batchfy(lowr_image, max_batch):
        inputs = image_sr_processor(batch, return_tensors="pt")
        # forward pass
        with torch.inference_mode():
            outputs = image_sr_model(pixel_values=inputs["pixel_values"].cuda())

        outputs = outputs.reconstruction.data.float().cpu().clamp_(0, 1).numpy()
        outputs = np.moveaxis(outputs, source=1, destination=-1)
        outputs = (outputs * 255.0).round().astype(np.uint8)  # float32 t

        for i in range(outputs.shape[0]):
            image = Image.fromarray(outputs[i])
            all_images.append(image)
    return all_images

def generate_image(
    query,
    temp = 1.0,
    topp = 0.8,
    cfg_weight = 3,
    icfg_weight = 1.2,
    tcfg_weight = 7.5,
    num_sample = 1,
    query1 = None,
    image1 = None,
    query2 = None,
    image2 = None,
    progress=gr.Progress(),
    using_retrieval=False,
    pix2pix = False
):
    set_seed(42)
    if pix2pix:
        if query1 is None:
            query1 = task_instruction["pix2pix"]
        logger.info(f"Query: {query} ---- Support 1 {query1} ---- Support 2 {query2}")
        logger.info(f"Temperature: {temp}")
        decoder = Instrictpix2pixGenerator(
            ra_cm3_model,
            task.source_dictionary,
            progress,
            beam_size=num_sample,
            temperature=temp,
            image_cfg_weight=icfg_weight,
            text_cfg_weight=tcfg_weight,
        ).cuda()
        text_unconditional = torch.tensor(
            [
                task.eod,
                task.cm3_sentinel_tokens_ind[0],
                task.cm3_break_ind,
            ],
            dtype=torch.long,
        )
        text_conditional = torch.tensor(
            form_prompt(
                query,
                query1,
                query2,
                image1,
                image2
            ),
            dtype=torch.long,
        )
        text_unconditional_withimage = torch.tensor(
            form_prompt(
                query,
                query1,
                query2,
                image1,
                image2, 
                dropquery=True
            ),
            dtype=torch.long,
        )
        tokens = decoder(
            text_conditional.unsqueeze(0),
            text_unconditional.unsqueeze(0),
            text_unconditional_withimage.unsqueeze(0),
        )["tokens"]

    else:
        if using_retrieval:
            #ret_docs = retriever.search(query)
            #query1, image1 = parse_doc(ret_docs[0])
            #query2, image2 = parse_doc(ret_docs[1])
            image_type = "token"
        else:
            image_type = "path"

        logger.info(f"Query: {query} ---- Support 1 {query1}, Support 2 {query2}")
        decoder = ImageSequenceGenerator(
            ra_cm3_model,
            task.source_dictionary,
            progress,
            beam_size=num_sample,
            temperature=temp,
            topp=topp,
            cfg_weight=cfg_weight,
        ).cuda()

        text_unconditional = torch.tensor(
            [
                task.eod,
                task.cm3_sentinel_tokens_ind[0],
                task.cm3_break_ind,
            ],
            dtype=torch.long,
        )
        text_conditional = torch.tensor(
            form_prompt(
                #decoder, 
                query, query1, query2, image1, image2, image_type=image_type
            ),
            dtype=torch.long,
        )
        tokens = decoder(text_conditional.unsqueeze(0), text_unconditional.unsqueeze(0))[
            "tokens"
        ]
    tokens = tokens.view(-1, tokens.size(2))

    all_images = []
    images = []
    for image in tokens.cpu().numpy().tolist():
        image = image[:1024]
        image = task.tokenizer.decode(image, skip_special_tokens=False)
        image = extract_image_tokens(image)
        image_full = image_model.decode(image)
        images.append(image_full)

    all_images = [add_sr(image, query) for image in images]
    return all_images

    
def generate_text(
    image,
    query,
    num_samples,
    temp,
    topp,
    cfg_weight,
    max_tokens,
    model,
    progress=gr.Progress(),
):
    text_conditional = torch.tensor(
        form_prompt(query=query, query1=None, query2=None, image1=image, image2=None),
        dtype=torch.long)
    total_max_tokens = text_conditional.size(-1) + max_tokens

    decoder = TextSequenceGenerator(
        model,
        task.source_dictionary,
        progress=progress,
        max_len=total_max_tokens,
        beam_size=num_samples,
        temperature=temp,
        topp=topp,
        cfg_weight=cfg_weight,
    ).cuda()

    tokens = decoder(text_conditional.unsqueeze(0))[
        "tokens"
    ]
    tokens = tokens.view(-1, tokens.size(2))

    # remove prefix tokens
    tokens = tokens[:, text_conditional.size(-1) :]

    all_captions = {}
    for ind, caption in enumerate(tokens.cpu().numpy().tolist()):
        caption = task.tokenizer.decode(caption, skip_special_tokens=False)
        caption = caption.split("<eoss>")[0].strip()
        all_captions[str(ind)] = caption

    return all_captions





def worker_main(cfg: MetaseqConfig, namespace_args=None):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    torch.set_num_threads(1)
    
    # make sure generations are stochastic since we have many workers
    torch.manual_seed(random.randint(1, 20000))
    torch.cuda.manual_seed(random.randint(1, 20000))
    model = load_model(cfg)
    #print(ra_cm3_model)
    if dist.get_rank() > 0:
        sys.stdout = open(os.devnull, "w")
    if dist.get_rank() == 0:   
        with gr.Blocks() as demo:
            gr.Markdown(
                f"""
                # Internal playground V{0.0}.
                Adjust parameters if needed and start playing.
                """)
            with gr.Row().style(equal_height=True):
                temp = gr.Number(
                    label="Temperature", info="Must be greater than 0", value=1.0
                )
                topp = gr.Number(label="TopP", info="Must be greater than 0", value=0.8)
                cfg_weight = gr.Number(
                    label="CFG Weight", info="Must be greater than 0", value=3
                )
                icfg_weight = gr.Number(
                        label="IMG CFG Weight", info="Must be greater than 0, used in pix2pix generation", value=1.2
                    )
                tcfg_weight = gr.Number(
                        label="TEXT CFG Weight", info="Must be greater than 0, used in pix2pix generation", value=7.5
                    )
                num_samples = gr.Number(
                    label="Number of samples to generate",
                    info="Must be greater than 0",
                    value=1,
                    precision=0,
                )
                max_new_tokens = gr.Number(
                        label="Max new tokens for text generation",
                        info="Must be greater than 0",
                        value=64,
                        precision=0,
                )

            with gr.Row().style(equal_height=True):
                with gr.Column():
                    query_text = gr.Textbox(
                        placeholder="Input your prompt here.\nExample 1: Tree in a desert.\nExample2: Describe all the objects in the given image in very detail.", 
                        lines=7,
                        label="Text prompt"
                    )
                    image = gr.Image()
                    btn = gr.Button("Generate text-to-image", full_width=False)
                    btn3 = gr.Button("Generate image to text <not working zero-shot fundamental model>", full_width=False)
                    btn2 = gr.Button("Modify image with instructions", full_width=False)
                    instruction = gr.Textbox(value="Edit first image following the instruction", lines=7, label="Instruction")
                
                gallery = gr.Gallery(
                    label="Generated images", elem_id="gallery"
                ).style(grid=[2], height="auto")
                output_view = gr.JSON(label="Generated Captions")
            
            
            generate_ret = functools.partial(generate_image, using_retrieval=False)
            generate_pix2pix = functools.partial(generate_image, pix2pix=True)

            btn.click(
                generate_ret,
                inputs=[
                    query_text,
                    temp,
                    topp,
                    cfg_weight,
                    icfg_weight,
                    tcfg_weight,
                    num_samples,
                    
                ],
                outputs=[gallery]
            )

            btn2.click(
                generate_pix2pix,
                inputs=[
                    query_text,
                    temp,
                    topp,
                    cfg_weight,
                    icfg_weight,
                    tcfg_weight,
                    num_samples,
                    instruction,
                    image
                ],
                outputs=[gallery]
            )

            btn3.click(
                generate_text,
                #[image, query_text, num_samples, temp, topp, cfg_weight, max_new_tokens],
                [image, query_text, num_samples, temp, topp, max_new_tokens],
                output_view,
            )
        demo.queue().launch(share=True)
    else:
        while True:
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
