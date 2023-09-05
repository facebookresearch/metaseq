import json, os, re, sys
import subprocess
import numpy as np
from metaseq import checkpoint_utils

from metaseq.tasks.streaming_language_modeling import parse_doc
import torch

import submitit
from tqdm import tqdm
import logging
import random
import shutil
import tempfile
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
from metaseq import checkpoint_utils, file_utils
from metaseq.service.utils import build_logger
from metaseq.distributed import fsdp_enable_wrap, fsdp_wrap

from itertools import islice

import importlib

import re
import functools

import ast
sys.path.append('/data/home/aiema/metaseq-internal/inference/')
from clip_rerank import ClipScorer
from post_reranking import OpenClip

sys.path.append('/data/home/aiema/metaseq-internal/demo/')
from image_tokenizer import VqganTokenizer
from sequence_generators import (
    ImageSequenceGenerator,
    SelfContrastiveImageSequenceGenerator,
    extract_image_tokens,
    map_old_image_token_to_new_image_token,
)


from retrieval import Retriever

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

model_factory = {
    "760m_54k": "/data/cm3z/liliyu/data/trained_models/ra_cm3/models/760m/ckpt-54000/checkpoint_54000_consolidated.pt",
    "350m_178k": "/data/cm3z/armenag/code/ra_cm3/models/350m/maiden_v0/checkpoint_178000_consolidated.pt",
    "760m_178k": "/data/cm3z/liliyu/data/trained_models/ra_cm3/760m/ckpt-178000/checkpoint_178000_consolidated.pt",
}

shutterstock_ret_kwargs = {
    "retrieve_meta": "/fsx-onellm/olggol/data/mscoco_racm3_data/sst_retrieval/collate_0_0/reclip_text_jsonl/final.jsonl",
    "faiss_index": "/fsx-onellm/olggol/data/mscoco_racm3_data/sst_retrieval/collate_0_0/reclip_text_index/text.index",
    # "retrieve_meta": "/fsx-cm3/liliyu/data/mscoco_racm3_data/sst_retrieval/collate_0_0/reclip_text_jsonl/final.jsonl",
    # "faiss_index": "/fsx-cm3/liliyu/data/mscoco_racm3_data/sst_retrieval/collate_0_0/reclip_text_index/text.index",
    "preprocess_text_type": "mscoco",
}

retriever = Retriever(**shutterstock_ret_kwargs)
image_model = VqganTokenizer()

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        
        
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

def form_prompt(decoder, query, query1, query2, image1, image2, image_type="path"):
    if image_type == "path":
        tokenize_func = lambda x: " ".join(image_model.encode(x).ids)
    elif image_type == "token":
        tokenize_func = lambda x: x
    all_tokens, doc1, doc2 = [], [], []
    if image1 is not None and query1 is not None:
        image1 = tokenize_func(image1)
        doc1 = (
            tokenize_text(query1)
            + [task.cm3_break_ind]
            + tokenize_image(image1)
            + [task.cm3_break_ind]
        )
    if image2 is not None and query2 is not None:
        image2 = tokenize_func(image2)
        doc2 = (
            tokenize_text(query2)
            + [task.cm3_break_ind]
            + tokenize_image(image2)
            + [task.cm3_break_ind]
        )
    query_token = tokenize_text(query) + [task.cm3_break_ind]
    all_tokens = [task.eod] + doc1 + doc2 + query_token
    return all_tokens

def load_model(cfg):
    #global models
    global task 
    global ra_cm3_models
    
    
     
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
    task = tasks.setup_task(cfg.task)

    def _build_model(cfg, task):
        model = task.build_model(cfg.model, True).cuda()
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
        models, _model_args, _task = _load_checkpoint()
        
    ra_cm3_models = models
    
    

# def form_prompts(text, task, args):
#     # np.random.shuffle(ra_docs)
#     text_encoded = torch.cat(
#         [torch.LongTensor([task.eod])]
#         + [torch.LongTensor(task.tokenizer.encode(text).ids + [task.cm3_break_ind])]
#     )
    
#     text_unconditional = torch.LongTensor(
#         [task.eod, task.cm3_sentinel_tokens_ind[0], task.cm3_break_ind]
#     )
#     return text_encoded, text_unconditional


def generate(args):

    print(f"Loading VQ-GAN")
    #image_model = VqganTokenizer()
    output_dir = "/data/home/aiema/gill/evals/gill_vist_outputs_test_2"
   
    vist_image_dir = '/data/home/aiema/gill/evals/sis/val_images/'
    vist_data_path = '/data/home/aiema/gill/evals/sis/val_formatted.json'
    
    decoder = ImageSequenceGenerator(
        ra_cm3_models[0],
        task.source_dictionary,
        None,
        1,
        temperature=1.0,
        cfg_weight=4,
        dtype=torch.bfloat16 if args.dtype == "bf16" else torch.float32,
    )

    decoder.cuda()

    with open(vist_data_path, 'r') as f:
        vist_data = json.load(f)
        story_ids = list(vist_data['annotations'].keys())
        
    text_unconditional = torch.tensor(
        [
            task.eod,
            task.cm3_sentinel_tokens_ind[0],
            task.cm3_break_ind,
        ],
        dtype=torch.long,
    )
    # Calculate the start and end index for each rank
    
    for story_idx, (story_id, story_data) in enumerate(islice(vist_data['annotations'].items(), 0, 500)):
        # Load all images except the last (we're generating the last one)
        image_paths = [os.path.join(vist_image_dir, s['image_id'] + '.png') for s in story_data][:-1]
        gt_image_id = story_data[-1]['image_id']
        captions = [s['caption'] for s in story_data]
        assert (len(image_paths) == len(captions) - 1) or (len(image_paths) == len(captions))

        should_process = True
        for path in image_paths:
            if not os.path.exists(path):
                print(f'Image not found: {path}. Skipping story {story_id}')
                should_process = False
                break
        
        if should_process:
            caption_range = range(len(captions))
            input_data = ''

            for i_i, i in enumerate(caption_range):
                caption = captions[i]
                input_data += caption

                # if i < len(captions) - 1:  # Use first n-1 images
                #     with open(image_paths[i], 'rb') as f:
                #         img = Image.open(f).convert('RGB').resize((224, 224))
                #         input_data.append(img)

            # Print outputs for first 3 examples as a sanity check.
            if story_idx < 3:
                print(input_data)

            result_dicts = []
            #ret_query = input_data[:77]
            
            ret_docs = retriever.search(input_data)
            query1, image1 = parse_doc(ret_docs[0])
            query2, image2 = parse_doc(ret_docs[1])
            image_type = "token"
            
                
            text_encoded = torch.tensor(
                form_prompt(
                    decoder, input_data, query1, query2, image1, image2, image_type=image_type
                ),
                dtype=torch.long,
            )
            result_dict = decoder(
                text_encoded.unsqueeze(0), text_unconditional.unsqueeze(0)
            )
            result_dicts.append(result_dict)

            tokens, scores = torch.cat(
                [result_dict["tokens"] for result_dict in result_dicts], dim=1
            ), torch.cat([result_dict["scores"] for result_dict in result_dicts], dim=1)
            tokens, scores = tokens.view(-1, tokens.size(2)), scores.view(
                -1, scores.size(2)
            )
                
        for i, image_beam in enumerate(tokens.cpu().numpy().tolist()):
            image = task.tokenizer.decode(image_beam, skip_special_tokens=False)
            image = extract_image_tokens(image)
            image = image[:1024]
            os.makedirs(os.path.dirname(os.path.join(output_dir, f'{gt_image_id}.png')), exist_ok=True)
            image_model.decode(image).save(os.path.join(output_dir, f'{gt_image_id}.png'))

    return 


def worker_main(cfg: MetaseqConfig, args=None):
    #os.environ["TOKENIZERS_PARALLELISM"] = "false"
    #torch.set_num_threads(8)
    global model
    # make sure generations are stochastic since we have many workers
    torch.manual_seed(random.randint(1, 20000))
    torch.cuda.manual_seed(random.randint(1, 20000))
    model = load_model(cfg)
    generate(args)
   
    

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
    parser.add_argument(
        "--ckpt",
        type=str,
        default="/data/cm3z/armenag/code/ra_cm3/models/760m/maiden_760_v0/checkpoint_70000_consolidated.pt",
        help="consolidated checkpoint",
    )
    parser.add_argument(
        "--jsonl",
        type=str,
        default="/data/cm3z/armenag/datasets/ra_cm3/fsdp-format/valid/shutterstock/00/00.jsonl",
        help="valid jsonl path",
    )
    
    parser.add_argument(
        "--contrastive", action="store_true", help="contrastive decoding"
    )
    parser.add_argument("--temp_student", type=float, default=0.5, help="temperature")
    parser.add_argument("--alpha", type=float, default=0.5, help="temperature")
    parser.add_argument("--beam-size", type=int, default=32, help="beam size")
    parser.add_argument(
        "--cfg-weight", type=float, default=5, help="classifier-free-guidance weight"
    )
    parser.add_argument(
        "--num-retrieved-doc", type=int, default=2, help="number of retrieved documents"
    )
    parser.add_argument("--output-dir", type=str, default=None, help="output directory")
    parser.add_argument("--nshard", type=int, default=100, help="number of shards")
    parser.add_argument("--slurm", action="store_true", help="running on slurm")
    parser.add_argument("--partition", type=str, default="cm3z", help="slurm partition")
    parser.add_argument(
        "--mscoco-val",
        type=str,
        default="/data/cm3z/bshi/datasets/ra_cm3/mscoco/gt_images/michi-tokenize-valid/",
        help="mscoco ground-truth dir",
    )
    parser.add_argument("--clip-rerank", action="store_true", help="reranking")
    parser.add_argument(
        "--nshard-per-beam", type=int, default=1, help="number of shards per beam"
    )
    parser.add_argument(
        "--retrieve-source",
        type=str,
        default="text",
        help="define if we retrieve using text or text+image",
    )
    parser.add_argument(
        "--n_examples",
        type=int,
        default=-1,
        help="number of examples, set it smaller for faster compute",
    )
    parser.add_argument("--dtype", type=str, default="fp32")

    #args = parser.parse_args()

    args = options.parse_args_and_arch(parser, input_args=flat_launch_args)
    args.data = os.path.dirname(args.path)  # hardcode the data arg
    cfg = convert_namespace_to_omegaconf(args)
    cfg.distributed_training.distributed_world_size = TOTAL_WORLD_SIZE

    model_overrides = ast.literal_eval(cfg.common_eval.model_overrides)
    model_overrides.update(INFERENCE_ARG_OVERRIDES)
    cfg.common_eval.model_overrides = str(model_overrides)

    distributed_utils.call_main(cfg, worker_main, args=args)


if __name__ == "__main__":
    if os.getenv("SLURM_NODEID") is None:
        logger.warning(
            f"Missing slurm configuration, defaulting to 'use entire node' for API"
        )
        os.environ["SLURM_NODEID"] = "0"
        os.environ["SLURM_NNODES"] = "1"
        os.environ["SLURM_NTASKS"] = "8"
        import socket

        os.environ["SLURM_STEP_NODELIST"] = socket.gethostname()
    cli_main()
