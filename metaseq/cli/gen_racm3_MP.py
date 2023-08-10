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

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


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
    
    


   


def form_prompts(text, ra_docs, task, args):
    # np.random.shuffle(ra_docs)
    ra_docs = ra_docs[: args.num_retrieved_doc]
    logger.info(f"Using {len(ra_docs)} retrieved documents")
    ra_indexes = []
    for ra_doc in ra_docs:
        ra_index = task.tokenize_single_doc(ra_doc, add_eod=False)
        ra_index = torch.LongTensor(ra_index + [task.cm3_break_ind])
        ra_indexes.append(ra_index)

    text_encoded = torch.cat(
        [torch.LongTensor([task.eod])]
        + ra_indexes
        + [torch.LongTensor(task.tokenizer.encode(text).ids + [task.cm3_break_ind])]
    )
    if args.num_retrieved_doc > 0:
        text_unconditional = torch.cat([torch.LongTensor([task.eod])] + ra_indexes)
    else:
        text_unconditional = torch.LongTensor(
            [task.eod, task.cm3_sentinel_tokens_ind[0], task.cm3_break_ind]
        )
    return text_encoded, text_unconditional


def generate(args, rank, nshard):

    clip_results = []
    if args.clip_rerank:
        clip_scorer = OpenClip("coca-mscoco-L")
    print(f"saving generated images to {args.output_dir}")
    print(f"Loading model from {args.ckpt}")
    # models, _, task = checkpoint_utils.load_model_ensemble_and_task(
    #     filenames=[args.ckpt],
    #     arg_overrides={
    #         "checkpoint_activations": False,
    #     },
    #     joint = True #Load the cross attention model
    # )

    print(f"Loading VQ-GAN")
    image_model = VqganTokenizer()

    lines = [json.loads(line) for line in open(args.jsonl)]
    if args.n_examples > 0:
        lines = lines[: args.n_examples]
    num_per_shard = (len(lines) + nshard - 1) // nshard
    line_indexes = list(range(0, len(lines)))
    lines = lines[num_per_shard * rank : num_per_shard * (rank + 1)]
    line_indexes = line_indexes[num_per_shard * rank : num_per_shard * (rank + 1)]

    # print(line_indexes)
    for line_index, line in zip(line_indexes, lines):
        exists = [
            os.path.exists(f"{args.output_dir}/{i}/{line_index}.jpg")
            for i in range(args.beam_size)
        ]
        # if sum(exists) == len(exists):
        if os.path.exists(f"{args.output_dir}/0/{line_index}.jpg"):
            print(f"line {line_index} already generated")
            line_indexes.remove(line_index)
            lines.remove(line)
        else:
            print(f"line {line_index} need to be generated")
    print(f"total number of generation is {len(line_indexes)}")
    print(line_indexes)

    beam_size_per_shard = (
        args.beam_size + args.nshard_per_beam - 1
    ) // args.nshard_per_beam
    args.temperature = 1.0

    if args.contrastive:
        decoder = SelfContrastiveImageSequenceGenerator(
            ra_cm3_models[0],
            task.source_dictionary,
            None,
            beam_size=beam_size_per_shard,
            temperature=args.temperature,
            temperature_cfg=args.temp_student,
            alpha=args.alpha,
            dtype=torch.bfloat16 if args.dtype == "bf16" else torch.float32,
        )
    else:
        decoder = ImageSequenceGenerator(
            ra_cm3_models[0],
            task.source_dictionary,
            None,
            beam_size_per_shard,
            temperature=args.temperature,
            cfg_weight=args.cfg_weight,
            dtype=torch.bfloat16 if args.dtype == "bf16" else torch.float32,
        )

    decoder.cuda()

    for line_index, line in tqdm(zip(line_indexes, lines)):
        logger.info(f"Generate line {line_index}")
        print(f"Generate line {line_index}")

        doc, ra_docs_image, ra_docs_txt = (
            line["text"],
            line["retrieved_docs_from_img"],
            line["retrieved_docs_from_txt"],
        )

        doc = parse_doc(doc)
        text, image = doc["text"], doc["image"]

        ra_docs = []
        if "text" in args.retrieve_source:
            ra_docs += ra_docs_txt
        if "image" in args.retrieve_source:
            ra_docs += ra_docs_image

        ## TODO: add randomization within each shard per beam
        result_dicts = []
        for i in range(0, args.beam_size, beam_size_per_shard):
            text_encoded, text_unconditional = form_prompts(text, ra_docs, task, args)
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

        if args.clip_rerank:
            tmp_dir = tempfile.mkdtemp()
            output_fns = [f"{tmp_dir}/{x}.jpg" for x in range(args.beam_size)]
        else:
            output_fns = [
                f"{args.output_dir}/{x}/{line_index}.jpg" for x in range(args.beam_size)
            ]

        for i, image_beam in enumerate(tokens.cpu().numpy().tolist()):
            image = task.tokenizer.decode(image_beam, skip_special_tokens=False)
            image = extract_image_tokens(image)
            image = image[:1024]
            # output_fn = f"{args.output_dir}/{i}/{line_index}.jpg"
            output_fn = output_fns[i]
            os.makedirs(os.path.dirname(output_fn), exist_ok=True)
            image_model.decode(image).save(output_fn)
        if args.clip_rerank:
            scores = clip_scorer.get_scores(output_fns, [text])
            prev_indexes = np.argsort(scores.reshape(-1)).tolist()[::-1]
            for i, prev_index in enumerate(prev_indexes):
                final_output_fn = f"{args.output_dir}/{i}/{line_index}.jpg"
                os.makedirs(os.path.dirname(final_output_fn), exist_ok=True)
                shutil.copyfile(output_fns[prev_index], final_output_fn)
                clip_results.append(
                    {
                        "caption": text,
                        "image": final_output_fn,
                        "clip_score": str(scores.reshape(-1)[prev_index]),
                        "rank": i,
                    }
                )
            shutil.rmtree(tmp_dir)
    return clip_results


def compute_fid(path1, path2, output_fn):
    os.makedirs(os.path.dirname(output_fn), exist_ok=True)
    cmd = f"python -m pytorch_fid {path1} {path2} --batch-size 128 > {output_fn}"
    subprocess.check_output(cmd, shell=True)
    lns = open(output_fn).readlines()
    assert len(lns) >= 1, f"FID output empty"
    fid = float(lns[-1].strip().split(":")[-1])
    return fid

def worker_main(cfg: MetaseqConfig, args=None):
    #os.environ["TOKENIZERS_PARALLELISM"] = "false"
    torch.set_num_threads(8)
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
        model_short_name = (
            args.ckpt if args.ckpt in model_factory else args.ckpt.split("/")[-1]
        )
        args.output_dir = f"{args.output_dir}model-{model_short_name}_cfg{args.cfg_weight}_t{args.temperature}_b{args.beam_size}_clip{args.clip_rerank}_r{args.retrieve_source}"
        if args.contrastive:
            args.output_dir += f"_st{args.temp_student}_alpha{args.alpha}"
        if args.n_examples > 0:
            args.output_dir += f"_n{args.n_examples}"

        # if args.ckpt in model_factory:
        #     args.ckpt = model_factory[args.ckpt]
        args.ckpt = "/fsx-llm/aiema/checkpoints/cross/consolidated/consolidated_5960_demo.pt"
        os.makedirs(args.output_dir, exist_ok=True)
        gen_dirs = [f"{args.output_dir}/{beam}" for beam in range(args.beam_size)]
        gt_dirs = [args.mscoco_val for _ in range(args.beam_size)]
        result_fns = [f"{args.output_dir}/fid-{beam}.txt" for beam in range(args.beam_size)]
        
        logger.info(f"Generate locally")
        generate(args, rank=0, nshard=1)
        logger.info(f"Computing FID")
        fids = []
        for gen_dir, gt_dir, result_fn in zip(gen_dirs, gt_dirs, result_fns):
            fid = compute_fid(gen_dir, gt_dir, result_fn)
            fids.append(fid)
      
    else:
        args.ckpt = "/fsx-llm/aiema/checkpoints/cross/consolidated/consolidated_5960_demo.pt"
        logger.info(f"Looping engaged!")
        generate(args, rank=0, nshard=1)        



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
        default="/fsx-onellm/mingdachen/from-lili/mscoco_racm3_data/text-mscoco-i30000-c1.jsonl",
        help="valid jsonl path",
    )
    #parser.add_argument("--temperature", type=float, default=1, help="temperature")
    parser.add_argument(
        "--contrastive", action="store_true", help="contrastive decoding"
    )
    parser.add_argument("--temp_student", type=float, default=0.5, help="temperature")
    parser.add_argument("--alpha", type=float, default=0.5, help="temperature")
    parser.add_argument("--beam-size", type=int, default=2, help="beam size")
    parser.add_argument(
        "--cfg-weight", type=float, default=4, help="classifier-free-guidance weight"
    )
    parser.add_argument(
        "--num-retrieved-doc", type=int, default=2, help="number of retrieved documents"
    )
    parser.add_argument("--output-dir", type=str, default="/data/home/aiema/test_FID/", help="output directory")
    parser.add_argument("--nshard", type=int, default=8, help="number of shards")
    parser.add_argument("--slurm", action="store_true", help="running on slurm")
    parser.add_argument("--partition", type=str, default="cm3z", help="slurm partition")
    parser.add_argument(
        "--mscoco-val",
        type=str,
        default="fsx-onellm/mingdachen/from-lili/mscoco_racm3_data/tokenizer-exp/multigen-256x256-N30k-raw/",
        help="mscoco ground-truth dir",
    )
    parser.add_argument("--clip-rerank", action="store_false", help="reranking")
    parser.add_argument(
        "--nshard-per-beam", type=int, default=4, help="number of shards per beam"
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
        default=10,
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
