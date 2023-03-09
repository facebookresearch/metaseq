import argparse
import os
import torch
import torch.distributed as dist
import torch.nn as nn
from tokenizers import Tokenizer, ByteLevelBPETokenizer
from typing import Any, List, Optional

try:
    torch.classes.load_library(os.environ.get("FT_PATH"))
except Exception:
    raise ImportError(
        "Please install FasterTransformer and provide a path to the binary"
        "`libth_transformer.so` via the environment variable `FT_PATH`."
    )

model = None
tokenizer = None
device = None

BOS_TOKEN = 0
PAD_TOKEN = 1
EOS_TOKEN = 2
UNK_TOKEN = 3


@torch.inference_mode()
def generate(
    inputs: List[List[int]],
    output_length: int,
    beam_width: int = 1,
    top_k: Optional[int] = 0,
    top_p: Optional[float] = 1.0,
    diversity_rate: Optional[float] = None,
    temperature: Optional[float] = 1.0,
    len_penalty: Optional[float] = None,
    repetition_penalty: Optional[float] = 1.0,
    presence_penalty: Optional[float] = None,
    random_seed: Optional[int] = 0,
    min_length: Optional[int] = None,
    bad_words_list: Optional[torch.Tensor] = None,
    return_cum_log_probs: Optional[int] = 0,
) -> List[Any]:
    inputs = [[EOS_TOKEN] + toks for toks in inputs]
    inputs = [torch.tensor(toks, dtype=torch.int32, device=device) for toks in inputs]
    lengths = torch.tensor([len(t) for t in inputs], dtype=torch.int32, device=device)
    inputs = nn.utils.rnn.pad_sequence(inputs, True, padding_value=PAD_TOKEN)

    if top_k is not None:
        top_k = torch.tensor([top_k], dtype=torch.int32)
    if top_p is not None:
        top_p = torch.tensor([top_p], dtype=torch.float32)
    if diversity_rate is not None:
        diversity_rate = torch.tensor([diversity_rate], dtype=torch.float32)
    if temperature is not None:
        temperature = torch.tensor([temperature], dtype=torch.float32)
    if len_penalty is not None:
        len_penalty = torch.tensor([len_penalty], dtype=torch.float32)
    if repetition_penalty is not None:
        repetition_penalty = torch.tensor([repetition_penalty], dtype=torch.float32)
    if presence_penalty is not None:
        presence_penalty = torch.tensor([presence_penalty], dtype=torch.float32)
    if random_seed is not None:
        random_seed = torch.tensor([random_seed], dtype=torch.int64)
    if min_length is not None:
        min_length = torch.tensor([min_length], dtype=torch.int64)

    outputs, output_lengths = model.forward(
        inputs,
        lengths,
        output_length,
        beam_width,
        top_k,
        top_p,
        diversity_rate,
        temperature,
        len_penalty,
        repetition_penalty,
        presence_penalty,
        min_length,
        random_seed,
        bad_words_list,
        return_cum_log_probs,
    )

    results = []
    beam_idx = 0
    special = outputs.new_tensor([BOS_TOKEN, PAD_TOKEN, EOS_TOKEN, UNK_TOKEN])
    for output, output_len in zip(outputs, output_lengths):
        mask = ~torch.isin(output[beam_idx], special)
        mask[1:] = mask[1:].cummin(dim=0)[0]

        tokens = output[beam_idx][1 : output_len[beam_idx]]
        tokens = tokens[mask[1 : output_len[beam_idx]]]
        results.append({"text": tokenizer.decode(tokens.tolist())})
    return [results]


def main(args: argparse.Namespace) -> None:
    global model, tokenizer, device
    dist.init_process_group(backend="mpi")
    world_size = dist.get_world_size()
    rank = dist.get_rank() % world_size
    device = torch.device(f"cuda:{dist.get_rank() % torch.cuda.device_count()}")
    torch.cuda.set_device(device)

    if args.tokenizer_file is not None:
        tokenizer = Tokenizer.from_file(args.tokenizer_file)
    else:
        tokenizer = ByteLevelBPETokenizer(args.vocab_file, args.merges_file)

    torch_dtypes = {"fp16": torch.half, "bf16": torch.bfloat16, "fp32": torch.float}
    dtype = torch_dtypes[args.dtype]

    state_dict = torch.load(f"{args.weight_path}/part-{rank}.pt")
    weights = [w.to(device, dtype) for w in state_dict["weights"]]
    int8_weights, int8_scales = [], []
    if args.int8_mode != 0 and {"int8_weights", "int8_scales"} <= state_dict.keys():
        int8_weights = [w.to(device=device) for w in state_dict["int8_weights"]]
        int8_scales = [w.to(device=device) for w in state_dict["int8_scales"]]

    kwargs = {
        "head_num": args.num_heads,
        "size_per_head": args.embed_size // args.num_heads,
        "inter_size": 4 * args.embed_size,
        "layer_num": args.num_layers,
        "expert_num": 0,
        "moe_k": 0,
        "moe_layer_index": [],
        "vocab_size": args.vocab_size,
        "start_id": 2,
        "end_id": 2,
        "tensor_para_size": world_size,
        "pipeline_para_size": 1,
        "int8_mode": args.int8_mode,
        "layernorm_eps": 1e-5,
        "layernorm_type": "pre_layernorm",
        "activation_type": "Relu",
        "has_positional_encoding": True,
        "has_pre_decoder_layernorm": False,
        "has_post_decoder_layernorm": True,
        "has_adapters": False,
        "adapter_inter_size": 0,
        "use_attention_linear_bias": False,
        "weights": weights,
        "int8_weights": int8_weights,
        "scale": int8_scales,
        "shared_contexts_ratio": 1.0,
    }
    model = torch.classes.FasterTransformer.ParallelGptOp(*kwargs.values())

    object = [None]
    while True:
        if torch.distributed.get_rank() == 0:
            prompt = input("\033[32mPrompt: \033[0;1m").rstrip()
            if not prompt:
                continue
            object = [[tokenizer.encode(prompt).ids]]

        dist.broadcast_object_list(object, src=0)
        output = generate(
            object[0],
            output_length=args.output_length,
            beam_width=args.beam_width,
            top_k=args.top_k,
            top_p=args.top_p,
            diversity_rate=args.diversity_rate,
            temperature=args.temperature,
            len_penalty=args.len_penalty,
            repetition_penalty=args.repetition_penalty,
            random_seed=0,
        )
        if torch.distributed.get_rank() == 0:
            print(f"Output: {output[0][0]['text']}")


def measure_time(func, *args, **kwargs):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    func(*args, **kwargs)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-layers", type=int, default=12)
    parser.add_argument("--num-heads", type=int, default=12)
    parser.add_argument("--embed-size", type=int, default=768)
    parser.add_argument("--vocab-size", type=int, default=50272)

    parser.add_argument("--vocab-file", type=str)
    parser.add_argument("--merges-file", type=str)
    parser.add_argument("--tokenizer-file", type=str, default=None)
    parser.add_argument("--weight-path", type=str)
    parser.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default="fp16")
    parser.add_argument("--int8-mode", type=int, default=0)

    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--output-length", type=int, default=256)
    parser.add_argument("--beam-width", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--len-penalty", type=float, default=0.0)
    parser.add_argument("--diversity-rate", type=float, default=0.0)
    parser.add_argument("--repetition-penalty", type=float, default=1.2)
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    main(args)
