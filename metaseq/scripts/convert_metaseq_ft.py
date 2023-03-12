# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import logging
import os
import re
from glob import glob
from typing import Any, Dict, List, Tuple
import fire
import torch


logging.basicConfig(format="%(asctime)s | %(name)s | %(message)s", level=logging.INFO)
logger: logging.Logger = logging.getLogger("metaseq.scripts.convert_metaseq_ft")


def convert_metaseq_ft(
    input: str,
    output: str,
    dtype: str = "fp16",
    quantize: bool = False,
) -> None:
    """
    Convert Metaseq model weights into FasterTransformer format. The model parallel
    parts in the input are expected to contain unflattened, FSDP-consolidated
    model weights. The number of model parallel parts remains unchanged.

    Args:
        :param input: A glob pattern specifying the path names of the input shards.
            (e.g. "checkpoints/opt-175b/reshard_no_os_unflat/reshard-model_part-*.pt").
        :param output: A string pattern specifying the path names of the output shards.
            Shard indices can be included in the path names if the pattern includes `{i}`.
            (e.g. "checkpoints/opt-175b-ft-mp8/part-{i}.pt").
    """
    torch_dtype = {"fp16": torch.float16, "fp32": torch.float32, "bf16": torch.bfloat16}
    assert dtype in torch_dtype
    dtype = torch_dtype[dtype]

    files = glob(input)
    if len(files) == 0:
        raise ValueError("The glob pattern doesn't match any model parallel parts.")
    files = sorted(files, key=lambda x: list(map(int, re.findall(r"\d+", x))))
    logger.info(f"Found {len(files)} model parallel parts ({files[0]} to {files[-1]})")

    logger.info("Merging embedding tokens across model parallel parts")
    embedding_tokens = torch.cat(
        [torch.load(f)["model"]["decoder.embed_tokens.weight"] for f in files]
    ).to(dtype=dtype)

    for i, file in enumerate(files):
        logger.info(f"Converting {file} into FasterTransformer format")
        state_dict = torch.load(file, torch.device("cpu"))["model"]
        weights = convert_weights(state_dict, embedding_tokens, dtype, quantize)

        output_file = output.format(i=i)
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        logger.info(f"Writing converted weights to {output_file}")
        torch.save(weights, output_file)
    logger.info("Done!")


def convert_weights(
    state_dict: Dict[str, Any],
    embedding_tokens: torch.Tensor,
    dtype: torch.dtype,
    quantize: bool = False,
) -> List[torch.Tensor]:
    regex = re.compile(r"decoder.layers.(\d+).fc1.weight")
    N = max(int(regex.findall(x)[0]) for x in filter(regex.match, state_dict)) + 1

    # fmt: off
    weights = []
    weights.extend([state_dict[f"decoder.layers.{i}.self_attn_layer_norm.weight"].to(dtype) for i in range(N)])
    weights.extend([state_dict[f"decoder.layers.{i}.self_attn_layer_norm.bias"].to(dtype) for i in range(N)])
    weights.extend([_kvq_to_qkv(state_dict[f"decoder.layers.{i}.self_attn.qkv_proj.weight"].to(dtype))for i in range(N)])
    weights.extend([_kvq_to_qkv(state_dict[f"decoder.layers.{i}.self_attn.qkv_proj.bias"].to(dtype))for i in range(N)])
    weights.extend([state_dict[f"decoder.layers.{i}.self_attn.out_proj.weight"].T.contiguous().to(dtype) for i in range(N)])
    weights.extend([state_dict[f"decoder.layers.{i}.self_attn.out_proj.bias"].to(dtype) for i in range(N)])
    weights.extend([state_dict[f"decoder.layers.{i}.final_layer_norm.weight"].T.contiguous().to(dtype) for i in range(N)])
    weights.extend([state_dict[f"decoder.layers.{i}.final_layer_norm.bias"].to(dtype) for i in range(N)])
    weights.extend([state_dict[f"decoder.layers.{i}.fc1.weight"].T.contiguous().to(dtype) for i in range(N)])
    weights.extend([state_dict[f"decoder.layers.{i}.fc1.bias"].to(dtype) for i in range(N)])
    weights.extend([state_dict[f"decoder.layers.{i}.fc2.weight"].T.contiguous().to(dtype) for i in range(N)])
    weights.extend([state_dict[f"decoder.layers.{i}.fc2.bias"].to(dtype) for i in range(N)])
    weights.append(state_dict["decoder.layer_norm.weight"].to(dtype))
    weights.append(state_dict["decoder.layer_norm.bias"].to(dtype))
    weights.append(state_dict["decoder.embed_positions.weight"][2:])
    weights.append(embedding_tokens)  # "model.wte"
    weights.append(embedding_tokens)  # "model.lm_head.weight"
    # fmt: on

    out = {"weights": weights}
    if quantize:
        out["int8_weights"], out["int8_scales"] = int8_weight_only_quantize(weights, N)
    return out


def int8_weight_only_quantize(
    weights: List[torch.Tensor], num_layers: int
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    try:
        torch.classes.load_library(os.environ.get("FT_PATH"))
    except Exception:
        raise ImportError(
            "Please install FasterTransformer and provide a path to the binary"
            "`libth_transformer.so` via the environment variable `FT_PATH`."
        )
    quant = torch.ops.fastertransformer.symmetric_quantize_last_axis_of_batched_matrix

    N = num_layers
    int8_weights, int8_scales = [None] * (N * 4), [None] * (N * 4)
    for i in range(N):
        int8_weights[i + 0 * N], int8_scales[i + 0 * N] = quant(
            weights[2 * N + i].flatten(1, 2), torch.int8
        )
        int8_weights[i + 1 * N], int8_scales[i + 1 * N] = quant(
            weights[4 * N + i], torch.int8
        )
        int8_weights[i + 2 * N], int8_scales[i + 2 * N] = quant(
            weights[8 * N + i], torch.int8
        )
        int8_weights[i + 3 * N], int8_scales[i + 3 * N] = quant(
            weights[10 * N + i], torch.int8
        )

        # Release memory taken by half / full precision weights
        weights[2 * N + i] = weights[2 * N + i].new_empty(0)
        weights[4 * N + i] = weights[4 * N + i].new_empty(0)
        weights[8 * N + i] = weights[8 * N + i].new_empty(0)
        weights[10 * N + i] = weights[10 * N + i].new_empty(0)
        torch.cuda.empty_cache()
    return int8_weights, int8_scales


def _kvq_to_qkv(t: torch.Tensor) -> torch.Tensor:
    t = t.view(3, t.size(0) // 3, *t.size()[1:])
    t = torch.cat([t[2:], t[:2]], dim=0)
    return t if t.ndim == 2 else t.permute(2, 0, 1).contiguous()


if __name__ == "__main__":
    """
    Example usage:
        python convert_metaseq_ft.py \
        --input "/data/checkpoints/opt-175b/reshard_no_os_unflat/reshard-model_part-*.pt" \
        --output "/data/checkpoints/opt-175b-ft-mp8/part-{i}.pt" \
        --dtype fp16
    """
    fire.Fire(convert_metaseq_ft)
