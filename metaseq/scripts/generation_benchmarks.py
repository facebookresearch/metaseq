import os
import json
import time
from transformers import GPT2Tokenizer
from metaseq import checkpoint_utils, tasks, utils
import torch
import torch.nn.functional as F
from metaseq.scripts.convert_to_singleton import create_generation_config_with_defaults
from metaseq.distributed import utils as dist_utils
from metaseq.distributed import fsdp_enable_wrap, fsdp_wrap
from metaseq.dataclass.configs import MetaseqConfig
import argparse
import urllib.request
import tarfile
import shutil

prompts = []


def load_prompts(path, max_len=10, max_sentences=20):
    all_data = list(open(path, "r"))
    for data in all_data:
        json_data = json.loads(data)
        s = " ".join(json_data["text"].split()[:max_len])
        prompts.append(s)
        if len(prompts) == max_sentences:
            break


def setup_vocab_and_merges(model_path):
    vocab_file = os.path.join(model_path, "gpt2-vocab.json")
    merges_file = os.path.join(model_path, "gpt2-merges.txt")
    tokenizer = GPT2Tokenizer(vocab_file, merges_file)
    tokenizer.save_pretrained(model_path)
    return vocab_file, merges_file, tokenizer


def load_mp_model_and_run_eval(cfg: MetaseqConfig, **kwargs):
    _, _, tokenizer = setup_vocab_and_merges(kwargs["model_path"])
    load_prompts(kwargs["prompt_path"])
    prompt_ids = []
    thread_times = [
        torch.zeros(1).cuda() for _ in range(dist_utils.get_model_parallel_world_size())
    ]
    begin = time.time()
    for prompt in prompts:
        input_ids = tensorize_input(tokenizer, prompt)
        input_ids = F.pad(
            input=input_ids,
            pad=(0, kwargs["padding_size"] - input_ids.shape[1], 0, 0),
            value=1,
        )
        prompt_ids.append(input_ids)
    end = time.time()

    prompt_ids = torch.cat(prompt_ids).cuda()
    prompt_loading = end - begin

    task = tasks.setup_task(cfg.task)

    def _build_model(cfg, task):
        # hardcoded to cpu & fp16
        # Hardcoded tensor_parallel_init_model_on_gpu to be True
        cfg.model.tensor_parallel_init_model_on_gpu = True
        model = task.build_model(cfg.model).half().cuda()
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

    begin = time.time()
    with torch.no_grad():
        model(prompt_ids)[0]

    end = time.time()
    times_list = torch.tensor(end - begin + prompt_loading).cuda()

    torch.distributed.all_gather(
        thread_times, times_list, group=dist_utils.get_global_group()
    )
    torch.save(thread_times, "./dependencies/times.pt")


def tensorize_input(tokenizer, prompts):
    input_ids = tokenizer(prompts, return_tensors="pt").input_ids
    input_ids = torch.cat([torch.tensor([[0]]), input_ids], dim=-1)
    input_ids = input_ids
    return input_ids


def generation_statistics_single(
    prompt_path, model_path, padding_size, item_name="restored.pt"
):
    vocab_file, merges_file, tokenizer = setup_vocab_and_merges(model_path)
    load_prompts(prompt_path)

    checkpoint = checkpoint_utils.load_model_ensemble_and_task(
        [os.path.join(model_path, item_name)],
        arg_overrides={
            "vocab_filename": vocab_file,
            "merges_filename": merges_file,
        },
    )

    model = checkpoint[0][0].eval()
    model = model.cuda()
    prompt_ids = []
    begin = time.time()
    for prompt in prompts:
        input_ids = tensorize_input(tokenizer, prompt)
        input_ids = F.pad(
            input=input_ids, pad=(0, padding_size - input_ids.shape[1], 0, 0), value=1
        )
        prompt_ids.append(input_ids)

    prompt_ids = torch.cat(prompt_ids).cuda()

    with torch.no_grad():
        model(prompt_ids)[0]

    end = time.time()
    total_gen_time = end - begin

    print("Total generation time " + str(total_gen_time))
    print("Words per second : " + str((len(prompts) * padding_size) / total_gen_time))


def generation_statistics(prompt_path, model_path, padding_size):

    cfg = create_generation_config_with_defaults(model_path)
    dist_utils.call_main(
        cfg,
        load_mp_model_and_run_eval,
        model_path=model_path,
        padding_size=padding_size,
        prompt_path=prompt_path,
        repeat=20,
    )

    thread_times = torch.load("./dependencies/times.pt")
    avg_total_gen_time = sum(thread_times).item() / len(thread_times)

    print("Total generation time " + str(avg_total_gen_time))
    print(
        "Words per second : " + str((len(prompts) * padding_size) / avg_total_gen_time)
    )


def download_data(key):
    links_to_data = {}
    links_to_data["125m"] = [
        "https://dl.fbaipublicfiles.com/opt/v1_20220502/125m/reshard-model_part-0.pt",
        "https://dl.fbaipublicfiles.com/opt/v1_20220502/125m/reshard-model_part-1.pt",
    ]
    links_to_data["350m"] = [
        "https://dl.fbaipublicfiles.com/opt/v1_20220502/350m/reshard.pt"
    ]
    links_to_data["1.3b"] = [
        "https://dl.fbaipublicfiles.com/opt/v1_20220502/1.3b/reshard-model_part-0.pt",
        "https://dl.fbaipublicfiles.com/opt/v1_20220502/1.3b/reshard-model_part-1.pt",
    ]
    links_to_data["2.7b"] = [
        "https://dl.fbaipublicfiles.com/opt/v1_20220502/2.7b/reshard-model_part-0.pt",
        "https://dl.fbaipublicfiles.com/opt/v1_20220502/2.7b/reshard-model_part-1.pt",
        "https://dl.fbaipublicfiles.com/opt/v1_20220502/2.7b/reshard-model_part-2.pt",
        "https://dl.fbaipublicfiles.com/opt/v1_20220502/2.7b/reshard-model_part-3.pt",
    ]
    links_to_data["6.7b"] = [
        "https://dl.fbaipublicfiles.com/opt/v1_20220502/6.7b/reshard-model_part-0.pt",
        "https://dl.fbaipublicfiles.com/opt/v1_20220502/6.7b/reshard-model_part-1.pt",
    ]
    links_to_data["13b"] = [
        "https://dl.fbaipublicfiles.com/opt/v1_20220502/13b/reshard-model_part-0.pt",
        "https://dl.fbaipublicfiles.com/opt/v1_20220502/13b/reshard-model_part-1.pt",
    ]
    links_to_data["30b"] = [
        "https://dl.fbaipublicfiles.com/opt/v1_20220502/30b/reshard-model_part-0.pt",
        "https://dl.fbaipublicfiles.com/opt/v1_20220502/30b/reshard-model_part-1.pt",
    ]
    links_to_data["66b"] = [
        "https://dl.fbaipublicfiles.com/opt/v1_20220502/66b/reshard-model_part-0-shard0.pt",
        "https://dl.fbaipublicfiles.com/opt/v1_20220502/66b/reshard-model_part-1-shard0.pt",
        "https://dl.fbaipublicfiles.com/opt/v1_20220502/66b/reshard-model_part-2-shard0.pt",
        "https://dl.fbaipublicfiles.com/opt/v1_20220502/66b/reshard-model_part-3-shard0.pt",
        "https://dl.fbaipublicfiles.com/opt/v1_20220502/66b/reshard-model_part-4-shard0.pt",
        "https://dl.fbaipublicfiles.com/opt/v1_20220502/66b/reshard-model_part-5-shard0.pt",
        "https://dl.fbaipublicfiles.com/opt/v1_20220502/66b/reshard-model_part-6-shard0.pt",
        "https://dl.fbaipublicfiles.com/opt/v1_20220502/66b/reshard-model_part-7-shard0.pt",
    ]

    if key not in links_to_data.keys():
        raise AssertionError(
            "You passed model name "
            + str(key)
            + ". Please make sure that model name is in "
            + str(list(links_to_data.keys()))
        )

    links_to_data["dependencies"] = [
        "http://dl.fbaipublicfiles.com/metaseq_benchmark_dependencies.tar.gz"
    ]

    # download and unzip dependencies
    for dependency in links_to_data["dependencies"]:
        urllib.request.urlretrieve(dependency, "./dependencies.tar.gz")
        file = tarfile.open("./dependencies.tar.gz")
        file.extractall()

    # download model checkpoint
    for shard in links_to_data[key]:
        file_name = shard.split("/")[-1]
        urllib.request.urlretrieve(shard, "./dependencies/" + file_name)


def download_prompts():
    link_to_prompt = "https://dl.fbaipublicfiles.com/CommonCrawlSmall.jsonl"
    urllib.request.urlretrieve(link_to_prompt, "./CommonCrawlSmall.jsonl")
    pass


def cleanup(prompts):
    shutil.rmtree("./dependencies")
    os.remove("./dependencies.tar.gz")
    os.remove(prompts)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", nargs="+")
    parser.add_argument("--prompt_path", nargs="?")
    parser.add_argument("--padding_size", nargs="?", default="512")
    args = parser.parse_args()

    if args.prompt_path is None:
        # using default prompts
        download_prompts()
        args.prompt_path = "./CommonCrawlSmall.jsonl"

    for key in args.model_name:
        print("Now running generation benchmarks for " + str(key))
        download_data(key)
        if key in ["350m"]:
            # single shard
            generation_statistics_single(
                prompt_path=args.prompt_path,
                model_path="./dependencies",
                padding_size=int(args.padding_size),
                item_name="reshard.pt",
            )
        else:
            generation_statistics(
                prompt_path=args.prompt_path,
                model_path="./dependencies",
                padding_size=int(args.padding_size),
            )
        cleanup(args.prompt_path)
