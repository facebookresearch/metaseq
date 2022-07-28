import os
from transformers import GPT2Tokenizer
from metaseq import checkpoint_utils, tasks, utils
import torch
from metaseq.scripts.convert_to_singleton import create_generation_config_with_defaults
from metaseq.distributed import utils as dist_utils
from metaseq.distributed import fsdp_enable_wrap, fsdp_wrap
from metaseq.dataclass.configs import MetaseqConfig
import argparse
import urllib.request
import tarfile
import shutil

prompts = ["Paris is the capital of France and it"] * 1000
links_to_data = {}
links_to_data["125m"] = [
    "https://dl.fbaipublicfiles.com/opt/v1_20220502/125m/reshard-model_part-0.pt",
    "https://dl.fbaipublicfiles.com/opt/v1_20220502/125m/reshard-model_part-1.pt",
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


links_to_data["dependencies"] = [
    "http://dl.fbaipublicfiles.com/metaseq_benchmark_dependencies.tar.gz"
]


def setup_vocab_and_merges(model_path):
    vocab_file = os.path.join(model_path, "gpt2-vocab.json")
    merges_file = os.path.join(model_path, "gpt2-merges.txt")
    tokenizer = GPT2Tokenizer(vocab_file, merges_file)
    tokenizer.save_pretrained(model_path)
    return vocab_file, merges_file, tokenizer


def tensorize_input(tokenizer, prompts):
    input_ids = tokenizer(prompts, return_tensors="pt").input_ids
    input_ids = torch.cat([torch.tensor([[0]]), input_ids], dim=-1)
    input_ids = input_ids
    return input_ids


def load_mp_model_and_run_eval(cfg: MetaseqConfig, **kwargs):
    _, _, tokenizer = setup_vocab_and_merges(kwargs["model_path"])
    prompt_ids = []
    thread_times = [
        torch.zeros(1).cuda() for _ in range(dist_utils.get_model_parallel_world_size())
    ]
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for prompt in prompts:
        input_ids = tensorize_input(tokenizer, prompt)
        prompt_ids.append(input_ids)

    prompt_ids = torch.cat(prompt_ids).cuda()
    end.record()
    # Waits for everything to finish running
    torch.cuda.synchronize()
    prompt_loading = start.elapsed_time(end)

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

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    with torch.no_grad():
        model(prompt_ids)[0]
    end.record()

    # Waits for everything to finish running
    torch.cuda.synchronize()

    total_time = start.elapsed_time(end) + prompt_loading
    # total time is in ms
    times_list = torch.tensor(total_time / 1000).cuda()

    torch.distributed.all_gather(
        thread_times, times_list, group=dist_utils.get_global_group()
    )
    return thread_times


def generation_statistics(model_path):

    cfg = create_generation_config_with_defaults(model_path)
    thread_times = dist_utils.call_main(
        cfg,
        load_mp_model_and_run_eval,
        model_path=model_path,
    )

    avg_total_gen_time = sum(thread_times).item() / len(thread_times)

    print("Total generation time " + str(avg_total_gen_time))
    print(
        "Words per second : "
        + str((len(prompts[0]) * len(prompts)) / avg_total_gen_time)
    )


def download_data(key):
    if key not in links_to_data.keys():
        raise AssertionError(
            "You passed model name "
            + str(key)
            + ". Please make sure that model name is in "
            + str(list(links_to_data.keys()))
        )

    # download and unzip dependencies
    for dependency in links_to_data["dependencies"]:
        urllib.request.urlretrieve(dependency, "./dependencies.tar.gz")
        file = tarfile.open("./dependencies.tar.gz")
        file.extractall()

    # download model checkpoint
    for shard in links_to_data[key]:
        file_name = shard.split("/")[-1]
        urllib.request.urlretrieve(shard, "./dependencies/" + file_name)


def cleanup_data():
    shutil.rmtree("./dependencies")
    os.remove("./dependencies.tar.gz")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", nargs="+")
    args = parser.parse_args()

    if args.model_name is None:
        raise AssertionError("Please pass at least one model name as argument")

    for key in args.model_name:
        print("Now running generation benchmarks for " + str(key))
        download_data(key)
        generation_statistics(
            model_path="./dependencies",
        )
        cleanup_data()
