import os

from metaseq import options
from metaseq.dataclass.utils import convert_namespace_to_omegaconf
from metaseq.distributed import utils as distributed_utils
from metaseq.hub_utils import GeneratorInterface
from metaseq.service.utils import build_logger
from metaseq.utils import print_r0

import importlib

if "METASEQ_SERVICE_CONSTANTS_MODULE" not in os.environ:
    constants_module = importlib.import_module("metaseq.service.constants")
else:
    constants_module = importlib.import_module(
        os.environ["METASEQ_SERVICE_CONSTANTS_MODULE"]
    )
TOTAL_WORLD_SIZE = constants_module.TOTAL_WORLD_SIZE
LAUNCH_ARGS = constants_module.LAUNCH_ARGS

logger = build_logger()


def encode_prompts(prompts, generator):
    if isinstance(prompts, str):
        # single string. tokenize and turn it to the single pre-tokenized case
        prompts = [generator.encode_fn(prompts)]
    assert isinstance(prompts, list)
    assert len(prompts) > 0
    if isinstance(prompts[0], str):
        # multi string
        prompts = [generator.encode_fn(p) for p in prompts]
    elif isinstance(prompts[0], int):
        # single pre-tokenized
        prompts = [prompts]
    assert isinstance(prompts[0], list)
    # final case: multi pre-tokenized
    assert len(prompts[0]) > 0
    return prompts


def generation_main(cfg):
    generator = GeneratorInterface(cfg)
    models = generator.load_model()  # noqa: F841

    prompts = ["this is a first test", "this is a sencond one"]
    tokens = encode_prompts(prompts, generator)
    MAX_TOKENS = [128] * len(tokens)
    generations = generator.generate(
        inputs=tokens,
        max_tokens=MAX_TOKENS,
        top_p=0.5,
        echo=True,
    )

    print_r0(generations[0][0]["text"])


if __name__ == "__main__":
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
    distributed_utils.call_main(cfg, generation_main)
