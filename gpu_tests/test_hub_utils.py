# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os

import numpy as np
import torch.distributed as dist
from numpy.random import RandomState

from metaseq.dataclass.configs import MetaseqConfig
from metaseq.distributed import utils as distributed_utils
from metaseq.hub_utils import GeneratorInterface
from metaseq.modules.megatron.mpu import destroy_model_parallel
from metaseq.scripts.convert_to_singleton import create_generation_config_with_defaults

PROMPT = [133, 313, 1224, 15, 5, 856, 17527, 594, 98, 5, 11471, 3820, 19, 514, 4]


def generate_using_generator_interface(cfg: MetaseqConfig, **kwargs):
    generator = GeneratorInterface(cfg)
    models = generator.load_model()  # noqa: F841

    request_object = {
        "inputs": [PROMPT],
        "temperature": 1.0,
        "max_tokens": [0],
        "min_tokens": [0],
        "top_p": 1.0,
        "n": 1,
        "best_of": 1,
        "echo": True,
        "logprobs": 0,
        "seed": 1,
    }

    generated_text = generator.generate(**request_object)

    # Destroy torch distributed process groups. This needs to be executed in each spawned process
    # https://github.com/pytorch/pytorch/issues/48203
    dist.destroy_process_group()
    return generated_text


# TEST FUNCTIONS #
def test_generator_interface(data_regression, num_regression):
    model_path = os.path.join(os.path.dirname(__file__), "125m")
    cfg = create_generation_config_with_defaults(
        model_path, ddp_backend="fully_sharded"
    )

    overall_generation = distributed_utils.call_main(
        cfg, generate_using_generator_interface
    )
    # We can potentially pass a list of inputs to the generate function.
    # Here, we look at the first (and only) generation
    [generation_for_prompt] = overall_generation
    # We use best_of = 1, so we get only one beam search result
    [generated_beam] = generation_for_prompt

    ndarray_data = {
        "token_scores": np.array(
            [
                np.nan if elem is None else elem
                for elem in generated_beam["token_scores"]
            ]
        )
    }
    generated_beam.pop("token_scores")

    num_regression.check(ndarray_data, default_tolerance=dict(atol=1e-2))
    data_regression.check(generated_beam)


def test_filter_special(data_regression):
    tokens = [123, 453, 653, 2, 345, 453]
    # Assuming a vocab size of 10
    vocab_size = 10

    prng = RandomState(1234567890)

    scores = prng.randn(len(tokens)).tolist()

    distributions = prng.randn(len(tokens), vocab_size)
    new_tokens, new_scores, distributions = GeneratorInterface._filter_special(
        pad_token_ind=1,
        special_token_inds=[0, 1, 2, 3],
        tokens=tokens,
        scores=scores,
        distributions=distributions,
    )

    # Since we got a special token at index 3, only the first three tokens should be returned.
    assert len(new_tokens) == 3
    assert len(new_scores) == 3
    data_regression.check(new_tokens, "test_filter_special_new_tokens")
    data_regression.check(new_scores, "test_filter_special_new_scores")
    data_regression.check(distributions.tolist(), "test_filter_special_distributions")


def teardown_function():
    # Tear down model parallel
    destroy_model_parallel()
    distributed_utils._USE_MEGATRON = False
