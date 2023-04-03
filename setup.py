#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys

from setuptools import Extension, find_packages, setup
from torch.utils.cpp_extension import CppExtension, BuildExtension


if sys.version_info < (3, 6):
    sys.exit("Sorry, Python >= 3.6 is required for metaseq.")


def write_version_py():
    with open(os.path.join("metaseq", "version.txt")) as f:
        version = f.read().strip()

    # append latest commit hash to version string
    try:
        sha = (
            subprocess.check_output(["git", "rev-parse", "HEAD"])
            .decode("ascii")
            .strip()
        )
        version += "+" + sha[:7]
    except Exception:
        pass

    # write version info to metaseq/version.py
    with open(os.path.join("metaseq", "version.py"), "w") as f:
        f.write('__version__ = "{}"\n'.format(version))
    return version


version = write_version_py()

with open("README.md") as f:
    readme = f.read()

if sys.platform == "darwin":
    extra_compile_args = ["-stdlib=libc++", "-O3"]
else:
    extra_compile_args = ["-std=c++11", "-O3"]


class NumpyExtension(Extension):
    """Source: https://stackoverflow.com/a/54128391"""

    def __init__(self, *args, **kwargs):
        self.__include_dirs = []
        super().__init__(*args, **kwargs)

    @property
    def include_dirs(self):
        import numpy

        return self.__include_dirs + [numpy.get_include()]

    @include_dirs.setter
    def include_dirs(self, dirs):
        self.__include_dirs = dirs


extension_modules = [
    NumpyExtension(
        "metaseq.data.data_utils_fast",
        sources=["metaseq/data/data_utils_fast.pyx"],
        language="c++",
        extra_compile_args=extra_compile_args,
    ),
    NumpyExtension(
        "metaseq.data.token_block_utils_fast",
        sources=["metaseq/data/token_block_utils_fast.pyx"],
        language="c++",
        extra_compile_args=extra_compile_args,
    ),
    # Reference:
    # https://github.com/ngoyal2707/Megatron-LM/commit/9a16189ab1b5537205c708f8c8f952f2ae2ae72b
    # TODO[susanz]: Figure out where cflags and cuda_flags should go.
    CppExtension(
        "metaseq.modules.megatron.fused_kernels.scaled_upper_triang_masked_softmax_cuda",
        sources=[
            "metaseq/modules/megatron/fused_kernels/scaled_upper_triang_masked_softmax.cpp",
            "metaseq/modules/megatron/fused_kernels/scaled_upper_triang_masked_softmax_cuda.cu",
        ],
        # extra_cflags=['-O3',],
        # extra_cuda_flags=['-O3', '--use_fast_math',
        #     "-U__CUDA_NO_HALF_OPERATORS__",
        #     "-U__CUDA_NO_HALF_CONVERSIONS__",
        #     "--expt-relaxed-constexpr",
        #     "--expt-extended-lambda",
        # ],
    ),
    CppExtension(
        "metaseq.modules.megatron.fused_kernels.scaled_masked_softmax_cuda",
        sources=[
            "metaseq/modules/megatron/fused_kernels/scaled_masked_softmax.cpp",
            "metaseq/modules/megatron/fused_kernels/scaled_masked_softmax_cuda.cu",
        ],
        # extra_cflags=['-O3',],
        # extra_cuda_flags=['-O3', '--use_fast_math',
        #     "-U__CUDA_NO_HALF_OPERATORS__",
        #     "-U__CUDA_NO_HALF_CONVERSIONS__",
        #     "--expt-relaxed-constexpr",
        #     "--expt-extended-lambda",
        # ],
    ),
]


if "clean" in sys.argv[1:]:
    # Source: https://bit.ly/2NLVsgE
    print("deleting Cython files...")
    import subprocess

    subprocess.run(
        ["rm -f metaseq/*.so metaseq/**/*.so metaseq/*.pyd metaseq/**/*.pyd"],
        shell=True,
    )


def do_setup():
    setup(
        name="metaseq",
        version=version,
        description="MetaSeq, a framework for large language models, from Meta",
        url="https://github.com/facebookresearch/metaseq",
        long_description=readme,
        long_description_content_type="text/markdown",
        setup_requires=[
            "cython",
            'numpy; python_version>="3.7"',
            "setuptools>=18.0",
        ],
        install_requires=[
            # protobuf version pinned due to tensorboard not pinning a version.
            #  https://github.com/protocolbuffers/protobuf/issues/10076
            "protobuf==3.20.2",
            "aim>=3.9.4",
            "azure-storage-blob",
            "boto3",
            "black==22.3.0",
            "click==8.0.4",
            "cython",
            'dataclasses; python_version<"3.7"',
            "editdistance",
            "fire",
            "flask==2.1.1",  # for api
            "hydra-core>=1.1.0,<1.2",
            "ipdb",
            "ipython",
            "Jinja2==3.1.1",  # for evals
            "markupsafe",  # for evals
            "more_itertools",
            "mypy",
            "ninja",
            'numpy; python_version>="3.7"',
            "omegaconf<=2.1.1",
            "portalocker>=2.5",
            "pre-commit",
            "pytest",
            "pytest-regressions",
            "regex",
            "scikit-learn",  # for evals
            "sacrebleu",  # for evals
            "tensorboard==2.8.0",
            "timeout-decorator",
            "tokenizers",
            "torch",
            "tqdm",
            "typing_extensions",
        ],
        packages=find_packages(
            exclude=[
                "scripts",
                "scripts.*",
                "tests",
                "tests.*",
            ]
        ),
        include_package_data=True,
        zip_safe=False,
        extras_require={
            # install via: pip install -e ".[dev]"
            "dev": [
                "flake8",
                "black==22.3.0",
                # test deps
                "iopath",
                "transformers",
                "pyarrow",
                "boto3",
                "pandas",
            ],
            # install via: pip install -e ".[multimodal]"
            "multimodal": [
                "albumentations",
                "dalle_pytorch",
                "einops",
                "matplotlib==3.5.0",
                "pytorchvideo==0.1.5",
                "wandb",
                "webdataset==0.1.103",
            ],
        },
        ext_modules=extension_modules,
        test_suite="tests",
        entry_points={
            "console_scripts": [
                "metaseq-train = metaseq.cli.train:cli_main",
                "metaseq-validate = metaseq.cli.validate:cli_main",
                "opt-baselines = metaseq.launcher.opt_baselines:cli_main",
                "metaseq-api-local = metaseq.cli.interactive_hosted:cli_main",
            ],
        },
        cmdclass={"build_ext": BuildExtension},
    )


if __name__ == "__main__":
    do_setup()
