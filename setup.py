#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys

from setuptools import Extension, find_packages, setup

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


extensions = [
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
]


cmdclass = {}


try:
    # torch is not available when generating docs
    from torch.utils import cpp_extension

    cmdclass["build_ext"] = cpp_extension.BuildExtension

except ImportError:
    pass


if "READTHEDOCS" in os.environ:
    # don't build extensions when generating docs
    extensions = []
    if "build_ext" in cmdclass:
        del cmdclass["build_ext"]

    # use CPU build of PyTorch
    dependency_links = [
        "https://download.pytorch.org/whl/cpu/torch-1.7.0%2Bcpu-cp36-cp36m-linux_x86_64.whl"
    ]
else:
    dependency_links = []


if "clean" in sys.argv[1:]:
    # Source: https://bit.ly/2NLVsgE
    print("deleting Cython files...")
    import subprocess

    subprocess.run(
        ["rm -f metaseq/*.so metaseq/**/*.so metaseq/*.pyd metaseq/**/*.pyd"],
        shell=True,
    )


def do_setup(package_data):
    setup(
        name="metaseq",
        version=version,
        description="MetaSeq, a framework for large language models, from Meta",
        url="https://github.com/fairinternal/metaseq",
        long_description=readme,
        long_description_content_type="text/markdown",
        setup_requires=[
            "cython",
            'numpy; python_version>="3.7"',
            "setuptools>=18.0",
        ],
        install_requires=[
            "azure-storage-blob",
            "boto3",
            "black==22.1.0",
            "click==8.0.4",
            "cython",
            'dataclasses; python_version<"3.7"',
            "editdistance",
            "fire",
            "flask==2.1.1",  # for api
            "hydra-core>=1.1.0",
            "iopath",
            "ipdb",
            "ipython",
            "Jinja2==3.1.1",  # for evals
            "markupsafe",  # for evals
            "more_itertools",
            "ninja",
            'numpy; python_version>="3.7"',
            "omegaconf",
            "pre-commit",
            "pytest",
            "regex",
            "sklearn",  # for evals
            "sacrebleu",  # for evals
            "tensorboard",
            "timeout-decorator",
            "tokenizers",
            "torch",
            "tqdm",
            "typing_extensions",
        ],
        dependency_links=dependency_links,
        packages=find_packages(
            exclude=[
                "scripts",
                "scripts.*",
                "tests",
                "tests.*",
            ]
        ),
        extras_require={
            "dev": [
                "flake8==3.9.2",
                "black==22.1.0",
                # test deps
                "iopath",
                "transformers",
                "pyarrow",
                "boto3",
            ]
        },
        package_data=package_data,
        ext_modules=extensions,
        test_suite="tests",
        entry_points={
            "console_scripts": [
                "metaseq-train = metaseq_cli.train:cli_main",
                "metaseq-validate = metaseq_cli.validate:cli_main",
                "opt-baselines = metaseq.launcher.opt_baselines:cli_main",
                "metaseq-api-local = metaseq_cli.interactive_hosted:cli_main",
            ],
        },
        cmdclass=cmdclass,
        zip_safe=False,
    )


def get_files(path, relative_to="metaseq"):
    all_files = []
    for root, _dirs, files in os.walk(path, followlinks=True):
        root = os.path.relpath(root, relative_to)
        for file in files:
            if file.endswith(".pyc"):
                continue
            all_files.append(os.path.join(root, file))
    return all_files


if __name__ == "__main__":
    package_data = {"metaseq": (get_files(os.path.join("metaseq", "config")))}
    do_setup(package_data)
