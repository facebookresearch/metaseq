# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""isort:skip_file"""

import argparse
import importlib
import os

from metaseq.dataclass import MetaseqDataclass
from metaseq.dataclass.utils import merge_with_parent, populate_dataclass
from hydra.core.config_store import ConfigStore

from .base_task import BaseTask, LegacyTask  # noqa


# register dataclass
TASK_DATACLASS_REGISTRY = {}
TASK_REGISTRY = {}
TASK_CLASS_NAMES = set()


def setup_task(cfg: MetaseqDataclass, **kwargs):
    task = None
    task_name = getattr(cfg, "task", None)

    if isinstance(task_name, str):
        # legacy tasks
        task = TASK_REGISTRY[task_name]
        if task_name in TASK_DATACLASS_REGISTRY:
            dc = TASK_DATACLASS_REGISTRY[task_name]
            cfg = populate_dataclass(dc(), cfg)
    else:
        task_name = getattr(cfg, "_name", None)

        if task_name and task_name in TASK_DATACLASS_REGISTRY:
            dc = TASK_DATACLASS_REGISTRY[task_name]
            cfg = merge_with_parent(dc(), cfg)
            task = TASK_REGISTRY[task_name]

    assert (
        task is not None
    ), f"Could not infer task type from {cfg}. Available tasks: {TASK_REGISTRY.keys()}"

    return task.setup_task(cfg, **kwargs)


def register_task(name, dataclass=None):
    """
    New tasks can be added to metaseq with the
    :func:`~metaseq.tasks.register_task` function decorator.

    For example::

        @register_task('classification')
        class ClassificationTask(BaseTask):
            (...)

    .. note::

        All Tasks must implement the :class:`~metaseq.tasks.BaseTask`
        interface.

    Args:
        name (str): the name of the task
    """

    def register_task_cls(cls):
        if name in TASK_REGISTRY:
            raise ValueError("Cannot register duplicate task ({})".format(name))
        if not issubclass(cls, BaseTask):
            raise ValueError(
                "Task ({}: {}) must extend BaseTask".format(name, cls.__name__)
            )
        if cls.__name__ in TASK_CLASS_NAMES:
            raise ValueError(
                "Cannot register task with duplicate class name ({})".format(
                    cls.__name__
                )
            )
        TASK_REGISTRY[name] = cls
        TASK_CLASS_NAMES.add(cls.__name__)

        if dataclass is not None and not issubclass(dataclass, MetaseqDataclass):
            raise ValueError(
                "Dataclass {} must extend MetaseqDataclass".format(dataclass)
            )

        cls.__dataclass = dataclass
        if dataclass is not None:
            TASK_DATACLASS_REGISTRY[name] = dataclass

            cs = ConfigStore.instance()
            node = dataclass()
            node._name = name
            cs.store(name=name, group="task", node=node, provider="metaseq")

        return cls

    return register_task_cls


def get_task(name):
    return TASK_REGISTRY[name]


# automatically import any Python files in the tasks/ directory
tasks_dir = os.path.dirname(__file__)
for file in os.listdir(tasks_dir):
    path = os.path.join(tasks_dir, file)
    if (
        not file.startswith("_")
        and not file.startswith(".")
        and (file.endswith(".py") or os.path.isdir(path))
    ):
        task_name = file[: file.find(".py")] if file.endswith(".py") else file
        module = importlib.import_module("metaseq.tasks." + task_name)

        # expose `task_parser` for sphinx
        if task_name in TASK_REGISTRY:
            parser = argparse.ArgumentParser(add_help=False)
            group_task = parser.add_argument_group("Task name")
            # fmt: off
            group_task.add_argument('--task', metavar=task_name,
                                    help='Enable this task with: ``--task=' + task_name + '``')
            # fmt: on
            group_args = parser.add_argument_group("Additional command-line arguments")
            TASK_REGISTRY[task_name].add_args(group_args)
            globals()[task_name + "_parser"] = parser
