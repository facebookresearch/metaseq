# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import ast
import inspect
import logging
import os
import re
from argparse import ArgumentError, ArgumentParser, Namespace
from dataclasses import _MISSING_TYPE, MISSING
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Type

from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf, open_dict
from omegaconf.errors import ConfigKeyError

from metaseq.dataclass import MetaseqDataclass
from metaseq.dataclass.configs import MetaseqConfig

logger = logging.getLogger(__name__)


def eval_str_list(x, x_type=float):
    if x is None:
        return None
    if isinstance(x, str):
        if len(x) == 0:
            return []
        x = ast.literal_eval(x)
    try:
        return list(map(x_type, x))
    except TypeError:
        return [x_type(x)]


def interpret_dc_type(field_type):
    if isinstance(field_type, str):
        raise RuntimeError("field should be a type")

    if field_type == Any:
        return str

    typestring = str(field_type)
    if re.match(
        r"(typing.|^)Union\[(.*), NoneType\]$", typestring
    ) or typestring.startswith("typing.Optional"):
        return field_type.__args__[0]
    return field_type


def gen_parser_from_dataclass(
    parser: ArgumentParser,
    dataclass_instance: MetaseqDataclass,
    delete_default: bool = False,
) -> None:
    """convert a dataclass instance to tailing parser arguments"""

    def argparse_name(name: str):
        if name in dataclass_instance.positional_args():
            return name
        if name == "_name":
            # private member, skip
            return None
        return "--" + name.replace("_", "-")

    def get_kwargs_from_dc(
        dataclass_instance: MetaseqDataclass, k: str
    ) -> Dict[str, Any]:
        """k: dataclass attributes"""

        kwargs = {}

        field_type = dataclass_instance._get_type(k)
        inter_type = interpret_dc_type(field_type)

        field_default = dataclass_instance._get_default(k)

        if isinstance(inter_type, type) and issubclass(inter_type, Enum):
            field_choices = [t.value for t in list(inter_type)]
        else:
            field_choices = None

        field_help = dataclass_instance._get_help(k)
        field_const = dataclass_instance._get_argparse_const(k)

        if isinstance(field_default, str) and field_default.startswith("${"):
            kwargs["default"] = field_default
        else:
            if field_default is MISSING:
                kwargs["required"] = True
            if field_choices is not None:
                kwargs["choices"] = field_choices
            if (
                isinstance(inter_type, type)
                and (issubclass(inter_type, List) or issubclass(inter_type, Tuple))
            ) or ("List" in str(inter_type) or "Tuple" in str(inter_type)):
                if "int" in str(inter_type):
                    kwargs["type"] = lambda x: eval_str_list(x, int)
                elif "float" in str(inter_type):
                    kwargs["type"] = lambda x: eval_str_list(x, float)
                elif "str" in str(inter_type):
                    kwargs["type"] = lambda x: eval_str_list(x, str)
                else:
                    raise NotImplementedError(
                        "parsing of type " + str(inter_type) + " is not implemented"
                    )
                if field_default is not MISSING:
                    kwargs["default"] = (
                        ",".join(map(str, field_default))
                        if field_default is not None
                        else None
                    )
            elif (
                isinstance(inter_type, type) and issubclass(inter_type, Enum)
            ) or "Enum" in str(inter_type):
                kwargs["type"] = str
                if field_default is not MISSING:
                    if isinstance(field_default, Enum):
                        kwargs["default"] = field_default.value
                    else:
                        kwargs["default"] = field_default
            elif inter_type is bool:
                kwargs["action"] = (
                    "store_false" if field_default is True else "store_true"
                )
                kwargs["default"] = field_default
            else:
                kwargs["type"] = inter_type
                if field_default is not MISSING:
                    kwargs["default"] = field_default

        kwargs["help"] = field_help
        if field_const is not None:
            kwargs["const"] = field_const
            kwargs["nargs"] = "?"

        return kwargs

    for k in dataclass_instance._get_all_attributes():
        field_name = argparse_name(dataclass_instance._get_name(k))
        field_type = dataclass_instance._get_type(k)
        if field_name is None:
            continue
        elif inspect.isclass(field_type) and issubclass(field_type, MetaseqDataclass):
            gen_parser_from_dataclass(parser, field_type(), delete_default)
            continue

        kwargs = get_kwargs_from_dc(dataclass_instance, k)

        field_args = [field_name]
        alias = dataclass_instance._get_argparse_alias(k)
        if alias is not None:
            field_args.append(alias)

        if "default" in kwargs:
            if isinstance(kwargs["default"], str) and kwargs["default"].startswith(
                "${"
            ):
                if kwargs["help"] is None:
                    # this is a field with a name that will be added elsewhere
                    continue
                else:
                    del kwargs["default"]
            if delete_default and "default" in kwargs:
                del kwargs["default"]
        try:
            parser.add_argument(*field_args, **kwargs)
        except ArgumentError:
            pass


def _set_legacy_defaults(args, cls):
    """Helper to set default arguments based on *add_args*."""
    if not hasattr(cls, "add_args"):
        return

    import argparse

    parser = argparse.ArgumentParser(
        argument_default=argparse.SUPPRESS, allow_abbrev=False
    )
    cls.add_args(parser)
    # copied from argparse.py:
    defaults = argparse.Namespace()
    for action in parser._actions:
        if action.dest is not argparse.SUPPRESS:
            if not hasattr(defaults, action.dest):
                if action.default is not argparse.SUPPRESS:
                    setattr(defaults, action.dest, action.default)
    for key, default_value in vars(defaults).items():
        if not hasattr(args, key):
            setattr(args, key, default_value)


def _override_attr(
    sub_node: str, data_class: Type[MetaseqDataclass], args: Namespace
) -> List[str]:
    overrides = []

    if not inspect.isclass(data_class) or not issubclass(data_class, MetaseqDataclass):
        return overrides

    def get_default(f):
        if not isinstance(f.default_factory, _MISSING_TYPE):
            return f.default_factory()
        return f.default

    for k, v in data_class.__dataclass_fields__.items():
        if k.startswith("_"):
            # private member, skip
            continue

        val = get_default(v) if not hasattr(args, k) else getattr(args, k)

        field_type = interpret_dc_type(v.type)
        if (
            isinstance(val, str)
            and not val.startswith("${")  # not interpolation
            and field_type != str
            and (
                not inspect.isclass(field_type) or not issubclass(field_type, Enum)
            )  # not choices enum
        ):
            # upgrade old models that stored complex parameters as string
            val = ast.literal_eval(val)

        if isinstance(val, tuple):
            val = list(val)

        v_type = getattr(v.type, "__origin__", None)
        if (
            (v_type is List or v_type is list or v_type is Optional)
            # skip interpolation
            and not (isinstance(val, str) and val.startswith("${"))
        ):
            # if type is int but val is float, then we will crash later - try to convert here
            if hasattr(v.type, "__args__"):
                t_args = v.type.__args__
                if len(t_args) == 1 and (t_args[0] is float or t_args[0] is int):
                    val = list(map(t_args[0], val))
        elif val is not None and (
            field_type is int or field_type is bool or field_type is float
        ):
            try:
                # Future reader, if you experience something other than a ValueError here,
                # we used to simply ignore all excepts. Add it to the allowlist.
                val = field_type(val)
            except ValueError:
                pass  # ignore errors here, they are often from interpolation args

        if val is None:
            overrides.append("{}.{}=null".format(sub_node, k))
        elif val == "":
            overrides.append("{}.{}=''".format(sub_node, k))
        elif isinstance(val, str):
            val = val.replace("'", r"\'")
            overrides.append("{}.{}='{}'".format(sub_node, k, val))
        elif isinstance(val, MetaseqDataclass):
            overrides += _override_attr(f"{sub_node}.{k}", type(val), args)
        elif isinstance(val, Namespace):
            sub_overrides, _ = override_module_args(val)
            for so in sub_overrides:
                overrides.append(f"{sub_node}.{k}.{so}")
        else:
            overrides.append("{}.{}={}".format(sub_node, k, val))

    return overrides


def migrate_registry(
    name, value, registry, args, overrides, deletes, use_name_as_val=False
):
    if value in registry:
        overrides.append("{}={}".format(name, value))
        overrides.append("{}._name={}".format(name, value))
        overrides.extend(_override_attr(name, registry[value], args))
    elif use_name_as_val and value is not None:
        overrides.append("{}={}".format(name, value))
    else:
        deletes.append(name)


def override_module_args(args: Namespace) -> Tuple[List[str], List[str]]:
    """use the field in args to overrides those in cfg"""
    overrides = []
    deletes = []

    for k in MetaseqConfig.__dataclass_fields__.keys():
        overrides.extend(
            _override_attr(k, MetaseqConfig.__dataclass_fields__[k].type, args)
        )

    if args is not None:
        if hasattr(args, "task"):
            from metaseq.tasks import TASK_DATACLASS_REGISTRY

            migrate_registry(
                "task", args.task, TASK_DATACLASS_REGISTRY, args, overrides, deletes
            )
        else:
            deletes.append("task")

        # these options will be set to "None" if they have not yet been migrated
        # so we can populate them with the entire flat args
        CORE_REGISTRIES = {"criterion", "optimizer", "lr_scheduler"}

        from metaseq.registry import REGISTRIES

        for k, v in REGISTRIES.items():
            if hasattr(args, k):
                migrate_registry(
                    k,
                    getattr(args, k),
                    v["dataclass_registry"],
                    args,
                    overrides,
                    deletes,
                    use_name_as_val=k not in CORE_REGISTRIES,
                )
            else:
                deletes.append(k)

        no_dc = True
        if hasattr(args, "arch"):
            from metaseq.models import ARCH_MODEL_REGISTRY, ARCH_MODEL_NAME_REGISTRY

            if args.arch in ARCH_MODEL_REGISTRY:
                m_cls = ARCH_MODEL_REGISTRY[args.arch]
                dc = getattr(m_cls, "__dataclass", None)
                if dc is not None:
                    m_name = ARCH_MODEL_NAME_REGISTRY[args.arch]
                    overrides.append("model={}".format(m_name))
                    overrides.append("model._name={}".format(args.arch))
                    # override model params with those exist in args
                    overrides.extend(_override_attr("model", dc, args))
                    no_dc = False
        if no_dc:
            deletes.append("model")

    return overrides, deletes


def convert_namespace_to_omegaconf(args: Namespace) -> DictConfig:
    """Convert a flat argparse.Namespace to a structured DictConfig."""
    # Here we are using field values provided in args to override counterparts inside config object
    overrides, deletes = override_module_args(args)

    # configs will be in metaseq/config after installation
    config_path = os.path.join("..", "config")

    GlobalHydra.instance().clear()

    with initialize(config_path=config_path):
        try:
            composed_cfg = compose("config", overrides=overrides)
        except Exception:
            logger.error("Error when composing. Overrides: " + str(overrides))
            raise

        for k in deletes:
            composed_cfg[k] = None

    cfg = OmegaConf.create(
        OmegaConf.to_container(composed_cfg, resolve=True, enum_to_str=True)
    )

    # hack to be able to set Namespace in dict config. this should be removed when we update to newer
    # omegaconf version that supports object flags, or when we migrate all existing models
    from omegaconf import _utils

    old_primitive = _utils.is_primitive_type
    _utils.is_primitive_type = lambda _: True

    if cfg.task is None and getattr(args, "task", None):
        cfg.task = Namespace(**vars(args))
        from metaseq.tasks import TASK_REGISTRY

        _set_legacy_defaults(cfg.task, TASK_REGISTRY[args.task])
        cfg.task._name = args.task
    if cfg.model is None and getattr(args, "arch", None):
        cfg.model = Namespace(**vars(args))
        from metaseq.models import ARCH_MODEL_REGISTRY

        _set_legacy_defaults(cfg.model, ARCH_MODEL_REGISTRY[args.arch])
        cfg.model._name = args.arch
    if cfg.optimizer is None and getattr(args, "optimizer", None):
        cfg.optimizer = Namespace(**vars(args))
        from metaseq.optim import OPTIMIZER_REGISTRY

        _set_legacy_defaults(cfg.optimizer, OPTIMIZER_REGISTRY[args.optimizer])
        cfg.optimizer._name = args.optimizer
    if cfg.lr_scheduler is None and getattr(args, "lr_scheduler", None):
        cfg.lr_scheduler = Namespace(**vars(args))
        from metaseq.optim.lr_scheduler import LR_SCHEDULER_REGISTRY

        _set_legacy_defaults(cfg.lr_scheduler, LR_SCHEDULER_REGISTRY[args.lr_scheduler])
        cfg.lr_scheduler._name = args.lr_scheduler
    if cfg.criterion is None and getattr(args, "criterion", None):
        cfg.criterion = Namespace(**vars(args))
        from metaseq.criterions import CRITERION_REGISTRY

        _set_legacy_defaults(cfg.criterion, CRITERION_REGISTRY[args.criterion])
        cfg.criterion._name = args.criterion

    _utils.is_primitive_type = old_primitive
    OmegaConf.set_struct(cfg, True)
    return cfg


def populate_dataclass(
    dataclass: MetaseqDataclass,
    args: Namespace,
) -> MetaseqDataclass:
    for k in dataclass.__dataclass_fields__.keys():
        if k.startswith("_"):
            # private member, skip
            continue
        if hasattr(args, k):
            setattr(dataclass, k, getattr(args, k))

    return dataclass


def overwrite_args_by_name(cfg: DictConfig, overrides: Dict[str, any]):
    # this will be deprecated when we get rid of argparse and model_overrides logic

    from metaseq.registry import REGISTRIES

    with open_dict(cfg):
        for k in cfg.keys():
            # "k in cfg" will return false if its a "mandatory value (e.g. ???)"
            if k in cfg and isinstance(cfg[k], DictConfig):
                if k in overrides and isinstance(overrides[k], dict):
                    for ok, ov in overrides[k].items():
                        if isinstance(ov, dict) and cfg[k][ok] is not None:
                            overwrite_args_by_name(cfg[k][ok], ov)
                        else:
                            cfg[k][ok] = ov
                else:
                    overwrite_args_by_name(cfg[k], overrides)
            elif k in cfg and isinstance(cfg[k], Namespace):
                for override_key, val in overrides.items():
                    setattr(cfg[k], override_key, val)
            elif k in overrides:
                if (
                    k in REGISTRIES
                    and overrides[k] in REGISTRIES[k]["dataclass_registry"]
                ):
                    cfg[k] = DictConfig(
                        REGISTRIES[k]["dataclass_registry"][overrides[k]]
                    )
                    overwrite_args_by_name(cfg[k], overrides)
                    cfg[k]._name = overrides[k]
                else:
                    cfg[k] = overrides[k]


def merge_with_parent(dc: MetaseqDataclass, cfg: MetaseqDataclass):
    try:
        merged_cfg = OmegaConf.merge(dc, cfg)
    except ConfigKeyError:
        # Workaround for missing keys - reverse the merge direction, then
        # merge back in to the flipped merge
        # See https://github.com/fairinternal/fairseq-big-internal/issues/115#issuecomment-1073129691
        flipped_merge = OmegaConf.merge(cfg, dc)
        merged_cfg = OmegaConf.merge(flipped_merge, cfg)

    # Logic from https://github.com/omry/omegaconf/issues/441#issuecomment-737558869 ?
    merged_cfg.__dict__["_parent"] = cfg.__dict__["_parent"]
    OmegaConf.set_struct(merged_cfg, True)
    return merged_cfg
