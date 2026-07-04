# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Configuration utility functions
"""

import sys, os

# Allow "import depth_anything_3" to resolve to "models.depth_anything_3"
# so that HuggingFace config YAML __object__ paths work without modification.
_models_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _models_root not in sys.path:
    sys.path.insert(0, _models_root)

import importlib
from pathlib import Path
from typing import Any, Callable, List, Union
from omegaconf import DictConfig, ListConfig, OmegaConf

if not OmegaConf.has_resolver("eval"):
    OmegaConf.register_new_resolver("eval", eval)


def load_config(path: str, argv: List[str] = None) -> Union[DictConfig, ListConfig]:
    """
    Load a configuration. Will resolve inheritance.
    Supports both file paths and module paths (e.g., depth_anything_3.configs.giant).
    """
    # Check if path is a module path (contains dots but no slashes and doesn't end with .yaml)
    if "." in path and "/" not in path and not path.endswith(".yaml"):
        # It's a module path, load from package resources
        path_parts = path.split(".")[1:]
        config_path = Path(__file__).resolve().parent
        for part in path_parts:
            config_path = config_path.joinpath(part)
        config_path = config_path.with_suffix(".yaml")
        config = OmegaConf.load(str(config_path))
    else:
        # It's a file path (absolute, relative, or with .yaml extension)
        config = OmegaConf.load(path)

    if argv is not None:
        config_argv = OmegaConf.from_dotlist(argv)
        config = OmegaConf.merge(config, config_argv)
    config = resolve_recursive(config, resolve_inheritance)
    return config


def resolve_recursive(
    config: Any,
    resolver: Callable[[Union[DictConfig, ListConfig]], Union[DictConfig, ListConfig]],
) -> Any:
    config = resolver(config)
    if isinstance(config, DictConfig):
        for k in config.keys():
            v = config.get(k)
            if isinstance(v, (DictConfig, ListConfig)):
                config[k] = resolve_recursive(v, resolver)
    if isinstance(config, ListConfig):
        for i in range(len(config)):
            v = config.get(i)
            if isinstance(v, (DictConfig, ListConfig)):
                config[i] = resolve_recursive(v, resolver)
    return config


def resolve_inheritance(config: Union[DictConfig, ListConfig]) -> Any:
    """
    Recursively resolve inheritance if the config contains:
    __inherit__: path/to/parent.yaml or a ListConfig of such paths.
    """
    if isinstance(config, DictConfig):
        inherit = config.pop("__inherit__", None)

        if inherit:
            inherit_list = inherit if isinstance(inherit, ListConfig) else [inherit]

            parent_config = None
            for parent_path in inherit_list:
                assert isinstance(parent_path, str)
                parent_config = (
                    load_config(parent_path)
                    if parent_config is None
                    else OmegaConf.merge(parent_config, load_config(parent_path))
                )

            if len(config.keys()) > 0:
                config = OmegaConf.merge(parent_config, config)
            else:
                config = parent_config
    return config


def import_item(path: str, name: str) -> Any:
    """
    Import a python item. Example: import_item("path.to.file", "MyClass") -> MyClass
    """
    try:
        return getattr(importlib.import_module(path), name)
    except ModuleNotFoundError:
        # HuggingFace configs use "depth_anything_3.*"; remap to "models.depth_anything_3.*"
        if path.startswith("depth_anything_3"):
            return getattr(importlib.import_module("models." + path), name)
        raise


def create_object(config: DictConfig) -> Any:
    """
    Create an object from config.
    The config is expected to contains the following:
    __object__:
      path: path.to.module
      name: MyClass
      args: as_config | as_params (default to as_config)
    """
    config = DictConfig(config)
    item = import_item(
        path=config.__object__.path,
        name=config.__object__.name,
    )
    args = config.__object__.get("args", "as_config")
    if args == "as_config":
        return item(config)
    if args == "as_params":
        config = OmegaConf.to_object(config)
        config.pop("__object__")
        return item(**config)
    raise NotImplementedError(f"Unknown args type: {args}")


def create_dataset(path: str, *args, **kwargs) -> Any:
    """
    Create a dataset. Requires the file to contain a "create_dataset" function.
    """
    return import_item(path, "create_dataset")(*args, **kwargs)


def to_dict_recursive(config_obj):
    if isinstance(config_obj, DictConfig):
        return {k: to_dict_recursive(v) for k, v in config_obj.items()}
    elif isinstance(config_obj, ListConfig):
        return [to_dict_recursive(item) for item in config_obj]
    return config_obj
