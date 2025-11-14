from __future__ import absolute_import, print_function
import os
import sys
from importlib import import_module
import json
import yaml
from . import filestream as fs

__all__ = [
    "Config",
]


def _check_path_exists(path, local_only=False):
    if local_only:
        fn = os.path.exists
    else:
        fn = fs.exists
    assert fn(path), "%s does not exists" % path


def load_pyfile(filename, allow_unsafe=False):
    _check_path_exists(filename, local_only=True)
    module_name = os.path.basename(filename)[:-3]
    config_dir = os.path.dirname(filename)
    sys.path.insert(0, config_dir)
    mod = import_module(module_name)
    sys.path.pop(0)
    cfg = {
        name: value for name, value in mod.__dict__.items() if not name.startswith("__")
    }
    sys.modules.pop(module_name)
    return cfg


def load_json(filename, allow_unsafe=False):
    _check_path_exists(filename)
    with fs.reader(filename) as f:
        cfg = json.load(f)
    return cfg


def load_yaml(filename, allow_unsafe=False):
    _check_path_exists(filename)
    with fs.reader(filename) as f:
        if allow_unsafe:
            cfg = yaml.unsafe_load(f)
        else:
            cfg = yaml.safe_load(f)
    return cfg


ext_to_load_fn_map = {
    ".py": load_pyfile,
    ".json": load_json,
    ".yaml": load_yaml,
    ".yml": load_yaml,
}


class Config(dict):
    """
    An wrapper of dict with easy attribute accessing and attribute
    protection.

    Examples
    --------
    >>> cfg = Config(dict(a=1))
    >>> cfg.a
    1
    >>> cfg.b = 2
    >>> cfg.b
    2
    >>> cfg.set_immutable()
    >>> cfg.a = 1
    # cannot success, will raise an AttributeError

    Parameters
    ----------
    cfg_dict : dict
        The initial value.
    """

    _BASE_CONFIG = "BASE_CONFIG"
    _RECURSIVE_UPDATE_BASE_CONFIG = "RECURSIVE_UPDATE_BASE_CONFIG"
    _MUTABLE = "_MUTABLE"

    @staticmethod
    def fromfile(filename, allow_unsafe=False):
        """Get :py:class:`Config` from file.

        Parameters
        ----------
        filename : str
            Supports file with suffix .yaml, .yml, .json, .py
        allow_unsafe : bool, optional
            Used when loading files in .yaml or .yml format, by default False

        Returns
        -------
        config: :py:class:`Config`
        """
        ext = os.path.splitext(filename)[-1]
        loader_fn = ext_to_load_fn_map.get(ext, None)
        if loader_fn is None:
            raise TypeError(
                "Unsupported ext %s, valid exts are %s"
                % (ext, ext_to_load_fn_map.keys())
            )
        cfg = loader_fn(filename, allow_unsafe)

        base_cfg_file = cfg.pop(Config._BASE_CONFIG, None)
        recursive_update_base_config = cfg.pop(
            Config._RECURSIVE_UPDATE_BASE_CONFIG, True
        )

        if base_cfg_file is None:
            return Config(cfg)
        else:
            if base_cfg_file.startswith("~"):
                base_cfg_file = os.path.expanduser(base_cfg_file)
            if not base_cfg_file.startswith("/"):
                # the path to base cfg is relative to the config file itself.
                base_cfg_file = os.path.join(os.path.dirname(filename), base_cfg_file)
            base_cfg = Config.load_file(base_cfg_file, allow_unsafe=allow_unsafe)

            def merge_a_into_b(a, b):
                # merge dict a into dict b. values in a will overwrite b.
                for k, v in a.items():
                    if recursive_update_base_config:
                        if isinstance(v, dict) and k in b:
                            if not isinstance(b[k], dict):
                                b[k] = dict()
                            merge_a_into_b(v, b[k])
                        else:
                            b[k] = v
                    else:
                        b[k] = v

            merge_a_into_b(cfg, base_cfg)
            return Config(base_cfg)

    def __init__(self, cfg_dict=None):
        assert isinstance(cfg_dict, dict)
        new_dict = {}
        for k, v in cfg_dict.items():
            if isinstance(v, dict):
                new_dict[k] = Config(v)
            elif isinstance(v, (list, tuple)):
                v = v.__class__(
                    [Config(v_i) if isinstance(v_i, dict) else v_i for v_i in v]
                )
                new_dict[k] = v
            else:
                new_dict[k] = v
        super(Config, self).__init__(new_dict)
        self.__dict__[Config._MUTABLE] = True

    def __getattr__(self, name):
        if name in self.keys():
            return self[name]
        else:
            raise AttributeError(name)

    def __setattr__(self, name, value):
        assert (
            name not in self.__dict__
        ), "Invalid attempt to modify internal Config state: {}".format(name)
        if self.__dict__[Config._MUTABLE]:
            if isinstance(value, dict):
                value = Config(value)
            elif isinstance(value, (list, tuple)):
                value = value.__class__(
                    [Config(v_i) if isinstance(v_i, dict) else v_i for v_i in value]
                )
            self[name] = value
        else:
            raise AttributeError(
                "Attempted to set {} to {}, but Config is immutable".format(name, value)
            )

    @staticmethod
    def _recursive_visit(obj, fn):
        if isinstance(obj, Config):
            fn(obj)
        if isinstance(obj, dict):
            for value_i in obj.values():
                Config._recursive_visit(value_i, fn)
        elif isinstance(obj, (list, tuple)):
            for value_i in obj:
                Config._recursive_visit(value_i, fn)

    def set_immutable(self):
        def _fn(obj):
            obj.__dict__[Config._MUTABLE] = False

        self._recursive_visit(self, _fn)

    def set_mutable(self):
        def _fn(obj):
            obj.__dict__[Config._MUTABLE] = True

        self._recursive_visit(self, _fn)

    def __str__(self):
        def _indent(s_, num_spaces):
            s = s_.split("\n")
            if len(s) == 1:
                return s_
            first = s.pop(0)
            s = [(num_spaces * " ") + line for line in s]
            s = "\n".join(s)
            s = first + "\n" + s
            return s

        r = ""
        s = []
        for k, v in sorted(self.items()):
            seperator = "\n" if isinstance(v, Config) else " "
            attr_str = "{}:{}{}".format(str(k), seperator, str(v))
            attr_str = _indent(attr_str, 2)
            s.append(attr_str)
        r += "\n".join(s)
        return r

    @staticmethod
    def _to_str(obj):
        if isinstance(obj, Config):
            return obj.to_str()
        elif isinstance(obj, (list, tuple)):
            str_value = []
            for sub in obj:
                str_value.append(Config._to_str(sub))
            return str_value
        elif not isinstance(obj, (int, float, bool, str)) and obj is not None:
            return obj.__str__()
        else:
            return obj

    def to_str(self):
        str_config = {}
        for k, v in self.items():
            str_config[k] = Config._to_str(v)
        return str_config

    def __repr__(self):
        return "{}({})".format(self.__class__.__name__, super(Config, self).__repr__())
