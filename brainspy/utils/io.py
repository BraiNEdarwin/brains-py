"""
Library that handles saving data
"""

import io
import os
import time
import pickle
import shutil
import torch
import yaml
import numpy as np


def save(mode, file_path, **kwargs):

    if mode == "numpy":
        np.savez(file_path, **kwargs)
    elif not kwargs["data"]:
        raise ValueError(f"Value dictionary is missing in kwargs.")
    else:
        if mode == "configs":
            save_configs(kwargs["data"], file_path)
        elif mode == "pickle":
            save_pickle(kwargs["data"], file_path)
        elif mode == "torch":
            save_torch(kwargs["data"], file_path)
        else:
            raise NotImplementedError(
                f"Mode {mode} is not recognised. Please choose a value between 'numpy', 'torch', 'pickle' and 'configs'."
            )
    # return path


def save_pickle(pickle_data, file_path):
    with open(file_path, "wb") as f:
        pickle.dump(pickle_data, f)
        f.close()


def save_torch(torch_model, file_path):
    """
    Saves the model in given path, all other attributes are saved under
    the 'info' key as a new dictionary.
    """
    torch_model.eval()
    state_dic = torch_model.state_dict()
    state_dic["info"] = torch_model.info
    torch.save(state_dic, file_path)


def load_configs(file_name):
    with open(file_name) as f:
        return yaml.load(f, Loader=IncludeLoader)


def save_configs(configs, file_name):
    with open(file_name, "w") as f:
        yaml.dump(configs, f, default_flow_style=False)


def create_directory(path, overwrite=False):
    """
    This function checks if there exists a directory filepath+datetime_name.
    If not it will create it and return this path.
    """
    if not os.path.exists(path):
        os.makedirs(path)
    elif overwrite:
        shutil.rmtree(path)
        os.makedirs(path)
    return path


def create_directory_timestamp(path, name, overwrite=False):
    datetime = time.strftime("%Y_%m_%d_%H%M%S")
    path = os.path.join(path, name + "_" + datetime)
    return create_directory(path, overwrite=overwrite)


class IncludeLoader(yaml.Loader):
    """
    yaml.Loader subclass handles "!include path/to/foo.yml" directives in config
    files.  When constructed with a file object, the root path for includes
    defaults to the directory containing the file, otherwise to the current
    working directory. In either case, the root path can be overridden by the
    `root` keyword argument.

    When an included file F contain its own !include directive, the path is
    relative to F's location.

    Example:
        YAML file /home/frodo/one-ring.yml:
            ---
            Name: The One Ring
            Specials:
                - resize-to-wearer
            Effects:
                - !include path/to/invisibility.yml

        YAML file /home/frodo/path/to/invisibility.yml:
            ---
            Name: invisibility
            Message: Suddenly you disappear!

        Loading:
            data = IncludeLoader(open('/home/frodo/one-ring.yml', 'r')).get_data()

        Result:
            {'Effects': [{'Message': 'Suddenly you disappear!', 'Name':
                'invisibility'}], 'Name': 'The One Ring', 'Specials':
                ['resize-to-wearer']}
    """

    def __init__(self, *args, **kwargs):
        super(IncludeLoader, self).__init__(*args, **kwargs)
        self.add_constructor("!include", self._include)
        if "root" in kwargs:
            self.root = kwargs["root"]
        elif isinstance(self.stream, io.TextIOWrapper):
            self.root = os.path.dirname(self.stream.name)
        else:
            self.root = os.path.curdir

    def _include(self, loader, node):
        oldRoot = self.root
        filename = os.path.join(self.root, loader.construct_scalar(node))
        self.root = os.path.dirname(filename)
        self.root = oldRoot
        with open(filename, "r") as f:
            return yaml.load(f)
