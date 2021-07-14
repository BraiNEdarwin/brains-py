"""
Library that handles loading and saving of data to a file.
It is also used to create a new directory with a timestamp.
"""

import io
import os
import time
import shutil
import yaml


def load_configs(file_name: str):
    """
    Loads a yaml file from the given file path.

    Parameters
    ----------
    file_name : str
        File object or path to yaml file.

    Returns
    -------
    dict : Python dictionary with formatted yaml data.

    Example
    --------
    file = "boolean.yaml"
    data = load_configs(file)

    """
    with open(file_name) as f:
        return yaml.load(f, Loader=IncludeLoader)


def save_configs(configs: dict, file_name: str):
    """
    Formats data from a dictionary and saves it to the given yaml file.

    Parameters
    ----------

    configs : dict
        Data to be stored in the yaml file.

    file_name : str
        File object or path to yaml file.

    Example
    --------
    configs = {"data" : "example"}
    file = "boolean.yaml"
    save_configs(configs,file)

    """
    with open(file_name, "w") as f:
        yaml.dump(configs, f, default_flow_style=False)


def create_directory(path: str, overwrite=False):
    """
    Creates a directory to the input path appending datetime to it.

    Parameters
    -----------

    path : str
        File object or path to file
    overwrite: boolean
        When True, if the directory exists, it will overwrite it. When False, if the directory
        exists it will not do anything. By default is False.

    Example
    -------
    path = "tests/unit/utils/testfiles"
    newpath = create_directory(path + "/TestDirectory")

    """
    if not os.path.exists(path):
        os.makedirs(path)
    elif overwrite:
        shutil.rmtree(path)
        os.makedirs(path)
    return path


def create_directory_timestamp(path: str, name: str, overwrite=False):
    """
    To create a directory with the given name and current timestamp if it does not already exist

    Parameters
    ----------

    path : str
        File object or path to file

    name : str

    Returns
    --------
    str : Path to file created - filepath+datetime_name

    Example
    --------
    path = "tests/unit/utils/testfiles"
    name = "TestDirectory"
    new_directory = create_directory_timestamp(self.path, name)

    """
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

    Example
    ---------
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
        """
        Constructor to initialize the file root and load the file.
        """
        super(IncludeLoader, self).__init__(*args, **kwargs)
        self.add_constructor("!include", self._include)
        if "root" in kwargs:
            self.root = kwargs["root"]
        elif isinstance(self.stream, io.TextIOWrapper):
            self.root = os.path.dirname(self.stream.name)
        else:
            self.root = os.path.curdir

    def _include(self, loader, node):
        """
        Method to add the load the file along with !include directive

        Parameters
        ----------
        loader : IncludeLoader
            loader object to construct a scalar node to the !include file
        node : str
            file path

        Returns
        -------
        dict
            loaded file as a python dictionary
        """
        oldRoot = self.root
        filename = os.path.join(self.root, loader.construct_scalar(node))
        self.root = os.path.dirname(filename)
        self.root = oldRoot
        with open(filename, "r") as f:
            return yaml.load(f, Loader=yaml.FullLoader)
