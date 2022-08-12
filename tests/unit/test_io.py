"""
Module to test the methods used in loading and saving of data from a file.
"""
import unittest
from brainspy.utils.io import (
    save_configs,
    load_configs,
    create_directory_timestamp,
    create_directory,
    IncludeLoader,
)
import shutil
import os


class IOTest(unittest.TestCase):
    """
    Tests for the io.py class. The class is used to test
    the methods used to load and save the contents of a file.
    """

    def __init__(self, test_name):
        super(IOTest, self).__init__()
        configs = {}
        configs["plateau_length"] = 80
        configs["slope_length"] = 20
        configs["batch"] = None
        configs["data"] = {"data1": "New data"}
        self.configs = configs
        self.path = os.path.join(os.getcwd(), "tests/unit/ut/testfiles")

    def test_save_load_configs(self):
        """
        Test for save_configs() method which saves a python dictionary to yaml file
        and for the load_configs(..) method to load the contents of the same file
        """
        path = self.path + "/testutil.yaml"
        save_configs(self.configs, path)
        x = load_configs(path)
        self.assertEqual(x["data"], self.configs["data"])
        self.assertEqual(x["slope_length"], 20)

    def test_load_worng_configs(self):
        """
        Test to load the correct file but use the wrong key which raises a KeyError
        """
        path = self.path + "/test_torch.yaml"
        x = load_configs(path)
        with self.assertRaises(KeyError):
            x["wrong_key"]

    def test_load_configs_none(self):
        """
        TyeError raised if a None type path is provided to load the file
        """
        path = None
        with self.assertRaises(TypeError):
            load_configs(path)

    def test_save_configs_none(self):
        """
        TyeError raised if a None type path is provided to save the configs
        """
        path = None
        with self.assertRaises(TypeError):
            save_configs(path)

    def test_load_wrong_path(self):
        """
        FileNotFoundError raised if the wrong path or filename is provided and no configs are loaded
        """
        path = "wrong_path"
        with self.assertRaises(FileNotFoundError):
            load_configs(path)

    def test_save_wrong_path(self):
        """
        TypeError raised if a the wrong path or filename is provided to save the configs
        """
        path = "wrong_path"
        with self.assertRaises(TypeError):
            save_configs(path)

    def test_createdir(self):
        """
        Test to create a new directory with a given file path
        """
        try:
            path = self.path + "/TestDirectory"
            newpath = create_directory(path)
            self.assertTrue(os.path.exists(newpath))
            shutil.rmtree(newpath)
        except Exception:
            self.fail('Unable to create dir')

    def test_createdexistingir(self):
        """
        Test to create a new directory with a given file path
        """
        try:
            path = self.path + "/TestDirectory"
            os.mkdir(path)
            newpath = create_directory(path, overwrite=True)
            self.assertTrue(os.path.exists(newpath))
            shutil.rmtree(newpath)
        except Exception:
            self.fail('Unable to create/overwrite dir')

    def test_createdir_none(self):
        """
        TypeError raised if a none type path is provided
        """
        path = None
        with self.assertRaises(TypeError):
            create_directory(path)

    def test_create_dir_timestamp(self):
        """
        Test to create a new directory with a given name and current timestamp
        """
        try:
            name = "TestDirectory"
            newpath = create_directory_timestamp(self.path, name)
            self.assertTrue(os.path.exists(newpath))
            shutil.rmtree(newpath)
        except Exception:
            self.fail('Unable to create/overwrite dir')

    def test_create_dir_timestamp_none(self):
        """
        Test to create a new directory with 'NoneType' name
        """
        name = None
        with self.assertRaises(TypeError):
            create_directory_timestamp(self.path, name)

    def test_includeloader_init(self):
        """
        Test for the _init_() method of IncludeLoader to check the correct initiliztaion of
        the file root
        """
        try:
            file = open(self.path + "/test_torch.yaml", "r")
            loader = IncludeLoader(file)
            self.assertEqual(loader.root, self.path)
            file.close()
        except Exception:
            self.fail('Unable to create/overwrite dir')

    def test_includeloader_init_none(self):
        """
        AttributeError is raised if a none type file is provided to the IncludeLoader class
        """
        with self.assertRaises(AttributeError):
            IncludeLoader(None)

    def test_includeloader_include(self):
        """
        Test for the _include() method of IncludeLoader.
        The !include directive is tested with the test_torch.yaml file which contains
        !include boolean.yaml
        """
        try:
            file = open(self.path + "/test_torch.yaml", "r")
            # !include
            loader = IncludeLoader(file)
            data = loader.get_data()
            self.assertEqual(
                data["processor_type"],
                "simulation_debug")  # data in the test_torch.yaml file
            self.assertEqual(data["algorithm"]["optimizer"],
                             "genetic")  # data in the boolean.yaml file
            file.close()
        except Exception:
            self.fail('Unable to create/overwrite dir')

    def runTest(self):
        self.test_save_load_configs()
        self.test_load_worng_configs()
        self.test_load_configs_none()
        self.test_save_configs_none()
        self.test_load_wrong_path()
        self.test_save_wrong_path()
        self.test_createdir()
        self.test_createdir_none()
        self.test_create_dir_timestamp()
        self.test_create_dir_timestamp_none()
        self.test_includeloader_init()
        self.test_includeloader_init_none()
        self.test_includeloader_include()
        self.test_createdexistingir()


if __name__ == "__main__":
    unittest.main()
