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
    Tests for the io.py class.
    To run the test, change the path variable in the places indicated below.
    """

    def __init__(self, test_name):
        super(IOTest, self).__init__()
        configs = {}
        configs["plateau_length"] = 80
        configs["slope_length"] = 20
        configs["batch"] = None
        configs["data"] = {"data1": "New data"}
        self.configs = configs
        self.path = "C:/users/humai/Downloads/brains-py/tests/unit/utils/testfiles"  # Enter path to the testfiles directory

    def test_save_load_configs(self):
        """
        Test for save_configs() method which saves a python dictionary to yaml file
        """
        path = self.path + "/testutil.yaml"
        save_configs(self.configs, path)
        x = load_configs(path)
        self.assertEqual(x["data"], self.configs["data"])
        self.assertEqual(x["slope_length"], 20)

    def test_createdir(self):
        """
        Test to create a new directory with a given file path
        """
        path = self.path + "/TestDirectory"
        newpath = create_directory(path)
        self.assertTrue(os.path.exists(newpath))
        shutil.rmtree(newpath)

    def test_create_dir_timestamp(self):
        """
        Test to create a new directory with a given name and current timestamp
        """
        name = "TestDirectory"
        newpath = create_directory_timestamp(self.path, name)
        self.assertTrue(os.path.exists(newpath))
        shutil.rmtree(newpath)

        # Test to create a new directory with 'NoneType' name

        name = None
        with self.assertRaises(TypeError):
            newpath = create_directory_timestamp(self.path, name)

    def test_includeloader_init(self):
        """
        Test for the _init_() method of IncludeLoader to check the correct initiliztaion of the file root
        """
        file = open(self.path + "/test_torch.yaml", "r")
        loader = IncludeLoader(file)
        self.assertEqual(loader.root, self.path)
        file.close()

    def test_includeloader_include(self):
        """
        Test for the _include() method of IncludeLoader.
        The !include directive is tested with the test_torch.yaml file which contains !include boolean.yaml
        """
        file = open(self.path + "/test_torch.yaml", "r")
        loader = IncludeLoader(file)
        data = loader.get_data()
        self.assertEqual(
            data["processor_type"], "simulation_debug"
        )  # data in the test_torch.yaml file
        self.assertEqual(
            data["algorithm"]["optimizer"], "genetic"
        )  # data in the boolean.yaml file
        file.close()

    def runTest(self):
        self.test_save_load_configs()
        self.test_createdir()
        self.test_create_dir_timestamp()
        self.test_includeloader_init()
        self.test_includeloader_include()


if __name__ == "__main__":
    unittest.main()
