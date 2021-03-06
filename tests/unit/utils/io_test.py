import unittest
import HtmlTestRunner
import torch
import numpy as np
from numpy import load, asarray
from brainspy.utils.io import *
from brainspy.utils.loader import *
import tracemalloc
import yaml


class IOTest(unittest.TestCase):
    """
    Tests for the manager.py class.
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
        self.path = "C:/users/humai/brains-py/tests/unit/utils/testfiles"  # Enter path to the testfiles directory

    def test_saveconfigs(self):
        """
        Test for save_configs() method which saves a python dictionary to yaml file
        """
        path = self.path + "/testutil.yaml"
        save_configs(self.configs, path)
        x = load_configs(path)
        self.assertEqual(x["data"], self.configs["data"])
        self.assertEqual(x["slope_length"], 20)

    def test_save(self):
        """
        Test for save() method with parameter - "configs"
        """
        path = self.path + "/testutil.yaml"
        save("configs", path, data=self.configs)
        x = load_configs(path)
        self.assertEqual(x["data"], self.configs["data"])
        self.assertEqual(x["batch"], None)
        with self.assertRaises(
            KeyError
        ):  # Testing for data that does not exist in the file
            val = x["non_existant"]

    def test_savepickle(self):
        """
        Test for save_pickle() method which saves a python dictionary to yaml file
        """
        path = self.path + "/testutil.pickle"
        tracemalloc.start()
        save_pickle(self.configs, path)
        file = open(path, "rb")
        x = pickle.load(file)
        self.assertEqual(x["data"], self.configs["data"])
        with self.assertRaises(
            KeyError
        ):  # Testing for data that does not exist in the file
            val = x["non_existant"]
        file.close()

    def test_savenumpy(self):
        """
        Test for the save() method with parameter "numpy"
        Saving 2 numpy arrays to a .npz file
        """
        path = self.path + "/testutil.npz"
        x = np.arange(10)
        y = np.arange(11, 20)
        save("numpy", path, x=x, y=y)
        with np.load(path) as data:
            x2 = data["x"]
            y2 = data["y"]
            self.assertEqual(x[0], x2[0])
            self.assertEqual(y[7], y2[7])

    def test_createdir(self):
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

    def runTest(self):
        self.test_saveconfigs()
        self.test_save()
        self.test_savepickle()
        self.test_savenumpy()
        self.test_createdir()


if __name__ == "__main__":
    unittest.main(
        testRunner=HtmlTestRunner.HTMLTestRunner(
            output="C:/users/humai/brains-py/tests/unit/utils/testfiles"  # Enter the path where you want to save the results
        )
    )
