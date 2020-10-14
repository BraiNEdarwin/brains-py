import unittest
import sys


class T1(unittest.TestCase):
    def test_A(self):
        pass

    def test_B(self):
        pass


class T2(unittest.TestCase):
    def test_A(self):
        pass

    def test_B(self):
        pass


if __name__ == "__main__":
    from HtmlTestRunner import HTMLTestRunner

    modules_to_test = unittest.defaultTestLoader.discover(start_dir='test/', pattern='*.py', top_level_dir=None)
    HTMLTestRunner(output='/home/unai/Documents/3-programming/brains-py/brains-py/test-reports').run(modules_to_test)
