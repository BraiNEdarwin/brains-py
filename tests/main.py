"""
Module for running all tests on brainspy.
"""

import unittest
import brainspy

if __name__ == "__main__":
    modules_to_test = unittest.defaultTestLoader.discover(start_dir="tests/",
                                                          pattern="*.py",
                                                          top_level_dir=None)
    
    runner = unittest.TextTestRunner()
    runner.run(modules_to_test)
    print(brainspy.__TEST_MODE__)
