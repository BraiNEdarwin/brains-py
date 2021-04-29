"""
Module for running all tests on brainspy.
"""

import unittest

if __name__ == "__main__":
    from HtmlTestRunner import HTMLTestRunner

    modules_to_test = unittest.defaultTestLoader.discover(start_dir="tests/",
                                                          pattern="*.py",
                                                          top_level_dir=None)
    HTMLTestRunner(output="tmp/test-reports").run(modules_to_test)
