"""
Module for running all tests on brainspy.
"""

import unittest
import brainspy

if __name__ == "__main__":
    from HtmlTestRunner import HTMLTestRunner
    brainspy.TEST_MODE = 'SIMULATION_PC'  # Available test modes: SIMULATION_PC, HARDWARE_CDAQ, HARDWARE_NIDAQ 
    modules_to_test = unittest.defaultTestLoader.discover(start_dir="tests/",
                                                          pattern="*.py",
                                                          top_level_dir=None)
    HTMLTestRunner(output="tmp/test-reports").run(modules_to_test)
    print(brainspy.TEST_MODE)
