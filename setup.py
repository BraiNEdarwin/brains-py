import subprocess as sp
import platform

from setuptools import setup, find_packages

proceed = True
requires = [
    "tqdm",
    "pyyaml",
    "nidaqmx",
    "matplotlib==3.4.2",
    # Packages for linting and code formatting
    "mypy==0.910",
    "mypy-extensions==0.4.3",
    "flake8==3.9.2",
    "yapf==0.31.0",
    # Packages for testing
    "unittest2",
    "html-testRunner",
]

assert (
    "\\npytorch " in str(sp.check_output("conda list pytorch", shell=True))
    or '\\ntorch ' in str(sp.check_output("pip list", shell=True))
), (
    "\n\n**********************************************************************\n"
    + "ERROR: PYTORCH INSTALLATION NOT PRESENT" +
    "\n**********************************************************************\n"
    "The correct brains-py setup requires to have pytorch installed prior\n" +
    "to the installation of brains-py. The installation of pytorch has to\n" +
    "be done manually, as it is computer-dependent (cpu or gpu). \n" +
    #"\n\n----------------------------------------------------------------------\n"
    "\n * \rPlease follow the installation instructions for installing pytorch\n"
    +
    "before installing brains-py. More information at:\nhttps://pytorch.org/ \n"
    +
    "\n**********************************************************************\n\n"
)

if platform.system() == "Windows":
    requires.append("pypiwin32")

setup(
    name="brains-py",
    version="0.8.2",
    description=
    "A python package to support the study of Dopant Network Processing Units as hardware accelerators for non-linear operations. Its aim is to support key functions for hardware setups and algorithms related to searching functionality on DNPUs and DNPU architectures both in simulations and in hardware.",
    url="https://github.com/BraiNEdarwin/brains-py",
    author=
    "Unai Alegre-Ibarra, and Hans-Christian Ruiz Euler. Other collaborations are acknowledged in https://github.com/BraiNEdarwin/brains-py",
    author_email="u.alegre@utwente.nl",
    license="GPL-3.0",
    packages=find_packages(),
    install_requires=requires,
    zip_safe=False,
)