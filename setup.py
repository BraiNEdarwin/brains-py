import platform
from setuptools import setup, find_packages

requires = ["nidaqmx", "pyyaml", "Pyro4","tqdm","tensorboard","unittest2","html-testRunner"]

if platform.system() == 'Windows':
    requires.append('pypiwin32')

setup(
    name="brains-py",
    version="0.8.2",
    description="A python package to support the study of Dopant Network Processing Units as hardware accelerators for non-linear operations. Its aim is to support key functions for hardware setups and algorithms related to searching functionality on DNPUs and DNPU architectures both in simulations and in hardware.  ",
    url="https://github.com/BraiNEdarwin/brains-py",
    author="Unai Alegre-Ibarra, and Hans-Christian Ruiz Euler. Other collaborations are acknowledged in https://github.com/BraiNEdarwin/brains-py",
    author_email="u.alegre@utwente.nl",
    license="GPL-3.0",
    packages=find_packages(),
    install_requires=requires,
    zip_safe=False,
)
