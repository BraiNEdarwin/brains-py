import subprocess as sp
import platform
from setuptools import setup, find_packages

with open("README.md", "r") as f:
    long_description = f.read()

proceed = True
requires = [
    "torch~=1.11",
    "torchvision~=0.12",
    "matplotlib~=3.4",
    "nidaqmx~=0.6",
    "tqdm~=4.63",
    "PyYAML==6.0",
    "unittest2==1.1.0",
    "pytest==7.1.2",
    "pytest-cov==3.0.0",
    "coverage==6.4"
]

if platform.system() == "Windows":
    requires.append("pywin32==225")
    
print(requires)
setup(
    name="brainspy",
    version="1.0.0",
    description=
    "A python package to support research on different nano-scale materials for creating hardware accelerators in the context of deep neural networks.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    keywords=['pytorch', 'nano', 'material', 'hardware acceleration', 'deep neural networks'],	
    author="Unai Alegre-Ibarra et al.",
    author_email="u.alegre@utwente.nl",
    license="GPL-3.0",
    python_requires='==3.9.1',
    package_dir={'': 'brainspy'},
    install_requires=requires,
    packages=find_packages('brainspy'),
    py_modules=["brainspy"],
    url="https://github.com/BraiNEdarwin/brains-py",
    zip_safe=False,
)
