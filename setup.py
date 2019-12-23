from setuptools import setup, find_packages

setup(name='brainspy-processors',
      version='0.0.0',
      description='A python package created and maintained by the Brains team of the NanoElectronics group at the University of Twente for managing different hardware boron-doped silicon chip processors and their simulations. ',
      url='https://github.com/BraiNEdarwin/brainspy-instruments',
      author='This has adopted part of the BRAINS skynet repository code, which has been cleaned and refactored. The maintainers of the code are Unai Alegre-Ibarra, and Hans-Christian Ruiz Euler.',
      author_email='b.vandeven@utwente.nl',
      license='GPL-3.0',
      packages=find_packages(),
      install_requires=[
      'numpy',
	  'nidaqmx',
	  'Pyro4'
      ],
      python_requires='~=3.6',
      zip_safe=False)
