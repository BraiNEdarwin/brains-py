* This repository uses the Python programming language, with [Anaconda](https://en.wikipedia.org/wiki/Anaconda_(Python_distribution)) as a package manager. In order to use this code, it is recommended to [download](https://www.anaconda.com/download) Anaconda (with python3 version) for your Operating System, and install it following the official instructions:
	* [Linux](https://docs.continuum.io/anaconda/install/linux/)
	* [Windows](https://docs.continuum.io/anaconda/install/windows/)
	* [Mac](https://docs.continuum.io/anaconda/install/mac-os/)
* The Anaconda package manager, is based on [environments](https://protostar.space/why-you-need-python-environments-and-how-to-manage-them-with-conda). In order to install the corresponding environment for the code hosted in this repository, follow these instructions:
	* [Clone](https://help.github.com/en/articles/cloning-a-repository) the repository into your computer.
	* Open the terminal in which anaconda is installed.
		* For Windows users, it might be installed as an independent terminal called Anaconda prompt.
		* For Mac and Linux users, it can be run from the regular terminal.
	* Inside the anaconda terminal, navigate to the main folder of the repository using the following commands:
		* *list directory* command: ```` ls````
		* *change directory* command: ```` cd my_folder````
		* Create an anaconda environment using the following command: ```` conda create -n bspy python ipython````
		* Install pytorch. Packages corresponding to pytorch are not automatically installed, as they depend on the particular device. You can find pytorch installation instructions [here](https://pytorch.org/). Please follow those instructions for completing the installation of this package.
	  * Install the environment:````python setup.py install````
