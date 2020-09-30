## Developer instructions ##
This code contains useful libraries that are expected to be used and maintained by different researchers and students at [University of Twente](https://www.utwente.nl/en/). If you would like to collaborate on the development of this or any projects of the [Brains Research Group](https://www.utwente.nl/en/brains/), you will be required to create a GitHub account. GitHub is based on the Git distributed version control system, if you are not familiar with Git, you can have a look at their [documentation](https://git-scm.com/).  If Git commands feel daunting, you are also welcome to use the graphical user interface provided by [GitHub Desktop](https://desktop.github.com/).

The development will follow the Github fork-and-pull workflow. You can have a look at how it works [here](https://reflectoring.io/github-fork-and-pull/).  Feel free to create your own fork of the repository. Make sure you [set this repository as upstream](https://help.github.com/en/articles/configuring-a-remote-for-a-fork), in order to be able to pull the latest changes of this repository to your own fork by [syncing](https://help.github.com/en/articles/syncing-a-fork). Pull requests on the original project will be accepted after maintainers have revised them. The code in this repository follows the [PEP8](https://www.python.org/dev/peps/pep-0008/) python coding style guide. Please make sure that your own fork is synced with this repository, and that it respects the PEP8 coding style.

## Installation instructions
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
  * Install the environment:````python setup.py develop````
  * Packages corresponding to pytorch are not automatically installed, as they depend on the particular device. You can find pytorch installation instructions [here](https://pytorch.org/). Please follow those instructions for completing the installation of this package.

## Development environment
We recommend you to use the open source development environment of [Visual Studio Code](https://code.visualstudio.com/download) for python, which can be installed following the official [guide](https://code.visualstudio.com/docs/setup/setup-overview). For Ubuntu users, it is recommended to be installed using snap: ````sudo snap install --classic code````. We also recommend you to use an auto-formatter in order to follow PEP8. You can install several extensions that will help you with auto-formatting the code:

 * Open your conda terminal and activate the environment (if you do not have it activated already):  ````conda activate bspy-instr````
 * Install the auto-formatter packages from pip:
	 * ````pip install autopep8````
	 * ````pip install flake8````
 * From the same terminal, Open Visual Studio Code with the command: ````code````
 * Go to the extensions marketplace (Ctrl+Shift+X)
 * Install the following extensions:
	 * Python (Microsoft)
	 * Python Extension Pack (Don Jayamanne)
	 * Python Docs (Mukundan)
	 * Python-autopep8 (himanoa)
	 * cornflakes-linter (kevinglasson)
 * On Visual Studio Code, press Ctrl+Shif+P and write "Open Settings (JSON)". The configuration file should look like this:
	 * Note: If you are using windows, you should add a line that points to your flake8 installation: ````"cornflakes.linter.executablePath": "C:/Users/Your_user/AppData/Local/Continuum/anaconda3/envs/Scripts/flake8.exe"````

````
{
	"[python]": {
	  "editor.tabSize": 4,
	  "editor.insertSpaces": true,
	  "editor.formatOnSave": true
	},
	"python.jediEnabled": true,
	"editor.suggestSelection": "first",
	"vsintellicode.modify.editor.suggestSelection": "automaticallyOverrodeDefaultValue"
}
````
