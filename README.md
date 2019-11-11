

# brainspy-processors #

A python package to support the study of the behaviour of nano-materials by enabling to seamlessly measure through a hardware connection or an equivalent simulation of a nano-device.  This package has been created and it is maintained by the [Brains](https://www.utwente.nl/en/brains/) team of the [NanoElectronics](https://www.utwente.nl/en/eemcs/ne/) research group at the University of Twente. The package is part of the [brainspy]() project, a set of python libraries to support the development of nano-scale in-materio hardware neural-network accelerators.

This package has been been mainly used for experiments related to obtaining neural-network like in-materio computation from boron-doped silicon.

 [![Tools](https://img.shields.io/badge/brainspy-algorithms-blue.svg)](https://github.com/BraiNEdarwin/brainspy-algorithms)  [![Theory](https://img.shields.io/badge/brainspy-tasks-lightblue.svg)](https://github.com/BraiNEdarwin/brainspy-tasks)  [![Tools](https://img.shields.io/badge/brainspy-smg-darkblue.svg)](https://github.com/BraiNEdarwin/brainspy-smg)

 If you are interested in
## 1. Supported processors ##
  ### 1.1 Hardware processor ###
The aim of the hardware processor is to measure a boron-doped silicon unit, or a boron-doped silicon chip architecture.

*Missing description of the architecture for hardware processors*.

This can be done using two main setups:
   * **CDAQ to NiDAQ**
	   * Missing description and reference to the manuals
	   * Dependencies
		   * nidaqmx
   * **CDAQ to CDAQ**
	   * Missing description and reference to the manuals
	   * Dependencies
		   * nidaqmx

  ### 1.2 Simulation ###
  There are two main approaches to simulate the behaviour of a boron-doped silicon unit, or a boron doped silicon chip architecture:
  #### 1.2.1 Neural-network based simulation ####     
  One approach is to use a neural network to approximate the results of the device.

  #### 1.2.1 Kinetic Monte Carlo based simulation ####     
Another one is to use a monte carlo algorithm to analyse the internal physics of the device.

## 2. Installation instructions ##
*Note: If you want to make development contributions to the code, please follow the instructions in ()[]

* This repository uses the Python programming language, with [Anaconda](https://en.wikipedia.org/wiki/Anaconda_(Python_distribution)) as a package manager. In order to use this code, it is recommended to [download](https://www.anaconda.com/download) Anaconda (with python3 version) for your Operating System, and install it following the official instructions:
	* [Linux](https://docs.continuum.io/anaconda/install/linux/)
	* [Windows](https://docs.continuum.io/anaconda/install/windows/)
	* [Mac](https://docs.continuum.io/anaconda/install/mac-os/)
* The Anaconda package manager, is based on [environments](https://protostar.space/why-you-need-python-environments-and-how-to-manage-them-with-conda). In order to install the corresponding environment for the code hosted in this repository, follow these instructions:
	* [Clone](https://help.github.com/en/articles/cloning-a-repository) the repository into your computer.
	* Open the terminal in which anaconda is installed.
		* For Windows users, it might be installed as an independent terminal called Anaconda prompt.
		* For Mac and Linux users, it can be run from the regular terminal.
	* Inside the anaconda terminal, navigate to the main folder of the repository, in which the file [conda-env-conf.yml]() is, using the following commands:
		* *list directory* command: ```` ls````
		* *change directory* command: ```` cd my_folder````
  * Install the environment:````conda env create -f conda-env-conf.yml ````
* Whenever developing or executing the code from this repository, the corresponding environment needs to be activated from the terminal where Anaconda is installed. This is done with the following command:
 ````conda activate bspyinstr````


## 3. Developer instructions ##
This code contains useful libraries that are expected to be used and maintained by different researchers and students at [University of Twente](https://www.utwente.nl/en/). If you would like to collaborate on the development of this or any projects of the [Brains Research Group](https://www.utwente.nl/en/brains/), you will be required to create a GitHub account. GitHub is based on the Git distributed version control system, if you are not familiar with Git, you can have a look at their [documentation](https://git-scm.com/).  If Git commands feel daunting, you are also welcome to use the graphical user interface provided by [GitHub Desktop](https://desktop.github.com/).

The development will follow the Github fork-and-pull workflow. You can have a look at how it works [here](https://reflectoring.io/github-fork-and-pull/).  Feel free to create your own fork of the repository. Make sure you [set this repository as upstream](https://help.github.com/en/articles/configuring-a-remote-for-a-fork), in order to be able to pull the latest changes of this repository to your own fork by [syncing](https://help.github.com/en/articles/syncing-a-fork). Pull requests on the original project will be accepted after maintainers have revised them. The code in this repository follows the [PEP8](https://www.python.org/dev/peps/pep-0008/) python coding style guide. Please make sure that your own fork is synced with this repository, and that it respects the PEP8 coding style.


## 3.1 Development environment
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

## Authors
The code is based on similar scripts from the [skynet](https://github.com/BraiNEdarwin/SkyNEt) legacy project. The original contributions to the scripts, which are the base of this project, can be found at skynet, and the authorship remains of those people who collaborated in it. Using existing scripts from skynet, a whole new structure has been designed and developed to be used as a general purpose python library.  The code has been greatly refactored and improved, and it is currently maintained by:
-   Unai Alegre Ibarra, [@ualegre](https://github.com/ualegre) ([u.alegre@utwente.nl](mailto:u.alegre@utwente.nl))
-   Hans Christian Ruiz-Euler, [@hcruiz](https://github.com/hcruiz) ([h.ruiz@utwente.nl](mailto:h.ruiz@utwente.nl))
-   Bram van de Ven, [@bbroo1](https://github.com/bbroo1) ([b.vandeven@utwente.nl](mailto:b.vandeven@utwente.nl))
