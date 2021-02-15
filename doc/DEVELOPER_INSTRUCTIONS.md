
## Developer instructions ##
This code contains useful libraries that are expected to be used and maintained by different researchers and students at [University of Twente](https://www.utwente.nl/en/). If you would like to collaborate on the development of this or any projects of the [Brains Research Group](https://www.utwente.nl/en/brains/), you will be required to create a GitHub account. GitHub is based on the Git distributed version control system, if you are not familiar with Git, you can have a look at their [documentation](https://git-scm.com/).  If Git commands feel daunting, you are also welcome to use the graphical user interface provided by [GitHub Desktop](https://desktop.github.com/).

The development will follow the Github fork-and-pull workflow. You can have a look at how it works [here](https://reflectoring.io/github-fork-and-pull/).  Feel free to create your own fork of the repository. Make sure you [set this repository as upstream](https://help.github.com/en/articles/configuring-a-remote-for-a-fork), in order to be able to pull the latest changes of this repository to your own fork by [syncing](https://help.github.com/en/articles/syncing-a-fork). Pull requests on the original project will be accepted after maintainers have revised them. The code in this repository follows the [PEP8](https://www.python.org/dev/peps/pep-0008/) python coding style guide. Please make sure that your own fork is synced with this repository, and that it respects the PEP8 coding style.

## Installation instructions
**0. Prerequisite installs:**

[Download](https://www.anaconda.com/download) Anaconda (with python3 version) for your Operating System, and install it following the official instructions:

-   [Linux](https://docs.continuum.io/anaconda/install/linux/)
-   [Windows](https://docs.continuum.io/anaconda/install/windows/)
-   [Mac](https://docs.continuum.io/anaconda/install/mac-os/)

Depending on your needs, you can install one or more of  [Clone](https://word2md.com/C:%5CData%5CSoftware%5CClone) the following repositories from https://github.com/BraiNEdarwin:

* [https://github.com/BraiNEdarwin/brains-py.git](https://github.com/BraiNEdarwin/brains-py.git)
* [https://github.com/BraiNEdarwin/brainspy-tasks.git](https://github.com/BraiNEdarwin/brainspy-tasks.git)
* [https://github.com/BraiNEdarwin/brainspy-smg.git](https://github.com/BraiNEdarwin/brainspy-smg.git)

If your system has a [GPU that supports CUDA](https://developer.nvidia.com/cuda-gpus), install CUDA on your system first. As of the time of writing (October 2020), Pytorch supports CUDA 10.2. We recommend that you install this version: [https://developer.nvidia.com/cuda-10.2-download-archive](https://developer.nvidia.com/cuda-10.2-download-archive). For some systems the latest version of CUDA (11.1) is also known to work, you can try this on your own risk: [https://developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads).

Below, you can find the instructions to install the environment. Please note that it is important to follow the order. Pytorch should be installed before brainspy, or numpy-related installation errors might arise. Note that pytorch installation is different depending on the system you are using. Make sure you follow the latest instructions for installation on [pytorch's official website](https://pytorch.org/). 

**1. Create** an environment (choose name, e.g. "bspy"):

``conda create -n bspy python``

**2. Activate** the environment:

``activate bspy``

**3.** install **pytorch** (depends on system, see [link](https://pytorch.org/get-started/locally/)), for example:

``conda install pytorch torchvision cpuonly -c pytorch``

To check whether CUDA works properly with Pytorch, launch Python and input the following:

&gt;&gt;&gt; ``import torch``
&gt;&gt;&gt; ``print(torch.cuda.is_available())``

It should return "True" if CUDA support is available.

**4.** go to the directory " **brains-py**", in the directory tree where you cloned the repository. Run the setup file:

``python setup.py develop``

**5.** (Optional) If you require to install brainspy-tasks, go to the directory " **brainspy-tasks**" . Run the setup file:

``python setup.py develop``

**6.** (Optional) If you require to install brainspy-smg, go to the directory " **brainspy-smg**". Run the setup file:

``python setup.py develop``


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
	 * Anaconda Extension Pack (Microsoft)
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
