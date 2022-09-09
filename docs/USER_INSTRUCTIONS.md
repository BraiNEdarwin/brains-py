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

``python setup.py install``

**5.** (Optional) If you require to install brainspy-tasks, go to the directory " **brainspy-tasks**" . Run the setup file:

``python setup.py install``

**6.** (Optional) If you require to install brainspy-smg, go to the directory " **brainspy-smg**". Run the setup file:

``python setup.py install``
