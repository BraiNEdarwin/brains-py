
# brains-py #

A python package to support the study of Dopant Network Processing Units [1][2] as hardware accelerators for non-linear operations. Its aim is to support key functions for hardware setups and algorithms related to searching functionality on DNPUs and DNPU architectures both in simulations and in hardware.  The package is part of the brains-py project, a set of python libraries to support the development of nano-scale in-materio hardware neural-network accelerators.

 *   [![Tools](https://img.shields.io/badge/brainspy-smg-darkblue.svg)](https://github.com/BraiNEdarwin/brainspy-smg): Library for creating surrogate models of materials.
 *  [![Theory](https://img.shields.io/badge/brainspy-tasks-lightblue.svg)](https://github.com/BraiNEdarwin/brainspy-tasks) An example library on the usage of brains-py for simple tasks such as VC-Dimension or the Ring classifier.

![Insert image](https://raw.githubusercontent.com/BraiNEdarwin/brains-py/master/doc/figures/packages.png)


## 1. General description ##
### 1.1 DNPUs ###
The basis of a DNPU is a lightly doped (n- or p-type) semiconductor with a nano-scale active region contacted by several electrodes [2]. Different materials can be used as dopant or host and the number of electrodes can vary. Once we choose a readout electrode, the device can be activated by applying voltages to the remaining electrodes, which we call activation electrodes. The dopants in the active region form an atomic-scale network through which the electrons can hopfrom one electrode to another. This physical process results in an output current at the readout that depends non-linearly on the voltages applied at the activation electrodes. By tuning the voltagesapplied to some of the electrodes, the output current can be controlled as a function of the voltages at the remaining electrodes. This tunability can be exploited to solve various linearly non-separable classification tasks.

![Insert Image](https://raw.githubusercontent.com/BraiNEdarwin/brains-py/master/doc/figures/dnpu.png)

### 1.2 Training DNPUs ###
This package supports the exploration of DNPUs and DNPU architectures in the following flavours:

**Genetic Algorithm**: In computer science and operations research, a genetic algorithm (GA) is a meta-heuristic inspired by the process of natural selection that belongs to the larger class of evolutionary algorithms (EA). Genetic algorithms are commonly used to generate high-quality solutions to optimization and search problems by relying on bio-inspired operators such as mutation, crossover and selection. This algorithm is suitable for experiments with reservoir computing. [4]

**Gradient Descent**: Gradient descent is a first-order iterative optimization algorithm for finding the minimum of a function. To find a local minimum of a function using gradient descent, one takes steps proportional to the negative of the gradient (or approximate gradient) of the function at the current point. If, instead, one takes steps proportional to the positive of the gradient, one approaches a local maximum of that function; the procedure is then known as gradient ascent [5].

**On-chip training** : The on-chip training enables to look for adequate control voltages for a task directly on the DNPU or on the DNPU architecture.

**Off-chip training** The off-chip training technique uses a Deep Neural Network to simulate the behaviour of a single DNPU [3]. This package supports to find fuctionality on surrogate models, or surrogate model architectures, and then seamlessly validate the results on hardware DNPU(s), or DNPU architectures.

### 1.3 Processors ###
The processor is the key unit for this package. It enables to seamlessly use Hardware or Software (Simulation) based  DNPUs. The structure of the processors is as follows:

* **DNPU**: is the main processing unit, which contains several control voltages declared as learnable parameters.
* **Processor**: Each DNPU contains a processor. This class enables to seamlessly handle changes of processor form Hardware to Software or viceversa. To change the processor of a DNPU simply call the hw_eval function with a new processor instance or configurations dictionary.
	* **Software processor** (SurrogateModel): It is a deep neural network with information about the  control voltage ranges, the amplification of the device and relevant noise simulations that it may have.
	* **Hardware processor**: It establishes a connection (for a single, or multiple hardware DNPUs) with one of the following National Instruments measurement devices:
		* CDAQ-to-NiDAQ
		* CDAQ-to-CDAQ
			* With a regular rack
			* With a real time rack


## 2. Installation instructions ##
The installation instructions differ depending on whether if you want to install as a developer or as a user of the library. Please follow the instructions that are most suitable for you:
* [User instructions](https://github.com/BraiNEdarwin/brains-py/blob/master/doc/USER_INSTRUCTIONS.md)
* [Developer instructions](https://github.com/BraiNEdarwin/brains-py/blob/master/doc/DEVELOPER_INSTRUCTIONS.md)

## 3. License and libraries ##
This code is released under the GNU GENERAL PUBLIC LICENSE Version 3. Click [here](https://github.com/BraiNEdarwin/brains-py/blob/master/doc/LICENSE) to see the full license.
The package relies on the following libraries:
* Pytorch
* Numpy
* Nidaqmx
* pyyaml
* Pyro4
* tqdm
* torch-optimizer
* pywin32
## 4. Bibliography

[1] Chen, T., van Gelder, J., van de Ven, B., Amitonov, S. V., de Wilde, B., Euler, H. C. R., ... & van der Wiel, W. G. (2020). Classification with a disordered dopant-atom network in silicon. _Nature_, _577_(7790), 341-345. [https://doi.org/10.1038/s41586-019-1901-0](https://doi.org/10.1038/s41586-019-1901-0)

[2] HCR Euler, U Alegre-Ibarra, B van de Ven, H Broersma, PA Bobbert and WG van der Wiel (2020). Dopant Network Processing Units: Towards Efficient Neural-network Emulators with High-capacity Nanoelectronic Nodes. [https://arxiv.org/abs/2007.12371](https://arxiv.org/abs/2007.12371)

[3] HCR Euler, MN Boon, JT Wildeboer, B van de Ven, T Chen, H Broersma, PA Bobbert, WG van der Wiel (2020). A Deep-Learning Approach to Realising Functionality in Nanoelectronic Devices.
[4] [Mitchell, Melanie (1996). An Introduction to Genetic Algorithms. Cambridge, MA: MIT Press. ISBN 9780585030944.](https://books.google.nl/books?hl=en&lr=&id=0eznlz0TF-IC&oi=fnd&pg=PP9&ots=shoJ1029Jc&sig=wZ2khjtK5Gf468MmMZ-xOxepr1M&redir_esc=y#v=onepage&q&f=false)
[5] [D. P. Bertsekas (1997). Nonlinear programming. Journal of the Operational Research Society, Valume 48, Issue 3](https://doi.org/10.1057/palgrave.jors.2600425).

## 5. Acknowledgements
This package has been created and it is maintained by the [Brains](https://www.utwente.nl/en/brains/) team of the [NanoElectronics](https://www.utwente.nl/en/eemcs/ne/) research group at the University of Twente. It has been designed and developed by:
-   **Unai Alegre-Ibarra**, [@ualegre](https://github.com/ualegre) ([u.alegre@utwente.nl](mailto:u.alegre@utwente.nl))
-   **Hans Christian Ruiz-Euler**, [@hcruiz](https://github.com/hcruiz) ([h.ruiz@utwente.nl](mailto:h.ruiz@utwente.nl))

With the contribution of:
-  **Bram van de Ven**, [@bbroo1](https://github.com/bbroo1) ([b.vandeven@utwente.nl](mailto:b.vandeven@utwente.nl)) : General improvements and testing of the different hardware drivers and devices.
- **Michel P. de Jong** [@xX-Michel-Xx](https://github.com/xX-Michel-Xx) ([m.p.dejong@utwente.nl](mailto:m.p.dejong@utwente.nl)): Testing of the package and identification of bugs.
 - **Michelangelo Barocci** [@micbar-21](https://github.com/micbar-21/).  The usage of multiple DNPUs simultaneously and the creation of an improved PCB for measuring DNPUs.
 - **Jochem Wildeboer** [@jtwild](https://github.com/jtwild/)  Nearest neighbour loss functions.
 - **Annefleur Uitzetter**: The genetic algorithm from SkyNEt.
 - **Marcus Boon**: [@Mark-Boon](https://github.com/Mark-Boon): The on-chip gradient descent.

Some of the code present in this project has been refactored from the [skynet](https://github.com/BraiNEdarwin/SkyNEt) legacy project. The original contributions to the scripts, which are the base of this project, can be found at skynet, and the authorship remains of those people who collaborated in it. Using existing scripts from skynet, a whole new structure has been designed and developed to be used as a general purpose python library.  
