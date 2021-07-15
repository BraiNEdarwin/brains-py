
# brains-py #

A python package based on PyTorch and NumPy to support the study of Dopant Network Processing Units (DNPUs) [[1]]((https://doi.org/10.1038/s41586-019-1901-0))[[2]](https://arxiv.org/abs/2007.12371) as hardware accelerators for non-linear operations. Its aim is to support key functions for hardware setups and algorithms related to searching functionality on DNPUs and DNPU architectures both in simulations and in hardware.  The package is part of the brains-py project, a set of python libraries to support the development of nano-scale in-materio hardware for designing hardware accelerators for neural-network like operations. This package is based on several peer-reviewed scientific publications. 

[![Tools](https://img.shields.io/badge/brainspy-smg-darkblue.svg)](https://github.com/BraiNEdarwin/brainspy-smg)[![Theory](https://img.shields.io/badge/brainspy-tasks-lightblue.svg)](https://github.com/BraiNEdarwin/brainspy-tasks) 


## 1. Instructions ##
You can find detailed instructions for the following topics on the [wiki](https://github.com/BraiNEdarwin/brains-py/wiki):

*  [Introduction](https://github.com/BraiNEdarwin/brains-py/wiki/A.-Introduction): Provides a general description behind the background of this project project. These instructions are strongly recommended for new students joining the research group.
*  [Package description](https://github.com/BraiNEdarwin/brains-py/wiki/B.-Package-description): Gives more information on how the package is structured and the particular usage of files.
*  [Installation instructions](https://github.com/BraiNEdarwin/brains-py/wiki/C.-Installation-Instructions): How to correctly install this package
*  [User instructions and usage examples](https://github.com/BraiNEdarwin/brains-py/wiki/D.-User-Instructions-and-usage-examples): Instructions for users and examples on how brains-py can be used for different purposes
*  [Developer instructions](https://github.com/BraiNEdarwin/brains-py/wiki/E.-Developer-Instructions): Instructions for people wanting to develop brains-py

## 2. License and libraries ##

This code is released under the GNU GENERAL PUBLIC LICENSE Version 3. Click [here](https://github.com/BraiNEdarwin/brains-py/blob/master/doc/LICENSE) to see the full license.
The package relies on the following libraries:
* General support libraries:
  * PyTorch, NumPy, tensorboard, and tqdm
* Drivers and hardware setups:
  * nidaqmx, Pyro4, pypiwin32
* Configurations:
  * pyyaml
* Linting, and testing:
  * unittest, mypy, flake8, html-testRunner

## 3. Related scientific publications

[1] Chen, T., van Gelder, J., van de Ven, B., Amitonov, S. V., de Wilde, B., Euler, H. C. R., ... & van der Wiel, W. G. (2020). Classification with a disordered dopant-atom network in silicon. _Nature_, _577_(7790), 341-345. [https://doi.org/10.1038/s41586-019-1901-0](https://doi.org/10.1038/s41586-019-1901-0)

[2] HCR Euler, U Alegre-Ibarra, B van de Ven, H Broersma, PA Bobbert and WG van der Wiel (2020). Dopant Network Processing Units: Towards Efficient Neural-network Emulators with High-capacity Nanoelectronic Nodes. https://arxiv.org/abs/2007.12371](https://arxiv.org/abs/2007.12371)

[3] HCR Euler, MN Boon, JT Wildeboer, B van de Ven, T Chen, H Broersma, PA Bobbert, WG van der Wiel (2020). A Deep-Learning Approach to Realising Functionality in Nanoelectronic Devices. [https://doi.org/10.1038/s41565-020-00779-y](https://doi.org/10.1038/s41565-020-00779-y)

## 4. Acknowledgements
This package has been created and it is maintained by the [Brains](https://www.utwente.nl/en/brains/) team of the [NanoElectronics](https://www.utwente.nl/en/eemcs/ne/) research group at the University of Twente. It has been designed by:
-   **Dr. Unai Alegre-Ibarra**, [@ualegre](https://github.com/ualegre) ([u.alegre@utwente.nl](mailto:u.alegre@utwente.nl)): Design, implementation, maintenance, linting tools, testing and documentation.
-   **Dr. Hans Christian Ruiz-Euler**, [@hcruiz](https://github.com/hcruiz) ([h.ruiz@utwente.nl](mailto:h.ruiz@utwente.nl)): Design and implementation of major features both in the legacy [SkyNEt](https://github.com/BraiNEdarwin/SkyNEt) repository and in this one.

With the contribution of:
-  **Bozhidar P. Petrov**, [@bozhidar-petrov](https://github.com/bozhidar-petrov) ([b.p.petrov@student.utwente.nl](mailto:b.p.petrov@student.utwente.nl)) : Testing, identification of bugs, linting tools and documentation.
-  **Humaid A. Mollah**, [@hnm27](https://github.com/hnm27) ([h.a.mollah@student.utwente.nl](mailto:h.a.mollah@student.utwente.nl)) : Testing, identification of bugs, linting tools, and documentation.
-  **Marcus Boon**: [@Mark-Boon](https://github.com/Mark-Boon): The on-chip gradient descent. The initial structure for the CDAQ to NiDAQ drivers in the legacy [SkyNEt](https://github.com/BraiNEdarwin/SkyNEt) repository.
 -  **Annefleur Uitzetter**: The genetic algorithm as shown in the legacy [SkyNEt](https://github.com/BraiNEdarwin/SkyNEt) repository.
 -  **Michelangelo Barocci** [@micbar-21](https://github.com/micbar-21/).  The usage of multiple DNPUs simultaneously on hardware and the creation of an improved PCB for measuring DNPUs.
-  **Bram van de Ven**, [@bbroo1](https://github.com/bbroo1) ([b.vandeven@utwente.nl](mailto:b.vandeven@utwente.nl)) : General improvements and testing of the different hardware drivers and devices and documentation.
- **Dr. ir. Michel P. de Jong** [@xX-Michel-Xx](https://github.com/xX-Michel-Xx) ([m.p.dejong@utwente.nl](mailto:m.p.dejong@utwente.nl)): Testing and identification of bugs, especially on the installation procedure.
 -  **Jochem Wildeboer** [@jtwild](https://github.com/jtwild/)  Nearest neighbour loss functions.

Other minor contributions might have been added, in form of previous scripts that have been improved and restructured from [SkyNEt](https://github.com/BraiNEdarwin/SkyNEt), and the authorship remains of those people who collaborated in it. 

This project has received financial support from:

- **University of Twente**
- **Dutch Research Council** 
  - HTSM grant no. 16237
  - Natuurkunde Projectruimte grant no. 680-91-114
- **Toyota Motor Europe N.V.**

