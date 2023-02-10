---
title: 'brains-py, A framework to support research on energy-efficient
    unconventional hardware for machine learning'
tags:
  - Dopant network processing units (DNPUs)
  - Material Learning
  - Machine Learning Hardware design
  - Efficient Computing
  - Materials Science
authors:
  - name: Unai Alegre-Ibarra
    orcid: 0000-0001-5957-7945
    equal-contrib: false
    affiliation: 1 # (Multiple affiliations must be quoted)
  - name: Hans-Christian Ruiz Euler
    equal-contrib: false # (This is how you can denote equal contributions between multiple authors)
    affiliation: 1
  - name: Humaid A.Mollah
    equal-contrib: false
    affiliation: 1 # (Multiple affiliations must be quoted)
  - name: Bozhidar P. Petrov
    equal-contrib: false # (This is how you can denote equal contributions between multiple authors)
    affiliation: 1
  - name: Srikumar S. Sastry
    equal-contrib: false
    affiliation: 1 # (Multiple affiliations must be quoted)
  - name: Marcus N. Boon
    equal-contrib: false # (This is how you can denote equal contributions between multiple authors)
    affiliation: 1
  - name: Michel P. de Jong
    equal-contrib: false
    affiliation: 1 # (Multiple affiliations must be quoted)
  - name: Mohamadreza Zolfagharinejad
    equal-contrib: false # (This is how you can denote equal contributions between multiple authors)
    affiliation: 1
  - name: Florentina M. J. Uitzetter
    corresponding: false # (This is how to denote the corresponding author)
    affiliation: 1
  - name: Bram van de Ven
    equal-contrib: false # (This is how you can denote equal contributions between multiple authors)
    affiliation: 1
  - name: António J. Sousa de Almeida
    corresponding: false # (This is how to denote the corresponding author)
    affiliation: 1
  - name: Sachin Kinge
    equal-contrib: false # (This is how you can denote equal contributions between multiple authors)
    affiliation: 1
  - name: Wilfred G. van der Wiel
    corresponding: false # (This is how to denote the corresponding author)
    affiliation: 2
affiliations:
 - name: MESA+ Institute for Nanotechnology & BRAINS Center for
    Brain-Inspired Nano Systems, University of Twente, Netherlands
   index: 1
 - name: Advanced Tech., Materials Engineering Div., Toyota Motor Europe,
    Belgium
   index: 2
date: 4 December 2022
bibliography: paper.bib

---

# Abstract

Projections about the limitations of digital computers for deep learning models are leading to a shift towards domain-specific hardware, where novel analogue components are sought after, due to their potential advantages in power consumption. This paper introduces brains-py, a generic framework to facilitate research on different sorts of disordered nano-material networks for natural and energy-efficient analogue computing. Mainly, it has been applied to the concept of dopant network processing units (DNPUs), a novel and promising CMOS-compatible nano-scale tunable system based on doped silicon with potentially very low-power consumption at the inference stage. The framework focuses on two material-learning-based approaches, for training DNPUs to compute supervised learning tasks: evolution-in-matter and surrogate models. While evolution-in-matter focuses on providing a quick exploration of newly manufactured single DNPUs, the surrogate model approach is used for the design and simulation of the interconnection between multiple DNPUs, enabling the exploration of their scalability. All simulation results can be seamlessly validated on hardware, saving time and costs associated with their reproduction. The framework is generic and can be reused for research on various materials with different design aspects, providing support for the most common tasks required for doing experiments with these novel materials.

# Statement of need

The breakthroughs of deep learning come along with high energy costs, related to the high throughput data-movement requirements for computing them. The increasing computational demands of these models, along with the projected traditional hardware limitations, are shifting the paradigm towards innovative hardware solutions, where analogue components for application-specific integrated circuits are keenly sought [@kaspar2021rise]. Dopant network processing units (DNPUs) are a novel concept consisting of a lightly doped semiconductor, edged with several electrodes, where hopping in dopant networks is the dominant charge transport mechanism [@chen2020classification] (See Figure 1). The output current of DNPUs is a non-linear function of the input voltages, which can be tuned for representing different complex functions. The process of finding adequate control voltages for a particular task is called training. Once the right voltage values are found, the same device can represent different complex functions on demand. The main advantages of these CMOS-compatible devices are the very low currents, their multi-scale tunability on a high dimensional parameter space, and their potential for extensive parallelism. Devices based on this concept enable the creation of potentially very low energy-consuming hardware for computing deep neural network (DNN) models, where each DNPU has a projected energy consumption of over 100 Tera-operations per second per Watt [@chen2020classification]. This concept is still in its infancy, and it can be developed in diverse ways. From various types of materials (host and dopants) to distinct dopant concentrations, different device dimensions or the number of electrodes, as well as different combinations of data-input, control and readout electrodes. Also, there are different ways in which DNPU-based circuits could be scaled up, and having to create hardware circuits from scratch is an arduous and time-consuming process. For this reason, this paper introduces brains-py, a generic framework to facilitate research on different sorts of disordered nano-material networks for natural and energy-efficient analogue computing. To the extent of the knowledge of the authors, there is no other similar works on the area.


#Framework description

The framework is composed of two main packages, brains-py (core) and brainspy-smg (surrogate model generator). The former package contains the whole structure for processors, managing the drivers for National Instruments devices that are connected to the DNPUs, and a set of utils functions that can be used for managing the signals, custom loss/fitness functions, linear transformations and i/o file management. The latter package contains the libraries required to prepare DNPU devices for data gathering (multiple and single IV curves), to gather information from multi-terminal devices in an efficient way [@ruiz2020deep], as well as training support for surrogate models, and consistency checks.

# Conclusions and future research lines

We introduce a framework for facilitating the characterisation of different materials that can be used for energy-efficient computations in the context of machine learning hardware development research. It supports developers during typical tasks required for the mentioned purpose, including preliminary direct measurements of devices, the gathering of data and training of surrogate models, and the possibility to seamlessly validate simulations of surrogate models in hardware. In this way, researchers can save a significant amount of energy and resources when exploring the abilities of different DNPU materials and designs for creating energy-efficient hardware. The libraries have been designed with reusability in mind.

# Projects

Tested to remove projects

# Related publications

Tested to remove related publications.

# Acknowledgements

This project has received financial support from the University of Twente, the Dutch Research Council ( HTSM grant no. 16237 and Natuurkunde Projectruimte grant no. 680-91-114 ) as well as from Toyota Motor Europe N.V. We acknowledge financial support from the HYBRAIN project funded by the European Union's Horizon Europe research and innovation programme under Grant Agreement No 101046878. This work was further funded by the Deutsche Forschungsgemeinschaft (DFG, German Research Foundation) through project 433682494 – SFB 1459.

# References
