
## Configuration files

The code performs tasks for which the details are specified in configuration files, including the location(s) of (an) input file(s). Seperate sets of configuration files are used for &quot;tasks&quot; (contained in ..\brainspy-tasks\) and &quot;surrogate model generation&quot; (contained in ..\brainspy-smg\). The configuration file templates for &quot;tasks&quot; are found here:

``..\brainspy-tasks\configs``

The package supports the use of either *yaml* or *json* files, although we recommend to use *yaml*. The configuration files for &quot;surrogate model generation&quot; are found here:

``..\brainspy-smg\configs``

Below, the kind of input/information that is required from the user in each configuration file is addressed.

**1) Boolean tasks config templates : (..\brainspy-tasks\configs\boolean.yaml)**

Contains input for the tasks in:

``..\brainspy-tasks\bspytasks\boolean\tasks``

Takes additional input from other config files:

``algorithm: !include defaults/algorithms/ga/boolean.yaml``

``accuracy: !include defaults/perceptron/accuracy.yaml``

Specifies whether the performed task is a simulation or is performed on hardware:

``processor_type: "simulation" # Possible values are: simulation, simulation_debug, cdaq_to_cdaq, and cdaq_to_nidaq``

For simulations, specifies the location of the model.pt file that contains the model used for the simulation, for example:

``torch_model_dict: 'C:/Data/model.pt'``

Make sure the directory names do not include spaces or special characters, like f.e. &quot;^&quot;.

**2) Ring tasks configs templates (..\brainspy-tasks\configs\ring.yaml)**

Contains input for the tasks in:

``..\brainspy-tasks\bspytasks\ring\tasks``

It can use previously generated data for the ring task or generates a new dataset:

``load: false # If load is false, it generates a new dataset. If load is a path to the data, it loads it to the data``

Takes additional input from other config files:

``algorithm: !include defaults/algorithms/gd/ring.yaml``

``processor: !include defaults/processors/simulation.yaml``

``accuracy: !include defaults/perceptron/accuracy.yaml #Configurations for the perceptron``

The file processor: !include defaults/processors/simulation.yaml contains the location of the file containing the model, for example:

``torch_model_dict: 'C:/Data/model.pt'``

**3) Surrogate model configs templates (..\brainspy-smg\configs\training\smg\_configs\_template.yaml)**

Contains input for training a model from a set of data (e.g. measurement data).

Specifies the location of the data file, e.g.:

``postprocessed_data_path: 'C:/Data/postprocessed_data.npz'``
