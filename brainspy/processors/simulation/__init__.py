"""
This package contains the files necessary for handling the 
software processors that handle internal differences within the 
main Processor class.
Instead of having a driver, it is composed of a deep neural 
network model (with frozen weights). The SoftwareProcessor module 
expects input data that is already plateaued. SoftwareProcessors do 
not require any sort of ramping in the inputs, as this would reduce
the overall performance in computing the outputs. However, plateaued
data is allowed as it might help with simulating noise effects, 
providing a more accurate output than that without noise. For faster
measurements, where noise simulation is not needed, a plateau of 
lenght 1 is recommended. Apart from the waveform difference, the 
SoftwareProcessor also applies to the output several effects such 
as the amplification correction of the device, clipping values, 
or relevant noise simulations. The hardware output is expected to
be in nano-Amperes. Therefore, in order to be able to read it, 
it is amplified. The surrogate model can add an amplification 
correction factor so that the original output in nano-Amperes 
is received.
See https://raw.githubusercontent.com/BraiNEdarwin/brains-py/master/docs/figures/processor.jpg
for more information.
"""