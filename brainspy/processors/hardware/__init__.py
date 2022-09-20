"""This package contains the files necessary for handling the hardware processors within the 
main Processor class. It is composed of a driver instance, which is a connection (for a single, 
or multiple hardware DNPUs) with one of the following National Instruments 
measurement devices, which are children of the NationalInstrumentsSetup 
class (CDAQ-to-NiDAQ or CDAQ-to-CDAQ). The HardwareProcessor module expects 
that the input data has already been plateaued, and creates (and removes) 
the ramps, required for not damaging the device. The class manages the creation 
of slopes for plateaued data before passing the data to the drivers, and it
also transforms PyTorch tensors to NumPy arrays, as this is the format that 
the drivers accept. Once it has read information from the hardware, it passes 
the data back to PyTorch tensors and it masks out the information in the 
output corresponding to the points in the ramp. It has been designed in such 
a way, that it enables to easily develop other hardware drivers if required, 
and the HardwareProcessor would behave in the exact same way."""
