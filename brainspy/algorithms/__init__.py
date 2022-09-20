"""
It provides different default algorithms for brains-py, that already add several 
features that are particular to dopant-networks. There are two main flavours of algorithms: 
Genetic Algorithm and Gradient Descent. Both algorithms can be executed seamlessly by importing 
and calling their corresponding 'train' function. For general purpose, the corresponding train 
function can be loaded from brainspy.utils.manager. For more advanced implementation, a custom
algorithm is recommended. Check the wiki for more information.
"""
__all__ = ["genetic", "gradient"]
