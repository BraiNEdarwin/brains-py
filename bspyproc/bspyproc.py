import bspyproc.processors.processor_mgr as processor_mgr
import bspyproc.architectures.architecture_mgr as architecture_mgr


def get_processor(configs):
    if configs['architecture'] == 'single_device':
        return processor_mgr.get_processor(configs)
    elif configs['architecture'] == 'multi_device':
        return processor_mgr.get_processor(configs)   
    elif configs['architecture'] == 'device_architecture':
        return architecture_mgr.get_architecture(configs)
    else:
        raise NotImplementedError(f"Architecture {configs['architecture']} is not recognised. The architecture has to be either 'single_device' or 'device_architecture'")
