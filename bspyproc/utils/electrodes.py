
import numpy as np
from bspyproc.bspyproc import get_processor

def load_voltage_ranges(configs):
    if configs['processor_type'] == 'dnpu' or configs['processor_type'] == 'surrogate':
        p = get_processor(configs)
        amplitude = np.array(p.info['data_info']['input_data']['amplitude'])
        offset = np.array(p.info['data_info']['input_data']['offset'])
    else:
        amplitude = np.array(configs['input_data']['amplitude'])
        offset = np.array(configs['input_data']['offset'])
    min_voltage = offset - amplitude
    max_voltage = offset + amplitude
    return min_voltage, max_voltage