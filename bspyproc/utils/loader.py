
import torch
from bspyproc.utils.pytorch import TorchUtils


def load_file(data_dir, file_type):
    if file_type == 'pt':
        state_dict = torch.load(data_dir, map_location=TorchUtils.get_accelerator_type())
        info = state_dict['info']
        del state_dict['info']
        info['smg_configs'] = info_consistency_check(info['smg_configs'])
        if 'amplification' not in info['data_info']['processor'].keys():
            info['data_info']['processor']['amplification'] = 1
    elif file_type == 'json':
        state_dict = None
        # TODO: Implement loading from a json file
        raise NotImplementedError(f"Loading file from a json file in TorchModel has not been implemented yet. ")
        # info = model_info loaded from a json file
    return info, state_dict


def info_consistency_check(model_info):
    """ It checks if the model info follows the expected standards.
    If it does not follow the standards, it forces the model to
    follow them and throws an exception. """
    # if type(model_info['activation']) is str:
    #    model_info['activation'] = nn.ReLU()
    if 'D_in' not in model_info['processor']['torch_model_dict']:
        model_info['processor']['torch_model_dict']['D_in'] = 7
        print('WARNING: The model loaded does not define the input dimension as expected. Changed it to default value: 7')
    if 'D_out' not in model_info['processor']['torch_model_dict']:
        model_info['processor']['torch_model_dict']['D_out'] = 1
        print('WARNING: The model loaded does not define the output dimension as expected. Changed it to default value: %d.' % 1)
    if 'hidden_sizes' not in model_info['processor']['torch_model_dict']:
        model_info['processor']['torch_model_dict']['hidden_sizes'] = [90] * 6
        print('WARNING: The model loaded does not define the input dimension as expected. Changed it to default value: %d.' % 90)
    return model_info
