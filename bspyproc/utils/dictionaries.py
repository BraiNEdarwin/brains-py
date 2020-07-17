
def info_consistency_check(self, model_info):
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
