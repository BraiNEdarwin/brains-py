def check_test_configs(test_dict):
    """
    Check if a value is present in a dict

    This is a helper function to test files which require connection
    to the hardware.
    """
    check = False
    for key, val in test_dict.items():
        if check:
            break
        if type(val) == str:
            if not val:
                check = True
                break
        if type(val) == list:
            if len(val) == 0:
                check = True
                break
        if type(val) == dict:
            check = check_test_configs(val)
    return check
