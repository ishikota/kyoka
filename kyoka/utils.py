import pickle


def build_not_implemented_msg(instance, method_name):
    base_msg = "[ {0} ] class does not implement [ {1} ] method"
    return base_msg.format(instance.__class__.__name__, method_name)

def pickle_data(file_path, data):
    with open(file_path, "wb") as f: pickle.dump(data, f)

def unpickle_data(file_path):
    with open(file_path, "rb") as f: return pickle.load(f)

def value_function_check(algorithm_name, valid_types, value_function):
    if not any([isinstance(value_function, valid_type) for valid_type in valid_types]):
        base_err_msg = 'You passed invalid type of value function for "%s" algorithm. '+\
                '(value function for "%s" algorithm must be child class of [%s])'
        valid_type_names = " or ".join([v_type.__name__ for v_type in valid_types])
        raise TypeError(base_err_msg % (algorithm_name, algorithm_name, valid_type_names))

