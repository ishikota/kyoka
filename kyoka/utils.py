import pickle

def build_not_implemented_msg(instance, method_name):
    base_msg = "[ {0} ] class does not implement [ {1} ] method"
    return base_msg.format(instance.__class__.__name__, method_name)

def pickle_data(file_path, data):
    with open(file_path, "wb") as f: pickle.dump(data, f)

def unpickle_data(file_path):
    with open(file_path, "rb") as f: return pickle.load(f)

