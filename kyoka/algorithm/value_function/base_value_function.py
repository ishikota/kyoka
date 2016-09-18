import pickle

class BaseValueFunction(object):

  __KEY_MAIN_DATA = "key_base_value_function_main_data"
  __KEY_ADDITINAL_DATA = "key_base_value_function_additinal_data"

  def __init__(self):
    self.additinal_data_holder = {}

  def setUp(self):
    pass

  def deepcopy(self):
    return self

  def save(self, save_file_path):
    data = {
        self.__KEY_MAIN_DATA: self.provide_data_to_store(),
        self.__KEY_ADDITINAL_DATA: self.additinal_data_holder
    }
    self.__pickle_data(save_file_path, data)

  def load(self, load_file_path):
    stored_data = self.__unpickle_data(load_file_path)
    self.receive_data_to_restore(stored_data[self.__KEY_MAIN_DATA])
    self.additinal_data_holder = stored_data[self.__KEY_ADDITINAL_DATA]

  def set_additinal_data(self, key, data):
    self.additinal_data_holder[key] = data

  def get_additinal_data(self, key):
    if self.additinal_data_holder.has_key(key):
      return self.additinal_data_holder[key]
    else:
      return None

  def provide_data_to_store(self):
    err_msg = self.__build_err_msg("provide_data_to_store")
    raise NotImplementedError(err_msg)

  def receive_data_to_restore(self, ):
    err_msg = self.__build_err_msg("receive_data_to_restore")
    raise NotImplementedError(err_msg)

  def __pickle_data(self, filepath, data):
    with open(filepath, "wb") as f:
      pickle.dump(data, f)

  def __unpickle_data(self, filepath):
    with open(filepath, "rb") as f:
      return pickle.load(f)

  def __build_err_msg(self, msg):
    base_msg = "save method called but [ {0} ] class does not implement [ {1} ] method"
    return base_msg.format(self.__class__.__name__, msg)

