import pickle

class BaseValueFunction(object):

  def setUp(self):
    pass

  def deepcopy(self):
    return self

  def save(self, save_file_path):
    data = self.provide_data_to_store()
    self.__pickle_data(save_file_path, data)

  def load(self, load_file_path):
    stored_data = self.__unpickle_data(load_file_path)
    self.receive_data_to_restore(stored_data)

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

