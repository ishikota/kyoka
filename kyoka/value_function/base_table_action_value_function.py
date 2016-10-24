from kyoka.value_function.base_action_value_function import BaseActionValueFunction
import os
import pickle

class BaseTableActionValueFunction(BaseActionValueFunction):

  SAVE_FILE_NAME = "table_action_value_function_data.pickle"

  def setUp(self):
    self.table = self.generate_initial_table()

  def calculate_value(self, state, action):
    return self.fetch_value_from_table(self.table, state, action)

  def update_function(self, state, action, new_value):
    self.update_table(self.table, state, action, new_value)

  def save(self, save_dir_path):
    file_path = os.path.join(save_dir_path, self.SAVE_FILE_NAME)
    self.__pickle_data(file_path, self.table)

  def load(self, load_dir_path):
    file_path = os.path.join(load_dir_path, self.SAVE_FILE_NAME)
    if not os.path.exists(file_path):
      raise IOError('The saved data of "TableActionValueFunction" is not found on [ %s ]'% load_dir_path)
    self.table = self.__unpickle_data(file_path)

  def generate_initial_table(self):
    err_msg = self.__build_err_msg("generate_initial_table")
    raise NotImplementedError(err_msg)

  def fetch_value_from_table(self, table, state, action):
    err_msg = self.__build_err_msg("fetch_value_from_table")
    raise NotImplementedError(err_msg)

  def update_table(self, table, state, action, new_value):
    err_msg = self.__build_err_msg("update_table")
    raise NotImplementedError(err_msg)


  def __gen_save_file_path(self, base_dir_path):
    return os.path.join(base_dir_path, self.SAVE_FILE_NAME)

  def __pickle_data(self, file_path, data):
    with open(file_path, "wb") as f:
      pickle.dump(data, f)

  def __unpickle_data(self, file_path):
    with open(file_path, "rb") as f:
      return pickle.load(f)

  def __build_err_msg(self, msg):
    base_msg = "[ {0} ] class does not implement [ {1} ] method"
    return base_msg.format(self.__class__.__name__, msg)

