from kyoka.algorithm.value_function.base_action_value_function import BaseActionValueFunction

class BaseTableActionValueFunction(BaseActionValueFunction):

  def setUp(self):
    self.table = self.generate_initial_table()

  def calculate_value(self, state, action):
    return self.fetch_value_from_table(self.table, state, action)

  def update_function(self, state, action, new_value):
    old_value = self.fetch_value_from_table(self.table, state, action)
    delta = new_value - old_value
    self.update_table(self.table, state, action, new_value)
    return delta

  def generate_initial_table(self):
    err_msg = self.__build_err_msg("generate_initial_table")
    raise NotImplementedError(err_msg)

  def fetch_value_from_table(self, table, state, action):
    err_msg = self.__build_err_msg("fetch_value_from_table")
    raise NotImplementedError(err_msg)

  def update_table(self, table, state, action, new_value):
    err_msg = self.__build_err_msg("update_table")
    raise NotImplementedError(err_msg)

  def deepcopy(self):
    return self

  def save(self, dest_file_path):
    pass

  def load(self, src_file_path):
    pass


  def __build_err_msg(self, msg):
    base_msg = "[ {0} ] class does not implement [ {1} ] method"
    return base_msg.format(self.__class__.__name__, msg)

