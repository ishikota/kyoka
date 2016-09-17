class BaseStateValueFunction(object):

  def calculate_state_value(self, state):
    err_msg = self.__build_err_msg("calculate_state_value")
    raise NotImplementedError(err_msg)

  def update_function(self, state, new_value):
    err_msg = self.__build_err_msg("update_function")
    raise NotImplementedError(err_msg)

  def setUp(self):
    pass

  def deepcopy(self):
    return self

  def save(self, dest_file_path):
    pass

  def load(self, src_file_path):
    pass


  def __build_err_msg(self, msg):
    base_msg = "[ {0} ] class does not implement [ {1} ] method"
    return base_msg.format(self.__class__.__name__, msg)

