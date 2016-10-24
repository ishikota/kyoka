from kyoka.value_function.base_value_function import BaseValueFunction

class BaseStateValueFunction(BaseValueFunction):

  def calculate_value(self, state):
    err_msg = self.__build_err_msg("calculate_value")
    raise NotImplementedError(err_msg)

  def update_function(self, state, new_value):
    err_msg = self.__build_err_msg("update_function")
    raise NotImplementedError(err_msg)

  def setUp(self):
    pass


  def __build_err_msg(self, msg):
    base_msg = "[ {0} ] class does not implement [ {1} ] method"
    return base_msg.format(self.__class__.__name__, msg)

