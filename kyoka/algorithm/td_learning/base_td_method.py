from kyoka.algorithm.base_rl_algorithm import BaseRLAlgorithm
from kyoka.value_function.base_action_value_function import BaseActionValueFunction
from kyoka.value_function.base_state_value_function import BaseStateValueFunction

class BaseTDMethod(BaseRLAlgorithm):

  def update_action_value_function(self, domain, policy, value_function):
    err_msg = self.__build_err_msg("update_action_value_function")
    raise NotImplementedError(err_msg)

  def update_value_function(self, domain, policy, value_function):
    self.__reject_state_value_function(value_function)
    self.update_action_value_function(domain, policy, value_function)

  def __reject_state_value_function(self, value_function):
    if not isinstance(value_function, BaseActionValueFunction):
      msg = 'TD method requires you to use "ActionValueFunction" (child class of [ BaseActionValueFunction ])'
      raise TypeError(msg)

  def __build_err_msg(self, msg):
    base_msg = "[ {0} ] class does not implement [ {1} ] method"
    return base_msg.format(self.__class__.__name__, msg)
