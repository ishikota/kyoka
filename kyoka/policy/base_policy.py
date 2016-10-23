from kyoka.value_function.base_action_value_function import BaseActionValueFunction
from kyoka.value_function.base_state_value_function import BaseStateValueFunction

class BasePolicy(object):

  def choose_action(self, domain, value_function, state, action=None):
    err_msg = self.__build_err_msg("choose_action")
    raise NotImplementedError(err_msg)

  def pack_arguments_for_value_function(self, value_function, state, action):
    if isinstance(value_function, BaseStateValueFunction):
      return [state]
    elif isinstance(value_function, BaseActionValueFunction):
      return [state, action]
    else:
      raise ValueError("Invalid value function is set")

  def __build_err_msg(self, msg):
    base_msg = "[ {0} ] class does not implement [ {1} ] method"
    return base_msg.format(self.__class__.__name__, msg)


