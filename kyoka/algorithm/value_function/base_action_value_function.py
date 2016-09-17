class BaseActionValueFunction(object):

  def calculate_action_value(self, state, action):
    err_msg = self.__build_err_msg("calculate_action_value")
    raise NotImplementedError(err_msg)

  def update_function(self, state, action, new_value):
    err_msg = self.__build_err_msg("update_function")
    raise NotImplementedError(err_msg)

  def setUp(self):
    pass

  def state_to_unique_value(self, state):
    return state

  def action_to_unique_value(self, action):
    return action


  def __build_err_msg(self, msg):
    base_msg = "[ {0} ] class does not implement [ {1} ] method"
    return base_msg.format(self.__class__.__name__, msg)

