from kyoka.algorithm.value_function.base_action_value_function import BaseActionValueFunction

class BaseTableActionValueFunction(BaseActionValueFunction):

  def generate_initial_Q_table(self):
    err_msg = self.__build_err_msg("calculate_action_value")
    raise NotImplementedError(err_msg)


  def setUp(self):
    self.Q_table = self.generate_initial_Q_table()

  def calculate_action_value(self, state, action):
    state_id = self.state_to_unique_value(state)
    action_id = self.action_to_unique_value(action)
    Q_value = self.Q_table[state_id][action_id]
    return Q_value

  def update_function(self, state, action, new_value):
    state_id = self.state_to_unique_value(state)
    action_id = self.action_to_unique_value(action)
    self.Q_table[state_id][action_id] = new_value


  def __build_err_msg(self, msg):
    base_msg = "[ {0} ] class does not implement [ {1} ] method"
    return base_msg.format(self.__class__.__name__, msg)

