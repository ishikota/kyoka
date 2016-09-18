from kyoka.algorithm.base_rl_algorithm import BaseRLAlgorithm
from kyoka.algorithm.value_function.base_table_action_value_function import BaseTableActionValueFunction
from kyoka.algorithm.value_function.base_table_state_value_function import BaseTableStateValueFunction

class MonteCarlo(BaseRLAlgorithm):

  def __init__(self):
    BaseRLAlgorithm.__init__(self)
    self.update_counter = None

  def update_value_function(self, domain, policy, value_function):
    self.__validate_value_function(value_function)
    self.__initialize_update_counter_if_needed(value_function)
    episode = self.generate_episode(domain, policy)
    delta_history = []
    for idx, turn_info in enumerate(episode):
      state, action, _next_state, _reward = turn_info
      Q_value = value_function.calculate_value(state, action)
      update_count = value_function.fetch_value_from_table(self.update_counter, state, action)
      following_reward = self.__calculate_following_state_reward(idx, episode)
      new_Q_value = self.__calculate_new_Q_value(Q_value, update_count, following_reward)
      value_function.update_table(self.update_counter, state, action, update_count + 1)
      delta = value_function.update_function(state, action, new_Q_value)
      delta_history.append(delta)
    return delta_history

  def __validate_value_function(self, value_function):
    valid_type = isinstance(value_function, BaseTableActionValueFunction) or \
        isinstance(value_function, BaseTableStateValueFunction)
    if not valid_type:
      raise TypeError(self.__build_type_error_message())

  def __build_type_error_message(self):
    return 'MonteCarlo method requires you to use "table" type action-value function.\
        (child class of [BaseTableActionValueFunction])'

  def __initialize_update_counter_if_needed(self, value_function):
    if self.update_counter is None:
      self.update_counter = value_function.generate_initial_table()

  def __calculate_following_state_reward(self, current_turn, episode):
    following_turn_info = episode[current_turn:]
    following_rewards = [reward for _, _, _, reward in following_turn_info]
    return sum(following_rewards)

  def __calculate_new_Q_value(self, Q_val_average, update_count, update_reward):
    return self.__calc_average_in_incremental_way(update_count, update_reward, Q_val_average)

  def __calc_average_in_incremental_way(self, k, r, Q):
    return Q + 1.0 / (k + 1) * (r - Q)

