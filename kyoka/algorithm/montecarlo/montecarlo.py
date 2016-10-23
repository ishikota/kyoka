from kyoka.algorithm.base_rl_algorithm import BaseRLAlgorithm
from kyoka.value_function.base_action_value_function import BaseActionValueFunction
from kyoka.value_function.base_state_value_function import BaseStateValueFunction
from kyoka.value_function.base_table_action_value_function import BaseTableActionValueFunction
from kyoka.value_function.base_table_state_value_function import BaseTableStateValueFunction

class MonteCarlo(BaseRLAlgorithm):

  __KEY_ADDITIONAL_DATA = "additinal_data_key_montecarlo_update_counter"

  def update_value_function(self, domain, policy, value_function):
    self.__validate_value_function(value_function)
    self.__initialize_update_counter_if_needed(value_function)
    update_counter = value_function.get_additinal_data(self.__KEY_ADDITIONAL_DATA)
    episode = self.generate_episode(domain, value_function, policy)
    for idx, turn_info in enumerate(episode):
      if isinstance(value_function, BaseActionValueFunction):
        self.__update_action_value_function(\
                domain, value_function, update_counter, episode, idx, turn_info)
      elif isinstance(value_function, BaseStateValueFunction):
        self.__update_state_value_function(\
                domain, value_function, update_counter, episode, idx, turn_info)
    value_function.set_additinal_data(self.__KEY_ADDITIONAL_DATA, update_counter)

  def __validate_value_function(self, value_function):
    valid_type = isinstance(value_function, BaseTableActionValueFunction) or \
        isinstance(value_function, BaseTableStateValueFunction)
    if not valid_type:
      raise TypeError(self.__build_type_error_message())

  def __build_type_error_message(self):
    return 'MonteCarlo method requires you to use "table" type function.\
        (child class of [BaseTableStateValueFunction or BaseTableActionValueFunction])'

  def __initialize_update_counter_if_needed(self, value_function):
    if value_function.get_additinal_data(self.__KEY_ADDITIONAL_DATA) is None:
      update_counter = value_function.generate_initial_table()
      value_function.set_additinal_data(self.__KEY_ADDITIONAL_DATA, update_counter)

  def __update_action_value_function(\
          self, domain, value_function, update_counter, episode, idx, turn_info):
      state, action, _next_state, _reward = turn_info
      Q_value = value_function.calculate_value(state, action)
      update_count = value_function.fetch_value_from_table(update_counter, state, action)
      following_reward = self.__calculate_following_state_reward(idx, episode)
      new_Q_value = self.__calculate_new_Q_value(Q_value, update_count, following_reward)
      value_function.update_table(update_counter, state, action, update_count + 1)
      value_function.update_function(state, action, new_Q_value)

  def __update_state_value_function(\
          self, domain, value_function, update_counter, episode, idx, turn_info):
      state, action, next_state, reward = turn_info
      Q_value = value_function.calculate_value(next_state)
      update_count = value_function.fetch_value_from_table(update_counter, next_state)
      following_reward = self.__calculate_following_state_reward(idx, episode)
      new_Q_value = self.__calculate_new_Q_value(Q_value, update_count, following_reward)
      value_function.update_table(update_counter, next_state, update_count + 1)
      value_function.update_function(next_state, new_Q_value)

  def __calculate_following_state_reward(self, current_turn, episode):
    following_turn_info = episode[current_turn:]
    following_rewards = [reward for _, _, _, reward in following_turn_info]
    return sum(following_rewards)

  def __calculate_new_Q_value(self, Q_val_average, update_count, update_reward):
    return self.__calc_average_in_incremental_way(update_count, update_reward, Q_val_average)

  def __calc_average_in_incremental_way(self, k, r, Q):
    return Q + 1.0 / (k + 1) * (r - Q)

