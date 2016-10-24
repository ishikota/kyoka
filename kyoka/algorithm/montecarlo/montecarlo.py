from kyoka.algorithm.base_rl_algorithm import BaseRLAlgorithm
from kyoka.value_function.base_action_value_function import BaseActionValueFunction
from kyoka.value_function.base_state_value_function import BaseStateValueFunction
from kyoka.value_function.base_table_action_value_function import BaseTableActionValueFunction
from kyoka.value_function.base_table_state_value_function import BaseTableStateValueFunction

import os
import pickle

class MonteCarlo(BaseRLAlgorithm):

  SAVE_FILE_NAME = "montecarlo_algorithm_state.pickle"

  def setUp(self, domain, policy, value_function):
    super(MonteCarlo, self).setUp(domain, policy, value_function)
    self.__validate_value_function(value_function)
    self.update_counter = value_function.generate_initial_table()

  def update_value_function(self, domain, policy, value_function):
    episode = self.generate_episode(domain, value_function, policy)
    for idx, turn_info in enumerate(episode):
      if isinstance(value_function, BaseActionValueFunction):
        self.__update_action_value_function(\
                domain, value_function, self.update_counter, episode, idx, turn_info)
      elif isinstance(value_function, BaseStateValueFunction):
        self.__update_state_value_function(\
                domain, value_function, self.update_counter, episode, idx, turn_info)

  def save_algorithm_state(self, save_dir_path):
    self.__pickle_data(self.__gen_save_file_path(save_dir_path), self.update_counter)

  def load_algorithm_state(self, load_dir_path):
    if not os.path.exists(self.__gen_save_file_path(load_dir_path)):
      raise IOError('The saved data of "MonteCarlo" algorithm is not found in [ %s ]'% load_dir_path)
    self.update_counter = self.__unpickle_data(self.__gen_save_file_path(load_dir_path))


  def __validate_value_function(self, value_function):
    valid_type = isinstance(value_function, BaseTableActionValueFunction) or \
        isinstance(value_function, BaseTableStateValueFunction)
    if not valid_type:
      raise TypeError(self.__build_type_error_message())

  def __build_type_error_message(self):
    return 'MonteCarlo method requires you to use "table" type function.\
        (child class of [BaseTableStateValueFunction or BaseTableActionValueFunction])'

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

  def __gen_save_file_path(self, base_dir_path):
    return os.path.join(base_dir_path, self.SAVE_FILE_NAME)

  def __pickle_data(self, file_path, data):
    with open(file_path, "wb") as f:
      pickle.dump(data, f)

  def __unpickle_data(self, file_path):
    with open(file_path, "rb") as f:
      return pickle.load(f)

