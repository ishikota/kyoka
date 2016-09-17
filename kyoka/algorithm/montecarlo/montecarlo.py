from kyoka.algorithm.base_rl_algorithm import BaseRLAlgorithm

class MonteCarlo(BaseRLAlgorithm):

  def update_value_function(self, domain, policy, value_function_):
    value_function = value_function_.deepcopy()
    episode = self.generate_episode(domain, policy)
    for idx, turn_info in enumerate(episode):
      state, action, _next_state, _reward = turn_info
      Q_val_info = value_function.calculate_value(state, action)
      following_reward = self.__calculate_following_state_reward(idx, episode)
      new_Q_val_info = self.__calculate_new_Q_value(Q_val_info, following_reward)
      value_function.update_function(state, action, new_Q_val_info)
    return value_function


  def __calculate_following_state_reward(self, current_turn, episode):
    following_turn_info = episode[current_turn:]
    following_rewards = [reward for _, _, _, reward in following_turn_info]
    return sum(following_rewards)

  def __calculate_new_Q_value(self, Q_val_info, update_reward):
    update_count, Q_val_average = Q_val_info
    new_Q_val = self.__calc_average_in_incremental_way(\
        update_count, update_reward, Q_val_average)
    return (update_count + 1, new_Q_val)

  def __calc_average_in_incremental_way(self, k, r, Q):
    return Q + 1.0 / (k + 1) * (r - Q)

