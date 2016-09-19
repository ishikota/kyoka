from kyoka.algorithm.base_rl_algorithm import BaseRLAlgorithm

class Sarsa(BaseRLAlgorithm):

  def __init__(self, alpha=0.1, gamma=0.9):
    self.alpha = alpha
    self.gamma = gamma

  def update_value_function(self, domain, policy, value_function):
    state = domain.generate_initial_state()
    action = policy.choose_action(state)
    delta_history = []
    while not domain.is_terminal_state(state):
      next_state = domain.transit_state(state, action)
      reward = domain.calculate_reward(next_state)
      next_action = policy.choose_action(next_state)
      new_Q_value = self.__calculate_new_Q_value(\
          value_function, state, action, next_state, next_action, reward)
      delta = value_function.update_function(state, action, new_Q_value)
      delta_history.append(delta)
      state, action = next_state, next_action
    return delta_history


  def __calculate_new_Q_value(self,\
      value_function, state, action,next_state, next_action, reward):
    Q_value = value_function.calculate_value(state, action)
    next_Q_value = value_function.calculate_value(next_state, next_action)
    return Q_value + self.alpha * (reward + self.gamma * next_Q_value - Q_value)


