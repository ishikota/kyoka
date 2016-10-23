from kyoka.algorithm.td_learning.base_td_method import BaseTDMethod
from kyoka.policy.greedy_policy import GreedyPolicy

class QLearning(BaseTDMethod):

  ACTION_ON_TERMINAL_FLG = "action_on_terminal"

  def __init__(self, alpha=0.1, gamma=0.9):
    BaseTDMethod.__init__(self)
    self.alpha = alpha
    self.gamma = gamma
    self.greedy_policy = GreedyPolicy()

  def update_action_value_function(self, domain, policy, value_function):
    state = domain.generate_initial_state()
    action = policy.choose_action(domain, value_function, state)
    while not domain.is_terminal_state(state):
      next_state = domain.transit_state(state, action)
      reward = domain.calculate_reward(next_state)
      next_action = self.__choose_action(domain, policy, value_function, next_state)
      greedy_action = self.__choose_action(domain, self.greedy_policy, value_function, next_state)
      new_Q_value = self.__calculate_new_Q_value(\
          value_function, state, action, next_state, greedy_action, reward)
      value_function.update_function(state, action, new_Q_value)
      state, action = next_state, next_action


  def __calculate_new_Q_value(self,\
      value_function, state, action, next_state, greedy_action, reward):
    Q_value = value_function.calculate_value(state, action)
    greedy_Q_value = self.__calculate_value(value_function, next_state, greedy_action)
    return Q_value + self.alpha * (reward + self.gamma * greedy_Q_value - Q_value)

  def __choose_action(self, domain, policy, value_function, state):
    if domain.is_terminal_state(state):
      return self.ACTION_ON_TERMINAL_FLG
    else:
      return policy.choose_action(domain, value_function, state)

  def __calculate_value(self, value_function, next_state, next_action):
    if self.ACTION_ON_TERMINAL_FLG == next_action:
      return 0
    else:
      return value_function.calculate_value(next_state, next_action)

