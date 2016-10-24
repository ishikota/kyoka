from kyoka.algorithm.td_learning.base_td_method import BaseTDMethod
from kyoka.algorithm.td_learning.eligibility_trace.action_eligibility_trace\
    import ActionEligibilityTrace as EligibilityTrace
from kyoka.policy.greedy_policy import GreedyPolicy

import os
import pickle

class QLambda(BaseTDMethod):

  SAVE_FILE_NAME = "qlambda_algorithm_state.pickle"
  ACTION_ON_TERMINAL_FLG = "action_on_terminal"

  def __init__(self, alpha=0.1, gamma=0.9, eligibility_trace=None):
    BaseTDMethod.__init__(self)
    self.alpha = alpha
    self.gamma = gamma
    self.greedy_policy = GreedyPolicy()
    self.trace = eligibility_trace

  def setUp(self, domain, policy, value_function):
    super(QLambda, self).setUp(domain, policy, value_function)
    if self.trace is None:
      self.trace = EligibilityTrace(EligibilityTrace.TYPE_ACCUMULATING)

  def save_algorithm_state(self, save_dir_path):
    self.__pickle_data(self.__gen_save_file_path(save_dir_path), self.trace.dump())

  def load_algorithm_state(self, load_dir_path):
    new_trace = EligibilityTrace(EligibilityTrace.TYPE_ACCUMULATING)
    trace_serial = self.__unpickle_data(self.__gen_save_file_path(load_dir_path))
    new_trace.load(trace_serial)
    self.trace = new_trace

  def update_action_value_function(self, domain, policy, value_function):
    current_state = domain.generate_initial_state()
    current_action = policy.choose_action(domain, value_function, current_state)
    while not domain.is_terminal_state(current_state):
      next_state = domain.transit_state(current_state, current_action)
      reward = domain.calculate_reward(next_state)
      next_action = self.__choose_action(domain, policy, value_function, next_state)
      greedy_action = self.__choose_action(domain, self.greedy_policy, value_function, next_state)
      delta = self.__calculate_delta(value_function,\
          current_state, current_action, next_state, greedy_action, reward)
      self.trace.update(current_state, current_action)
      for state, action, eligibility in self.trace.get_eligibilities():
        new_Q_value = self.__calculate_new_Q_value(\
            value_function, state, action, eligibility, delta)
        value_function.update_function(state, action, new_Q_value)
        self.trace.decay(state, action)
      if greedy_action != next_action:
        self.trace.clear()
      current_state, current_action = next_state, next_action

  def __calculate_delta(self,\
      value_function, state, action, next_state, greedy_action, reward):
    Q_value = value_function.calculate_value(state, action)
    greedy_Q_value = self.__calculate_value(value_function, next_state, greedy_action)
    return reward + self.gamma * greedy_Q_value - Q_value

  def __calculate_new_Q_value(self,\
      value_function, state, action, eligibility, delta):
    Q_value = value_function.calculate_value(state, action)
    return Q_value + self.alpha * delta * eligibility

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

  def __gen_save_file_path(self, base_dir_path):
    return os.path.join(base_dir_path, self.SAVE_FILE_NAME)

  def __pickle_data(self, file_path, data):
    with open(file_path, "wb") as f:
      pickle.dump(data, f)

  def __unpickle_data(self, file_path):
    with open(file_path, "rb") as f:
      return pickle.load(f)

