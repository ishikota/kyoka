from kyoka.algorithm.td_learning.base_td_method import BaseTDMethod
from kyoka.algorithm.td_learning.eligibility_trace.action_eligibility_trace\
    import ActionEligibilityTrace as EligibilityTrace
from kyoka.policy.greedy_policy import GreedyPolicy

class QLambda(BaseTDMethod):

  __KEY_ADDITIONAL_DATA = "additinal_data_key_q_lambda_eligibility_trace"
  ACTION_ON_TERMINAL_FLG = "action_on_terminal"

  def __init__(self, alpha=0.1, gamma=0.9, eligibility_trace=None):
    BaseTDMethod.__init__(self)
    self.alpha = alpha
    self.gamma = gamma
    self.trace = eligibility_trace if eligibility_trace else self.__generate_default_trace()

  def update_action_value_function(self, domain, policy, value_function):
    self.__setup_trace(value_function)
    greedy_policy = GreedyPolicy(domain, value_function)
    current_state = domain.generate_initial_state()
    current_action = policy.choose_action(current_state)
    while not domain.is_terminal_state(current_state):
      next_state = domain.transit_state(current_state, current_action)
      reward = domain.calculate_reward(next_state)
      next_action = self.__choose_action(domain, policy, next_state)
      greedy_action = self.__choose_action(domain, greedy_policy, next_state)
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
    self.__save_trace(value_function)

  def __calculate_delta(self,\
      value_function, state, action, next_state, greedy_action, reward):
    Q_value = value_function.calculate_value(state, action)
    greedy_Q_value = self.__calculate_value(value_function, next_state, greedy_action)
    return reward + self.gamma * greedy_Q_value - Q_value

  def __calculate_new_Q_value(self,\
      value_function, state, action, eligibility, delta):
    Q_value = value_function.calculate_value(state, action)
    return Q_value + self.alpha * delta * eligibility

  def __choose_action(self, domain, policy, state):
    if domain.is_terminal_state(state):
      return self.ACTION_ON_TERMINAL_FLG
    else:
      return policy.choose_action(state)

  def __calculate_value(self, value_function, next_state, next_action):
    if self.ACTION_ON_TERMINAL_FLG == next_action:
      return 0
    else:
      return value_function.calculate_value(next_state, next_action)

  def __generate_default_trace(self):
    return EligibilityTrace(EligibilityTrace.TYPE_ACCUMULATING)

  def __setup_trace(self, value_function):
    trace_dump = value_function.get_additinal_data(self.__KEY_ADDITIONAL_DATA)
    if trace_dump:
      self.trace.load(trace_dump)

  def __save_trace(self, value_function):
    value_function.set_additinal_data(self.__KEY_ADDITIONAL_DATA, self.trace.dump())

