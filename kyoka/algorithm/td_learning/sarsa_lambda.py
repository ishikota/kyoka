from kyoka.algorithm.base_rl_algorithm import BaseRLAlgorithm
from kyoka.algorithm.td_learning.eligibility_trace.action_eligibility_trace\
    import ActionEligibilityTrace as EligibilityTrace

class SarsaLambda(BaseRLAlgorithm):

  ACTION_ON_TERMINAL_FLG = "action_on_terminal"

  def __init__(self, alpha=0.1, gamma=0.9, eligibility_trace=None):
    self.alpha = alpha
    self.gamma = gamma
    self.trace = eligibility_trace if eligibility_trace else self.__generate_default_trace()

  def update_value_function(self, domain, policy, value_function):
    current_state = domain.generate_initial_state()
    current_action = policy.choose_action(current_state)
    update_delta_history = []
    while not domain.is_terminal_state(current_state):
      next_state = domain.transit_state(current_state, current_action)
      reward = domain.calculate_reward(next_state)
      next_action = self.__choose_action(domain, policy, next_state)
      delta = self.__calculate_delta(\
          value_function, current_state, current_action, next_state, next_action, reward)
      self.trace.update(current_state, current_action)
      for state, action, eligibility in self.trace.get_eligibilities():
        new_Q_value = self.__calculate_new_Q_value(\
            value_function, state, action, eligibility, delta)
        update_delta = value_function.update_function(state, action, new_Q_value)
        update_delta_history.append(update_delta)
        self.trace.decay(state, action)
      current_state, current_action = next_state, next_action
    return update_delta_history


  def __calculate_delta(self,\
      value_function, state, action,next_state, next_action, reward):
    Q_value = value_function.calculate_value(state, action)
    next_Q_value = self.__calculate_value(value_function, next_state, next_action)
    return reward + self.gamma * next_Q_value - Q_value

  def __calculate_new_Q_value(self, value_function, state, action, eligibility, delta):
    Q_value = self.__calculate_value(value_function, state, action)
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

