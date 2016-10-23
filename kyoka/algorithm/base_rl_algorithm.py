class BaseRLAlgorithm(object):

  def __init__(self):
    self.callbacks = []

  def update_value_function(self, domain, policy, value_function):
    err_msg = self.__build_err_msg("update_value_function")
    raise NotImplementedError(err_msg)

  def GPI(self, domain, policy, value_function, finish_rules, debug=False):
    iteration_counter = 0
    [callback.before_gpi_start(domain, value_function) for callback in self.callbacks]
    while True:
      [callback.before_update(iteration_counter, domain, value_function) for callback in self.callbacks]
      self.update_value_function(domain, policy, value_function)
      [callback.after_update(iteration_counter, domain, value_function) for callback in self.callbacks]
      iteration_counter += 1
      for finish_rule in self.__wrap_rule_if_single(finish_rules):
        if finish_rule.satisfy_condition(iteration_counter):
          finish_msg = finish_rule.generate_finish_message(iteration_counter)
          [callback.after_gpi_finish(domain, value_function) for callback in self.callbacks]
          return finish_msg

  def generate_episode(self, domain, value_function, policy):
    state = domain.generate_initial_state()
    episode = []
    while not domain.is_terminal_state(state):
      action = policy.choose_action(domain, value_function, state)
      next_state = domain.transit_state(state, action)
      reward = domain.calculate_reward(next_state)
      episode.append((state, action, next_state, reward))
      state = next_state
    return episode

  def set_gpi_callback(self, callback):
    self.callbacks.append(callback)


  def __wrap_rule_if_single(self, finish_rule):
    return [finish_rule] if not isinstance(finish_rule, list) else finish_rule

  def __build_err_msg(self, msg):
    return "Accessed [ {0} ] method of BaseRLAlgorithm which should be overridden".format(msg)

