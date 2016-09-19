class BaseRLAlgorithm(object):

  def update_value_function(self, domain, policy, value_function):
    err_msg = self.__build_err_msg("update_value_function")
    raise NotImplementedError(err_msg)

  def GPI(self, domain, policy, value_function, finish_rules, debug=False):
    iteration_counter = 0
    while True:
      delta = self.update_value_function(domain, policy, value_function)
      iteration_counter += 1
      for finish_rule in self.__wrap_rule_if_single(finish_rules):
        if finish_rule.satisfy_condition(iteration_counter, delta):
          finish_msg = finish_rule.generate_finish_message(iteration_counter, delta)
          return finish_msg

  def generate_episode(self, domain, policy):
    state = domain.generate_initial_state()
    episode = []
    while not domain.is_terminal_state(state):
      action = policy.choose_action(state)
      next_state = domain.transit_state(state, action)
      reward = domain.calculate_reward(next_state)
      episode.append((state, action, next_state, reward))
      state = next_state
    return episode


  def __wrap_rule_if_single(self, finish_rule):
    return [finish_rule] if not isinstance(finish_rule, list) else finish_rule

  def __build_err_msg(self, msg):
    return "Accessed [ {0} ] method of BaseRLAlgorithm which should be overridden".format(msg)

