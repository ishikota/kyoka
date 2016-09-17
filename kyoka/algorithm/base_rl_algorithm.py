class BaseRLAlgorithm(object):

  def GPI(self, domain, policy, value_function):
    err_msg = self.__build_err_msg("GPI")
    raise NotImplementedError(err_msg)

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


  def __build_err_msg(self, msg):
    return "Accessed [ {0} ] method of BaseRLAlgorithm which should be overridden".format(msg)

