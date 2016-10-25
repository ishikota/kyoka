from kyoka.finish_rule.watch_iteration_count import WatchIterationCount

class BaseRLAlgorithm(object):

  def setUp(self, domain, policy, value_function):
    self.domain = domain
    self.value_function = value_function
    self.value_function.setUp()
    self.policy = policy

  def save(self, save_dir_path):
    self.value_function.save(save_dir_path)
    self.save_algorithm_state(save_dir_path)

  def load(self, load_dir_path):
    self.value_function.load(load_dir_path)
    self.load_algorithm_state(load_dir_path)

  def save_algorithm_state(self, save_dir_path):
    pass

  def load_algorithm_state(self, load_dir_path):
    pass

  def update_value_function(self, domain, policy, value_function):
    err_msg = self.__build_err_msg("update_value_function")
    raise NotImplementedError(err_msg)

  def run_gpi(self, nb_iteration, finish_rules=[], callbacks=[], verbose=1):
    callbacks = self.__wrap_item_if_single(callbacks)
    finish_rules = self.__wrap_item_if_single(finish_rules)
    finish_rules.append(WatchIterationCount(nb_iteration, log_interval=float('inf') if verbose==0 else 1))
    [finish_rule.log_start_message() for finish_rule in finish_rules]
    [callback.before_gpi_start(self.domain, self.value_function) for callback in callbacks]

    iteration_counter = 0
    while True:
      [callback.before_update(iteration_counter, self.domain, self.value_function) for callback in callbacks]
      self.update_value_function(self.domain, self.policy, self.value_function)
      [callback.after_update(iteration_counter, self.domain, self.value_function) for callback in callbacks]
      iteration_counter += 1
      for finish_rule in finish_rules:
        if finish_rule.satisfy_condition(iteration_counter):
          [callback.after_gpi_finish(self.domain, self.value_function) for callback in callbacks]
          return

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


  def __wrap_item_if_single(self, item):
    return [item] if not isinstance(item, list) else item

  def __build_err_msg(self, msg):
    return "Accessed [ {0} ] method of BaseRLAlgorithm which should be overridden".format(msg)

