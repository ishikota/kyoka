from kyoka.policy.epsilon_greedy_policy import EpsilonGreedyPolicy
from kyoka.callback.epsilon_annealer import EpsilonAnnealer
from kyoka.callback.finish_rule.watch_iteration_count import WatchIterationCount

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

  def run_gpi(self, nb_iteration, callbacks=None, verbose=1):
    self.__check_setup_call()
    default_finish_rule = WatchIterationCount(nb_iteration, verbose)
    callbacks = self.__setup_callbacks(default_finish_rule, callbacks)
    [callback.before_gpi_start(self.domain, self.value_function) for callback in callbacks]

    iteration_counter = 1
    while True:
      [callback.before_update(iteration_counter, self.domain, self.value_function) for callback in callbacks]
      self.update_value_function(self.domain, self.policy, self.value_function)
      [callback.after_update(iteration_counter, self.domain, self.value_function) for callback in callbacks]
      for finish_rule in callbacks:
        if finish_rule.interrupt_gpi(iteration_counter, self.domain, self.value_function):
          [callback.after_gpi_finish(self.domain, self.value_function) for callback in callbacks]
          if finish_rule != default_finish_rule:
            default_finish_rule.log(default_finish_rule.generate_finish_message(iteration_counter))
          return
      iteration_counter += 1

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

  def __check_setup_call(self):
    if not all([hasattr(self, attr) for attr in ["domain", "value_function", "policy"]]):
      raise Exception('You need to call "setUp" method before calling "run_gpi" method.')

  def __setup_callbacks(self, default_finish_rule, user_callbacks):
    user_callbacks = self.__wrap_item_if_single(user_callbacks)
    default_callbacks = [default_finish_rule]
    if isinstance(self.policy, EpsilonGreedyPolicy) and self.policy.do_annealing:
      default_callbacks.append(EpsilonAnnealer(self.policy))
    return default_callbacks + user_callbacks

  def __wrap_item_if_single(self, item):
    if item is None: item = []
    if not isinstance(item, list): item = [item]
    return item

  def __build_err_msg(self, msg):
    return "Accessed [ {0} ] method of BaseRLAlgorithm which should be overridden".format(msg)

